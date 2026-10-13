# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import math
import os
from collections.abc import Mapping
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TOOL_EXECUTION_RECORDS_FIELD = "tool_execution_records"


def _to_json_safe(value: Any, seen: Optional[set[int]] = None) -> Any:
    """Convert arbitrary tool metrics to values accepted by strict JSON encoders."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return _to_json_safe(value.value, seen)

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return "<recursive>"

    if isinstance(value, Mapping):
        seen.add(value_id)
        result = {str(key): _to_json_safe(item, seen) for key, item in value.items()}
        seen.remove(value_id)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        seen.add(value_id)
        result = [_to_json_safe(item, seen) for item in value]
        seen.remove(value_id)
        return result
    if hasattr(value, "model_dump"):
        try:
            return _to_json_safe(value.model_dump(), seen)
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _to_json_safe(value.tolist(), seen)
        except Exception:
            pass

    try:
        text = str(value)
    except Exception:
        text = f"<{type(value).__name__}>"
    return text


def _exception_message(exc: BaseException) -> str:
    try:
        message = str(exc).strip()
    except Exception:
        message = ""
    return (message or type(exc).__name__)[:1000]


def _canonical_tool_error(message: str) -> ToolResponse:
    return ToolResponse(text=f"[ToolError] Error when executing tool: {message}")


def _with_tool_format_instruction(messages: list[dict[str, Any]], instruction: str | None) -> list[dict[str, Any]]:
    """Return a prompt copy containing parser-specific tool-format guidance."""
    copied_messages = [dict(message) for message in messages]
    if not instruction:
        return copied_messages

    if copied_messages and copied_messages[0].get("role") == "system":
        system_message = copied_messages[0]
        content = system_message.get("content")
        if isinstance(content, str):
            if instruction not in content:
                system_message["content"] = f"{content.rstrip()}\n\n{instruction}" if content.strip() else instruction
        elif isinstance(content, list):
            copied_content = [dict(item) if isinstance(item, Mapping) else item for item in content]
            if not any(isinstance(item, Mapping) and item.get("text") == instruction for item in copied_content):
                copied_content.append({"type": "text", "text": instruction})
            system_message["content"] = copied_content
        else:
            system_message["content"] = instruction
    else:
        copied_messages.insert(0, {"role": "system", "content": instruction})

    return copied_messages


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # Per-sample tool selection: filter global tools by extra_info.tool_selection
        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        # Keep ToolAgentLoop outputs schema-stable even when the model made no
        # tool calls in this trajectory.
        agent_data.extra_fields.setdefault(TOOL_EXECUTION_RECORDS_FIELD, [])

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        format_instruction = self.tool_parser.get_tool_format_instruction(
            [tool.tool_schema for tool in active_tools.values()]
        )
        agent_data.messages = _with_tool_format_instruction(agent_data.messages, format_instruction)
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls (use per-sample tools if routed)
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for call_index, tool_call in enumerate(agent_data.tool_calls[: self.max_parallel_calls]):
            tasks.append(
                self._call_tool(
                    tool_call,
                    agent_data.tools_kwargs,
                    agent_data,
                    call_index=call_index,
                )
            )
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, tool_metrics in responses:
            # _call_tool has already copied tool_metrics into a JSON-safe
            # execution record. Keep the named variable here to make it clear
            # that the third return value is intentionally preserved.
            del tool_metrics
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self,
        tool_call: FunctionCall,
        tools_kwargs: dict[str, Any],
        agent_data: AgentData,
        call_index: Optional[int] = None,
    ) -> tuple[ToolResponse, float, dict]:
        """Call a tool and retain a structured, JSON-safe lifecycle record."""
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tool_name = str(getattr(tool_call, "name", "") or "")
        record: dict[str, Any] = {
            "tool_name": tool_name,
            "turn_index": getattr(agent_data, "assistant_turns", None),
            "call_index": call_index,
            "stage": "parse_arguments",
            "status": "pending",
            "error_type": None,
            "error_message": "",
            "dispatch_succeeded": False,
            "instance_created": False,
            "backend_invoked": False,
            "tool_execute_invoked": False,
            "execution_succeeded": False,
            "output_usable": False,
            "tool_returned": False,
            "tool_metrics": {},
        }

        def store_record() -> None:
            safe_record = _to_json_safe(record)
            records = agent_data.extra_fields.setdefault(TOOL_EXECUTION_RECORDS_FIELD, [])
            if not isinstance(records, list):
                records = [_to_json_safe(records)]
                agent_data.extra_fields[TOOL_EXECUTION_RECORDS_FIELD] = records
            records.append(safe_record)

        def fail(stage: str, error_type: str, message: str) -> tuple[ToolResponse, float, dict]:
            record.update(
                {
                    "stage": stage,
                    "status": "error",
                    "error_type": error_type,
                    "error_message": message[:1000],
                }
            )
            store_record()
            logger.warning(f"Error when executing tool ({stage}/{error_type}): {message}")
            return _canonical_tool_error(message), 0.0, {}

        validation_errors = getattr(tool_call, "validation_errors", None) or []
        is_valid = getattr(tool_call, "is_valid", not validation_errors)
        if not is_valid or validation_errors:
            safe_errors = _to_json_safe(validation_errors)
            record["validation_errors"] = safe_errors
            error_codes = [str(error.get("code", "")) for error in safe_errors if isinstance(error, dict)]
            stage = "lookup_tool" if error_codes and error_codes[0] == "invalid_tool_name" else "parse_arguments"
            error_type = "invalid_tool_name" if stage == "lookup_tool" else "invalid_arguments"
            messages = [
                str(error.get("message", ""))
                for error in safe_errors
                if isinstance(error, dict) and error.get("message")
            ]
            message = "; ".join(messages) or "Tool call failed parser validation"
            return fail(stage, error_type, message)

        try:
            tool_args = json.loads(tool_call.arguments)
            if not isinstance(tool_args, dict):
                raise TypeError("tool arguments must decode to a JSON object")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return fail("parse_arguments", "invalid_arguments", _exception_message(exc))

        if tool_name not in active_tools:
            return fail("lookup_tool", "invalid_tool_name", f"Unknown tool name: {tool_name!r}")

        tool = active_tools[tool_name]
        record["dispatch_succeeded"] = True
        kwargs = (tools_kwargs or {}).get(tool_name, {}) or {}
        instance_id = None
        try:
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            record["instance_created"] = instance_id is not None
        except Exception as exc:
            return fail("create", "tool_creation_error", _exception_message(exc))

        try:
            record["tool_execute_invoked"] = True
            # Entering a wrapper's execute() method does not prove that its
            # remote/local backend was reached. An authoritative tool status
            # may promote this to True or False below.
            record["backend_invoked"] = None
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
            if not isinstance(tool_execution_response, ToolResponse):
                raise TypeError(
                    f"tool.execute() returned {type(tool_execution_response).__name__}; expected ToolResponse"
                )
            safe_metrics = _to_json_safe(res if res is not None else {})
            record["tool_metrics"] = safe_metrics
            record.update(
                {
                    "stage": "execute",
                    # A normal Python return only proves that the tool wrapper
                    # returned. Tools that catch backend failures internally
                    # must provide execution_status before these two fields can
                    # be treated as authoritative.
                    "status": "returned",
                    "tool_returned": True,
                    "execution_succeeded": None,
                    "output_usable": None,
                    "response_empty": tool_execution_response.is_empty(),
                }
            )

            execution_status = safe_metrics.get("execution_status") if isinstance(safe_metrics, dict) else None
            if isinstance(execution_status, dict):
                for field in (
                    "stage",
                    "status",
                    "error_type",
                    "error_message",
                    "backend_invoked",
                    "execution_succeeded",
                    "output_usable",
                ):
                    if field in execution_status:
                        record[field] = execution_status[field]
        except Exception as exc:
            message = _exception_message(exc)
            record.update(
                {
                    "stage": "execute",
                    "status": "error",
                    "error_type": "tool_execution_error",
                    "error_message": message,
                    "execution_succeeded": False,
                    "output_usable": False,
                }
            )
            tool_execution_response = _canonical_tool_error(message)
            tool_reward = 0.0
            res = {}
            logger.warning(f"Error when executing tool (execute/tool_execution_error): {message}")
        finally:
            if instance_id:
                try:
                    await tool.release(instance_id)
                except Exception as exc:
                    release_message = _exception_message(exc)
                    if record["status"] != "error":
                        record.update(
                            {
                                "stage": "release",
                                "status": "error",
                                "error_type": "tool_release_error",
                                "error_message": release_message,
                            }
                        )
                    else:
                        record["release_error"] = release_message
                    logger.warning(f"Error when releasing tool: {release_message}")

        store_record()

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map
