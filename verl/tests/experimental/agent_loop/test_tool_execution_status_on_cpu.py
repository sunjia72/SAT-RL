# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
from types import SimpleNamespace

import pytest

import verl.tools.sandbox_fusion_tools as sandbox_fusion_module
from verl.experimental.agent_loop.tool_agent_loop import (
    TOOL_EXECUTION_RECORDS_FIELD,
    AgentState,
    ToolAgentLoop,
)
from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class _FakeTool:
    def __init__(self, *, create_error: Exception | None = None, execute_error: Exception | None = None):
        self.create_error = create_error
        self.execute_error = execute_error
        self.release_calls: list[str] = []

    async def create(self, **kwargs):
        del kwargs
        if self.create_error:
            raise self.create_error
        return "instance-1", ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        del instance_id, parameters, kwargs
        if self.execute_error:
            raise self.execute_error
        metrics = {
            "latency": float("nan"),
            "labels": {"solver", "python"},
            "execution_status": {
                "stage": "execute",
                "status": "success",
                "error_type": None,
                "backend_invoked": True,
                "execution_succeeded": True,
                "output_usable": True,
            },
        }
        return ToolResponse(text="stdout: sat\n"), 0.25, metrics

    async def release(self, instance_id):
        self.release_calls.append(instance_id)


def _loop_and_agent(tools):
    loop = object.__new__(ToolAgentLoop)
    loop.tools = tools
    loop.max_tool_response_length = 4096
    loop.tool_response_truncate_side = "left"
    agent_data = SimpleNamespace(
        _active_tools=tools,
        extra_fields={},
        assistant_turns=2,
    )
    return loop, agent_data


def test_call_tool_preserves_json_safe_metrics_and_status():
    tool = _FakeTool()
    loop, agent_data = _loop_and_agent({"code_interpreter": tool})
    tool_call = SimpleNamespace(name="code_interpreter", arguments='{"code": "print(1)"}')

    response, reward, metrics = asyncio.run(loop._call_tool(tool_call, {}, agent_data, call_index=3))

    assert response.text == "stdout: sat\n"
    assert reward == 0.25
    assert metrics["execution_status"]["execution_succeeded"] is True
    assert tool.release_calls == ["instance-1"]

    records = agent_data.extra_fields[TOOL_EXECUTION_RECORDS_FIELD]
    assert len(records) == 1
    record = records[0]
    assert record["tool_name"] == "code_interpreter"
    assert record["turn_index"] == 2
    assert record["call_index"] == 3
    assert record["dispatch_succeeded"] is True
    assert record["instance_created"] is True
    assert record["execution_succeeded"] is True
    assert record["output_usable"] is True
    assert record["tool_metrics"]["latency"] == "nan"
    assert set(record["tool_metrics"]["labels"]) == {"solver", "python"}
    json.dumps(records, allow_nan=False)


@pytest.mark.parametrize(
    ("tool_call", "expected_stage", "expected_error_type"),
    [
        (
            SimpleNamespace(
                name="bad_name",
                arguments="{}",
                is_valid=False,
                validation_errors=[{"code": "invalid_tool_name", "message": "unknown function"}],
            ),
            "lookup_tool",
            "invalid_tool_name",
        ),
        (
            SimpleNamespace(
                name="code_interpreter",
                arguments="{}",
                is_valid=False,
                validation_errors=[{"code": "missing_required_parameter", "message": "code is required"}],
            ),
            "parse_arguments",
            "invalid_arguments",
        ),
        (
            SimpleNamespace(name="code_interpreter", arguments="{"),
            "parse_arguments",
            "invalid_arguments",
        ),
        (
            SimpleNamespace(name="bad_name", arguments="{}"),
            "lookup_tool",
            "invalid_tool_name",
        ),
    ],
)
def test_call_tool_rejects_invalid_calls_before_dispatch(tool_call, expected_stage, expected_error_type):
    tool = _FakeTool()
    loop, agent_data = _loop_and_agent({"code_interpreter": tool})

    response, reward, metrics = asyncio.run(loop._call_tool(tool_call, {}, agent_data))

    assert response.text.startswith("[ToolError] Error when executing tool:")
    assert reward == 0.0
    assert metrics == {}
    assert tool.release_calls == []
    record = agent_data.extra_fields[TOOL_EXECUTION_RECORDS_FIELD][0]
    assert record["stage"] == expected_stage
    assert record["error_type"] == expected_error_type
    assert record["dispatch_succeeded"] is False
    assert record["backend_invoked"] is False


@pytest.mark.parametrize(
    ("tool", "expected_stage", "expected_error_type", "expected_release_calls"),
    [
        (_FakeTool(create_error=RuntimeError("create failed")), "create", "tool_creation_error", []),
        (
            _FakeTool(execute_error=RuntimeError("execute failed")),
            "execute",
            "tool_execution_error",
            ["instance-1"],
        ),
    ],
)
def test_call_tool_categorizes_lifecycle_exceptions(tool, expected_stage, expected_error_type, expected_release_calls):
    loop, agent_data = _loop_and_agent({"code_interpreter": tool})
    tool_call = SimpleNamespace(name="code_interpreter", arguments="{}")

    response, reward, _ = asyncio.run(loop._call_tool(tool_call, {}, agent_data))

    assert response.text.startswith("[ToolError] Error when executing tool:")
    assert reward == 0.0
    assert tool.release_calls == expected_release_calls
    record = agent_data.extra_fields[TOOL_EXECUTION_RECORDS_FIELD][0]
    assert record["stage"] == expected_stage
    assert record["error_type"] == expected_error_type
    assert record["dispatch_succeeded"] is True
    assert record["tool_execute_invoked"] is (expected_stage == "execute")
    expected_backend_invoked = None if expected_stage == "execute" else False
    assert record["backend_invoked"] is expected_backend_invoked


def test_call_tool_does_not_invent_success_for_legacy_caught_failure():
    class _CaughtFailureTool(_FakeTool):
        async def execute(self, instance_id, parameters, **kwargs):
            del instance_id, parameters, kwargs
            return ToolResponse(text="[ToolError] backend rejected the program"), 0.0, {}

    loop, agent_data = _loop_and_agent({"code_interpreter": _CaughtFailureTool()})
    tool_call = SimpleNamespace(name="code_interpreter", arguments="{}")

    response, _, _ = asyncio.run(loop._call_tool(tool_call, {}, agent_data))

    assert response.text.startswith("[ToolError]")
    record = agent_data.extra_fields[TOOL_EXECUTION_RECORDS_FIELD][0]
    assert record["status"] == "returned"
    assert record["tool_returned"] is True
    assert record["execution_succeeded"] is None
    assert record["output_usable"] is None


def test_legacy_release_failure_is_categorized_without_inventing_execute_status():
    class _ReleaseFailureTool(_FakeTool):
        async def execute(self, instance_id, parameters, **kwargs):
            del instance_id, parameters, kwargs
            return ToolResponse(text="sat"), 0.0, {}

        async def release(self, instance_id):
            self.release_calls.append(instance_id)
            raise RuntimeError("release failed")

    tool = _ReleaseFailureTool()
    loop, agent_data = _loop_and_agent({"code_interpreter": tool})

    response, _, _ = asyncio.run(
        loop._call_tool(SimpleNamespace(name="code_interpreter", arguments="{}"), {}, agent_data)
    )

    assert response.text == "sat"
    assert tool.release_calls == ["instance-1"]
    record = agent_data.extra_fields[TOOL_EXECUTION_RECORDS_FIELD][0]
    assert record["stage"] == "release"
    assert record["status"] == "error"
    assert record["error_type"] == "tool_release_error"
    assert record["execution_succeeded"] is None
    assert record["output_usable"] is None


def test_pending_state_injects_real_qwen_tool_names_without_mutating_raw_prompt():
    schema = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "Execute solver code.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    )
    tool = SimpleNamespace(tool_schema=schema)
    loop = object.__new__(ToolAgentLoop)
    loop.tools = {"code_interpreter": tool}
    loop.tool_schemas = [schema.model_dump(exclude_unset=True, exclude_none=True)]
    loop.tool_parser = Qwen3XMLToolParser(SimpleNamespace())
    captured = {}

    async def apply_chat_template(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return [1, 2, 3]

    loop.apply_chat_template = apply_chat_template
    raw_prompt = [{"role": "user", "content": "Solve this puzzle."}]
    agent_data = SimpleNamespace(
        messages=list(raw_prompt),
        image_data=None,
        video_data=None,
        _active_tools=loop.tools,
        _active_tool_schemas=loop.tool_schemas,
    )

    state = asyncio.run(loop._handle_pending_state(agent_data, {}))

    assert state is AgentState.GENERATING
    assert raw_prompt == [{"role": "user", "content": "Solve this puzzle."}]
    assert captured["messages"][0]["role"] == "system"
    instruction = captured["messages"][0]["content"]
    assert "<function=code_interpreter>" in instruction
    assert "<parameter=code>" in instruction
    assert "example_function_name" not in instruction
    assert captured["kwargs"]["tools"] == loop.tool_schemas


def _sandbox_tool_without_init():
    tool = object.__new__(SandboxFusionTool)
    tool.sandbox_fusion_url = "http://sandbox.invalid"
    tool.memory_limit_mb = 128
    tool.log_path = None
    tool._instance_dict = {}
    return tool


def test_execute_code_preserves_legacy_response_and_optionally_returns_status(monkeypatch):
    metadata = {
        "status": "success",
        "api_status": "Success",
        "compile_status": "Finished",
        "run_status": "Finished",
        "exit_code": 0,
        "stdout": "sat\n",
        "stderr": "",
    }
    monkeypatch.setattr(sandbox_fusion_module, "_process_single_case", lambda *args, **kwargs: (True, metadata))
    tool = _sandbox_tool_without_init()

    legacy_response = tool.execute_code("instance-1", "print('sat')")
    response, metrics = tool.execute_code("instance-1", "print('sat')", return_metadata=True)

    assert isinstance(legacy_response, ToolResponse)
    assert response.text == legacy_response.text
    execution_status = metrics["execution_status"]
    assert execution_status["status"] == "success"
    assert execution_status["execution_succeeded"] is True
    assert execution_status["output_usable"] is True
    assert execution_status["run_status"] == "Finished"
    assert execution_status["exit_code"] == 0


def test_execute_code_distinguishes_execution_success_from_usable_output(monkeypatch):
    metadata = {
        "status": "success",
        "api_status": "Success",
        "compile_status": "Finished",
        "run_status": "Finished",
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
    }
    monkeypatch.setattr(sandbox_fusion_module, "_process_single_case", lambda *args, **kwargs: (True, metadata))
    tool = _sandbox_tool_without_init()

    response, metrics = tool.execute_code("instance-1", "x = 1", return_metadata=True)

    assert "[ToolWarning]" in response.text
    execution_status = metrics["execution_status"]
    assert execution_status["status"] == "error"
    assert execution_status["error_type"] == "empty_output"
    assert execution_status["execution_succeeded"] is True
    assert execution_status["output_usable"] is False
