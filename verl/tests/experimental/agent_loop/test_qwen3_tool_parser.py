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

import pytest

from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema


class _TextTokenizer:
    def __init__(self, text: str = ""):
        self.text = text

    def decode(self, _response_ids):
        return self.text


@pytest.fixture
def tools() -> list[OpenAIFunctionToolSchema]:
    return [
        OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "Execute solver code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "retries": {"type": "integer"},
                            "solver": {"type": "string", "enum": ["z3", "sympy"]},
                        },
                        "required": ["code"],
                    },
                },
            }
        )
    ]


def _error_codes(call) -> set[str]:
    return {error.code for error in call.validation_errors}


def test_valid_call_uses_exact_schema_and_converts_declared_types(tools):
    parser = Qwen3XMLToolParser(_TextTokenizer())

    call = parser._parse_xml_function_call(
        "code_interpreter>\n"
        "<parameter=code>\nprint('ok')\n</parameter>\n"
        "<parameter=retries>\n2\n</parameter>\n"
        "<parameter=solver>\nz3\n</parameter>\n",
        tools,
    )

    assert call.is_valid
    assert call.name == "code_interpreter"
    assert json.loads(call.arguments) == {"code": "print('ok')", "retries": 2, "solver": "z3"}


def test_unknown_name_is_retained_and_never_fuzzy_corrected(tools):
    parser = Qwen3XMLToolParser(_TextTokenizer())

    call = parser._parse_xml_function_call(
        "function_call>\n<parameter=code>\nprint('ok')\n</parameter>\n",
        tools,
    )

    assert call.name == "function_call"
    assert not call.is_valid
    assert "invalid_tool_name" in _error_codes(call)
    assert call.raw_call.startswith("function_call>")


def test_arguments_are_validated_for_unknown_missing_type_and_enum(tools):
    parser = Qwen3XMLToolParser(_TextTokenizer())

    call = parser._parse_xml_function_call(
        "code_interpreter>\n"
        "<parameter=unknown>\nvalue\n</parameter>\n"
        "<parameter=retries>\nnot-an-integer\n</parameter>\n"
        "<parameter=solver>\nortools\n</parameter>\n",
        tools,
    )

    assert _error_codes(call) == {
        "unknown_parameter",
        "missing_required_parameter",
        "invalid_parameter_type",
        "invalid_parameter_value",
    }


def test_malformed_xml_is_returned_as_invalid_call_instead_of_disappearing(tools):
    text = "reasoning\n<tool_call>\n<function=code_interpreter>\n<parameter-code>\nprint('ok')\n</parameter>"
    parser = Qwen3XMLToolParser(_TextTokenizer(text))

    content, calls = asyncio.run(parser.extract_tool_calls([], tools))

    assert content == "reasoning\n"
    assert len(calls) == 1
    assert calls[0].name == "code_interpreter"
    assert not calls[0].is_valid
    assert "malformed_tool_call" in _error_codes(calls[0])
    assert "missing_required_parameter" in _error_codes(calls[0])


def test_empty_tool_call_is_surfaced_as_malformed(tools):
    parser = Qwen3XMLToolParser(_TextTokenizer("<tool_call></tool_call>"))

    _, calls = asyncio.run(parser.extract_tool_calls([], tools))

    assert len(calls) == 1
    assert calls[0].name == ""
    assert "malformed_tool_call" in _error_codes(calls[0])


def test_schema_specific_prompt_guidance_uses_real_names(tools):
    parser = Qwen3XMLToolParser(_TextTokenizer())

    instruction = parser.get_tool_format_instruction(tools)

    assert "<function=code_interpreter>" in instruction
    assert "<parameter=code>" in instruction
    assert 'print("hello")' in instruction
    assert "example_function_name" not in instruction
    assert "example_parameter" not in instruction
