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

import torch

from verl import DataProto
from verl.workers.reward_manager.extra_info import build_reward_extra_info
from verl.workers.reward_manager.naive import NaiveRewardManager


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del token_ids, skip_special_tokens
        return "decoded"


def test_build_reward_extra_info_merges_both_agent_loop_layouts_without_mutation():
    dataset_extra_info = {"dataset_field": "kept", "tool_execution_records": ["dataset"]}
    tool_extra_fields = {"tool_field": "nested", "tool_execution_records": ["nested"]}
    direct_records = [{"status": "success"}]
    non_tensor_batch = {
        "extra_info": dataset_extra_info,
        "tool_extra_fields": tool_extra_fields,
        "tool_execution_records": direct_records,
    }

    result = build_reward_extra_info(non_tensor_batch)
    result["new_reward_field"] = 1

    assert result == {
        "dataset_field": "kept",
        "tool_field": "nested",
        "tool_execution_records": direct_records,
        "new_reward_field": 1,
    }
    assert dataset_extra_info == {"dataset_field": "kept", "tool_execution_records": ["dataset"]}
    assert tool_extra_fields == {"tool_field": "nested", "tool_execution_records": ["nested"]}


def test_build_reward_extra_info_uses_nested_records_when_direct_field_is_absent():
    nested_records = [{"status": "error"}]

    result = build_reward_extra_info(
        {
            "extra_info": None,
            "tool_extra_fields": {"tool_execution_records": nested_records},
        }
    )

    assert result == {"tool_execution_records": nested_records}


def test_naive_reward_manager_passes_direct_records_and_preserves_dataset_extra_info():
    captured_extra_info = None

    def compute_score(data_source, solution_str, ground_truth, extra_info):
        nonlocal captured_extra_info
        del data_source, solution_str, ground_truth
        captured_extra_info = extra_info
        return 1.0

    dataset_extra_info = {"dataset_field": "kept"}
    records = [{"tool_name": "code_interpreter", "status": "success"}]
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[10, 11]]),
            "responses": torch.tensor([[12, 13]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        },
        non_tensors={
            "data_source": ["test"],
            "reward_model": [{"ground_truth": "answer"}],
            "extra_info": [dataset_extra_info],
            "tool_extra_fields": [{"nested_field": "available"}],
            "tool_execution_records": [records],
        },
    )
    manager = NaiveRewardManager(tokenizer=_Tokenizer(), num_examine=0, compute_score=compute_score)

    manager(data)

    assert captured_extra_info is not None
    assert captured_extra_info["dataset_field"] == "kept"
    assert captured_extra_info["nested_field"] == "available"
    assert captured_extra_info["tool_execution_records"] == records
    assert captured_extra_info["num_turns"] is None
    assert captured_extra_info["rollout_reward_scores"] == {}
    assert dataset_extra_info == {"dataset_field": "kept"}
