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

import json

import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _tool_execution_records_for_dump


def test_tool_execution_records_are_extracted_per_sample():
    direct_records = [{"tool_name": "code_interpreter", "status": "success"}]
    data = DataProto.from_dict(
        tensors={"dummy": torch.zeros((2, 1))},
        non_tensors={
            "tool_execution_records": [direct_records, []],
            "tool_extra_fields": [
                {"tool_execution_records": [{"status": "nested-must-not-win"}]},
                {},
            ],
        },
    )

    assert _tool_execution_records_for_dump(data) == [direct_records, []]


def test_generation_dump_round_trips_nested_tool_records(tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_steps = 7
    records = [
        {
            "tool_name": "code_interpreter",
            "stage": "execute",
            "status": "error",
            "error_type": "runtime_error",
            "execution_succeeded": False,
            "output_usable": False,
        }
    ]

    trainer._dump_generations(
        inputs=["problem"],
        outputs=["answer"],
        gts=["yes"],
        scores=[0.0],
        reward_extra_infos_dict={"correctness": [0.0], "tool_execution_records": [records]},
        dump_path=str(tmp_path),
    )

    entry = json.loads((tmp_path / "7.jsonl").read_text().strip())
    assert entry["tool_execution_records"] == records
    assert entry["correctness"] == 0.0
