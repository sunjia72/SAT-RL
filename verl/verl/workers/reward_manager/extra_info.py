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

from collections.abc import Mapping
from typing import Any

TOOL_EXECUTION_RECORDS_FIELD = "tool_execution_records"
TOOL_EXTRA_FIELDS_FIELD = "tool_extra_fields"


def build_reward_extra_info(non_tensor_batch: Mapping[str, Any]) -> dict[str, Any]:
    """Build an isolated ``extra_info`` mapping for one reward request.

    Agent-loop output can expose tool metadata in two layouts: inside
    ``tool_extra_fields`` (the reward-loop layout) or directly in the sample's
    non-tensor batch. Merge both without mutating the dataset-owned
    ``extra_info`` mapping. A direct execution-record field takes precedence
    because it is the canonical per-sample output field.
    """

    dataset_extra_info = non_tensor_batch.get("extra_info")
    extra_info = dict(dataset_extra_info) if isinstance(dataset_extra_info, Mapping) else {}

    tool_extra_fields = non_tensor_batch.get(TOOL_EXTRA_FIELDS_FIELD)
    if isinstance(tool_extra_fields, Mapping):
        extra_info.update(tool_extra_fields)

    if TOOL_EXECUTION_RECORDS_FIELD in non_tensor_batch:
        extra_info[TOOL_EXECUTION_RECORDS_FIELD] = non_tensor_batch[TOOL_EXECUTION_RECORDS_FIELD]

    return extra_info
