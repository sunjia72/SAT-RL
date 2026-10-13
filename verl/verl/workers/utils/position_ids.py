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


def flatten_nested_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Flatten nested position ids while keeping lane-first multimodal layout."""
    values = position_ids.values()
    if position_ids.dim() != 3:
        return values

    samples = position_ids.unbind()
    if not samples:
        return values

    lane_size = samples[0].shape[0]
    if any(sample.dim() != 2 or sample.shape[0] != lane_size for sample in samples):
        return values

    total_seq_len = sum(sample.shape[1] for sample in samples)
    if values.dim() == 2 and values.shape[0] == lane_size and values.shape[1] == total_seq_len:
        return values

    return torch.cat(samples, dim=1)
