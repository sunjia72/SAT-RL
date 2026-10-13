import torch

from verl.workers.utils.position_ids import flatten_nested_position_ids


def test_flatten_nested_position_ids_keeps_variable_length_lane_first_values():
    samples = [torch.arange(4 * 5).view(4, 5), torch.arange(4 * 7).view(4, 7)]
    position_ids = torch.nested.as_nested_tensor(samples, layout=torch.jagged)

    values = flatten_nested_position_ids(position_ids)

    assert values.shape == (4, 12)
    torch.testing.assert_close(values, position_ids.values())


def test_flatten_nested_position_ids_repairs_equal_length_jagged_dim_flip():
    samples = [torch.arange(4 * 5).view(4, 5), torch.arange(4 * 5, 4 * 10).view(4, 5)]
    position_ids = torch.nested.as_nested_tensor(samples, layout=torch.jagged)

    values = flatten_nested_position_ids(position_ids)

    assert values.shape == (4, 10)
    torch.testing.assert_close(values, torch.cat(samples, dim=1))


def test_flatten_nested_position_ids_handles_singleton_lane_equal_lengths():
    samples = [torch.arange(5).view(1, 5), torch.arange(5, 10).view(1, 5)]
    position_ids = torch.nested.as_nested_tensor(samples, layout=torch.jagged)

    values = flatten_nested_position_ids(position_ids)

    assert values.shape == (1, 10)
    torch.testing.assert_close(values, torch.cat(samples, dim=1))
