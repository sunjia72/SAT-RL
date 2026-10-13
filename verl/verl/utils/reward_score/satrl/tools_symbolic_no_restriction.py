from verl.utils.reward_score.satrl.tools_symbolic import compute_score as _compute_symbolic_score


async def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info={},
    tool_success_score=0.2,
    solver_usage_score=0.2,
    **kwargs,
):
    """
    Same as tools_symbolic.compute_score, but solver usage can be any supported
    symbolic backend instead of the sampled_tool stored in extra_info.
    """
    # Legacy penalty kwargs are intentionally ignored. Failed tool responses
    # are reported as diagnostics but do not reduce the reward.
    kwargs.pop("failed_tool_penalty_score", None)
    kwargs.pop("failed_tool_penalty_cap", None)
    kwargs.pop("solver_usage_requirement", None)
    return await _compute_symbolic_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        tool_success_score=tool_success_score,
        solver_usage_score=solver_usage_score,
        solver_usage_requirement="any",
        **kwargs,
    )
