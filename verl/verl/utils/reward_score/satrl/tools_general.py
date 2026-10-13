from verl.utils.reward_score.satrl.tools_symbolic import compute_score as _compute_symbolic_score

async def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info={},
    tool_success_score=0.2,
    solver_usage_score=0.0,  # override default here
    **kwargs,
):
    """
    Same as tools_symbolic.compute_score, but with solver/formula rewards disabled.
    IMPORTANT: this wrapper must be async and must await the underlying async reward.
    """
    # Legacy penalty kwargs are intentionally ignored. Failed tool responses
    # are reported as diagnostics but do not reduce the reward.
    kwargs.pop("failed_tool_penalty_score", None)
    kwargs.pop("failed_tool_penalty_cap", None)
    kwargs.pop("enable_logic_equivalent", None)
    kwargs.pop("include_tool_reward_in_score", None)
    kwargs.pop("include_solver_reward_in_score", None)
    return await _compute_symbolic_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        tool_success_score=tool_success_score,
        solver_usage_score=solver_usage_score,
        enable_logic_equivalent=False,
        include_tool_reward_in_score=True,
        include_solver_reward_in_score=False,
        **kwargs,
    )
