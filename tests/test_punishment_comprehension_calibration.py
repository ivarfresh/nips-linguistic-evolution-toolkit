from scripts.run_punishment_comprehension_calibration import (
    FIXED_MYTH,
    RETURN_RATIOS,
    TRIALS_PER_CELL,
    VARIANTS,
    controlled_messages,
    trial_specs,
)


def test_calibration_matrix_is_balanced_and_randomized():
    specs = trial_specs()
    assert len(specs) == len(VARIANTS) * len(RETURN_RATIOS) * TRIALS_PER_CELL
    assert len({spec["trial_id"] for spec in specs}) == len(specs)
    for variant in VARIANTS:
        for return_ratio in RETURN_RATIOS:
            assert sum(
                spec["variant"] == variant
                and spec["return_ratio"] == return_ratio
                for spec in specs
            ) == TRIALS_PER_CELL
    assert [spec["call_order"] for spec in specs] == list(range(len(specs)))


def test_single_variant_matrix_supports_cross_model_check():
    specs = trial_specs(
        variants=["current"],
        trials_per_cell=3,
        order_seed=17,
    )
    assert len(specs) == len(RETURN_RATIOS) * 3
    assert {spec["variant"] for spec in specs} == {"current"}
    assert all(
        sum(spec["return_ratio"] == ratio for spec in specs) == 3
        for ratio in RETURN_RATIOS
    )


def test_custom_near_zero_return_matrix():
    ratios = [0, 1 / 60, 1 / 30, 0.05, 1 / 15]
    specs = trial_specs(
        variants=["current"],
        trials_per_cell=2,
        order_seed=23,
        return_ratios=ratios,
    )
    assert len(specs) == 10
    assert {spec["return_ratio"] for spec in specs} == set(ratios)


def test_controlled_messages_hold_state_fixed_except_return_and_wording():
    _, current_messages, current_state = controlled_messages("current", 0.5)
    _, salient_messages, salient_state = controlled_messages("cost_salient", 0.5)

    assert current_state == salient_state
    assert current_messages[2]["content"] == FIXED_MYTH
    assert current_messages[4]["content"] == '{"send": 5}'
    assert "Spending is optional" not in current_messages[0]["content"]
    assert "Spending is optional" not in current_messages[-1]["content"]
    assert "Spending is optional" in salient_messages[0]["content"]
    assert "Spending is optional" in salient_messages[-1]["content"]

    _, low_messages, low_state = controlled_messages("current", 0.0)
    assert low_state["sent"] == current_state["sent"] == 5
    assert low_state["received"] == current_state["received"] == 15
    assert low_messages[:5] == current_messages[:5]
    assert low_messages[-1] != current_messages[-1]
