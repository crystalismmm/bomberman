import numpy as np

from agent_code.q_learning_advanced_v3_agent.callbacks import (
    N_FEATURES,
    best_safe_bombing_position,
    bomb_quality_at_position,
    can_escape_from_position,
    direction_to_bombing_position,
    state_to_features,
)


def create_field(size: int = 9) -> np.ndarray:
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def create_state(
    field: np.ndarray,
    position: tuple[int, int],
) -> dict:
    return {
        "field": field,
        "self": ("test_agent", 0, True, position),
        "bombs": [],
        "others": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }


def test_no_crates_means_no_bomb_target():
    field = create_field()
    state = create_state(field, (4, 4))
    features = state_to_features(state)

    assert features is not None
    assert features.shape == (N_FEATURES,)
    assert bomb_quality_at_position(state, (4, 4)) == 0
    assert best_safe_bombing_position(state) == (
        None,
        None,
        0.0,
    )
    assert features[19] == 0
    assert features[26] == 0


def test_unsafe_bombing_position_is_not_a_target():
    field = create_field(size=7)
    position = (3, 3)

    for crate_position in [
        (2, 3),
        (4, 3),
        (3, 2),
        (3, 4),
    ]:
        field[crate_position] = 1

    state = create_state(field, position)

    assert not can_escape_from_position(state, position)
    assert bomb_quality_at_position(state, position) == 0
    assert best_safe_bombing_position(state) == (
        None,
        None,
        0.0,
    )


def test_safe_current_bombing_position_has_positive_quality():
    field = create_field()
    field[7, 4] = 1
    state = create_state(field, (4, 4))
    features = state_to_features(state)

    direction, distance, score = best_safe_bombing_position(
        state
    )

    assert features is not None
    assert direction is None
    assert distance == 0
    assert score > 0
    assert features[19] > 0
    assert features[26] > 0


def test_direction_to_safe_bombing_position():
    field = create_field()
    field[7, 4] = 1
    state = create_state(field, (2, 4))

    assert direction_to_bombing_position(state) == "RIGHT"


if __name__ == "__main__":
    test_no_crates_means_no_bomb_target()
    test_unsafe_bombing_position_is_not_a_target()
    test_safe_current_bombing_position_has_positive_quality()
    test_direction_to_safe_bombing_position()
    print("All V3 safety tests passed.")
