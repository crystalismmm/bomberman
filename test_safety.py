import numpy as np

from agent_code.q_learning_advanced_agent.callbacks import (
    N_FEATURES,
    MOVE_DELTAS,
    danger_time_map,
    direction_to_bombing_position,
    direction_to_safety,
    state_to_features,
    can_escape_after_bomb,
    count_crates_in_blast,
    distance_to_nearest_bombing_position,
)


# The bombing-direction features are currently the last four features.
BOMBING_DIRECTION_START = N_FEATURES - len(MOVE_DELTAS)


def create_field(size: int = 9) -> np.ndarray:
    """Create an empty test field surrounded by walls."""
    field = np.zeros((size, size), dtype=int)

    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1

    return field


def create_state(
    field: np.ndarray,
    position: tuple[int, int],
    bombs: list[tuple[tuple[int, int], int]] | None = None,
    explosion_map: np.ndarray | None = None,
) -> dict:
    """Create a minimal game state for testing."""
    if bombs is None:
        bombs = []

    if explosion_map is None:
        explosion_map = np.zeros(field.shape, dtype=int)

    return {
        "field": field,
        "self": ("test_agent", 0, True, position),
        "bombs": bombs,
        "others": [],
        "coins": [],
        "explosion_map": explosion_map,
    }


def assert_valid_features(features: np.ndarray | None) -> np.ndarray:
    """Check the general properties of a feature vector."""
    assert features is not None
    assert features.shape == (N_FEATURES,)
    assert np.all(np.isfinite(features))

    return features


def test_already_safe():
    field = create_field()

    state = create_state(
        field,
        position=(3, 3),
        bombs=[((4, 4), 3)],
    )

    assert direction_to_safety(state) is None


def test_escape_possible():
    field = create_field()

    state = create_state(
        field,
        position=(4, 4),
        bombs=[((4, 4), 3)],
    )

    direction = direction_to_safety(state)

    assert direction in MOVE_DELTAS


def test_escape_too_late():
    field = create_field()

    state = create_state(
        field,
        position=(4, 4),
        bombs=[((4, 4), 1)],
    )

    assert direction_to_safety(state) is None


def test_agent_trapped():
    field = np.full((5, 5), -1, dtype=int)
    field[2, 2] = 0

    state = create_state(
        field,
        position=(2, 2),
        bombs=[((2, 2), 3)],
    )

    assert direction_to_safety(state) is None


def test_overlapping_bombs():
    field = create_field()

    state = create_state(
        field,
        position=(3, 3),
        bombs=[
            ((4, 4), 3),
            ((4, 2), 1),
        ],
    )

    danger_map = danger_time_map(state)

    assert danger_map[4, 3] == 1


def test_current_explosion():
    field = create_field()
    explosion_map = np.zeros(field.shape, dtype=int)
    explosion_map[3, 3] = 1

    state = create_state(
        field,
        position=(4, 4),
        explosion_map=explosion_map,
    )

    danger_map = danger_time_map(state)

    assert danger_map[3, 3] == 0


def test_safe_features():
    field = create_field()

    state = create_state(
        field,
        position=(3, 3),
    )

    features = assert_valid_features(state_to_features(state))

    assert features[13] == 0
    assert np.sum(features[14:18]) == 0
    assert features[18] == 1
    assert features[19] == 0

    # No crates means that there is no useful bombing position.
    assert np.sum(features[BOMBING_DIRECTION_START:N_FEATURES]) == 0


def test_escape_features():
    field = create_field()

    state = create_state(
        field,
        position=(4, 4),
        bombs=[((4, 4), 3)],
    )

    features = assert_valid_features(state_to_features(state))

    assert features[13] == 1
    assert np.sum(features[14:18]) == 1


def test_no_escape_features():
    field = create_field()

    state = create_state(
        field,
        position=(4, 4),
        bombs=[((4, 4), 1)],
    )

    features = assert_valid_features(state_to_features(state))

    assert features[13] == 1
    assert np.sum(features[14:18]) == 0
    assert features[18] == 0


def test_can_escape_after_bomb():
    field = create_field()

    state = create_state(
        field,
        position=(4, 4),
    )

    assert can_escape_after_bomb(state)


def test_cannot_escape_after_bomb():
    field = np.full((5, 5), -1, dtype=int)
    field[2, 2] = 0

    state = create_state(
        field,
        position=(2, 2),
    )

    assert not can_escape_after_bomb(state)


def test_no_bomb_available():
    field = create_field()

    state = create_state(
        field,
        position=(4, 4),
    )

    state["self"] = (
        "test_agent",
        0,
        False,
        (4, 4),
    )

    assert not can_escape_after_bomb(state)


def test_no_crates_in_blast():
    field = create_field()

    count = count_crates_in_blast(field, (4, 4))

    assert count == 0


def test_multiple_crates_in_blast():
    field = create_field()

    field[5, 4] = 1
    field[6, 4] = 1
    field[4, 2] = 1
    field[2, 2] = 1  # Not in the blast line

    count = count_crates_in_blast(field, (4, 4))

    state = create_state(
        field,
        position=(4, 4),
    )

    features = assert_valid_features(state_to_features(state))

    assert count == 3
    assert np.isclose(features[19], 3 / 12)


def test_wall_blocks_crate():
    field = create_field()

    field[5, 4] = -1
    field[6, 4] = 1

    count = count_crates_in_blast(field, (4, 4))

    assert count == 0


def test_current_position_is_bombing_position():
    field = create_field()
    field[7, 4] = 1

    state = create_state(
        field,
        position=(4, 4),
    )

    features = assert_valid_features(state_to_features(state))

    assert count_crates_in_blast(field, (4, 4)) == 1
    assert direction_to_bombing_position(state) is None

    # The agent is already standing at a useful bombing position.
    assert np.sum(features[BOMBING_DIRECTION_START:N_FEATURES]) == 0


def test_direction_to_bombing_position():
    field = create_field()
    field[7, 4] = 1

    state = create_state(
        field,
        position=(2, 4),
    )

    features = assert_valid_features(state_to_features(state))

    assert direction_to_bombing_position(state) == "RIGHT"

    bombing_features = features[
        BOMBING_DIRECTION_START:N_FEATURES
    ]

    right_index = list(MOVE_DELTAS).index("RIGHT")

    assert np.sum(bombing_features) == 1
    assert bombing_features[right_index] == 1


def test_distance_at_bombing_position():
    field = create_field()
    field[7, 4] = 1

    state = create_state(field, position=(4, 4))

    assert distance_to_nearest_bombing_position(state) == 0


def test_distance_to_bombing_position():
    field = create_field()
    field[7, 4] = 1

    state = create_state(field, position=(2, 4))

    assert distance_to_nearest_bombing_position(state) == 2


def test_no_bombing_position():
    field = create_field()
    state = create_state(field, position=(4, 4))

    assert distance_to_nearest_bombing_position(state) is None


if __name__ == "__main__":
    test_already_safe()
    test_escape_possible()
    test_escape_too_late()
    test_agent_trapped()
    test_overlapping_bombs()
    test_current_explosion()

    test_safe_features()
    test_escape_features()
    test_no_escape_features()

    test_can_escape_after_bomb()
    test_cannot_escape_after_bomb()
    test_no_bomb_available()

    test_no_crates_in_blast()
    test_multiple_crates_in_blast()
    test_wall_blocks_crate()

    test_current_position_is_bombing_position()
    test_direction_to_bombing_position()

    test_distance_at_bombing_position()
    test_distance_at_bombing_position()
    test_no_bombing_position()

    print("All advanced-agent tests passed.")