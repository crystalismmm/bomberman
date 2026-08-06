import numpy as np

try:
    from agent_code.q_learning_advanced_v4_agent.callbacks import (
        MOVE_DELTAS,
        N_FEATURES,
        state_to_features,
    )
except ModuleNotFoundError as exc:
    if exc.name != "agent_code.q_learning_advanced_v4_agent":
        raise

    from agent_code.q_learning_advanced_v3_agent.callbacks import (
        MOVE_DELTAS,
        N_FEATURES,
        state_to_features,
    )


WAIT_SAFE_INDEX = 24
SAFE_MOVE_EXISTS_INDEX = 25
MOVE_SAFETY_START = 27
MOVE_SAFETY_END = MOVE_SAFETY_START + len(MOVE_DELTAS)


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
    bombs: list[tuple[tuple[int, int], int]] | None = None,
    others: list[tuple] | None = None,
    explosion_map: np.ndarray | None = None,
) -> dict:
    if bombs is None:
        bombs = []

    if others is None:
        others = []

    if explosion_map is None:
        explosion_map = np.zeros_like(field)

    return {
        "field": field,
        "self": ("test_agent", 0, True, position),
        "bombs": bombs,
        "others": others,
        "coins": [],
        "explosion_map": explosion_map,
    }


def get_features(state: dict) -> np.ndarray:
    features = state_to_features(state)

    assert features is not None
    assert N_FEATURES == 31
    assert features.shape == (31,)
    assert np.all(np.isfinite(features))

    return features


def safety_index(action: str) -> int:
    move_order = list(MOVE_DELTAS)
    return MOVE_SAFETY_START + move_order.index(action)


def test_feature_vector_shape():
    field = create_field()
    state = create_state(field, (4, 4))

    get_features(state)


def test_all_actions_safe_without_bombs():
    field = create_field()
    state = create_state(field, (4, 4))
    features = get_features(state)

    assert features[WAIT_SAFE_INDEX] == 1
    assert features[SAFE_MOVE_EXISTS_INDEX] == 1
    assert np.all(features[MOVE_SAFETY_START:MOVE_SAFETY_END] == 1)


def test_wall_marks_only_up_as_unsafe():
    field = create_field()
    field[4, 3] = -1

    state = create_state(field, (4, 4))
    features = get_features(state)

    assert features[safety_index("UP")] == 0
    assert features[safety_index("RIGHT")] == 1
    assert features[safety_index("DOWN")] == 1
    assert features[safety_index("LEFT")] == 1
    assert features[SAFE_MOVE_EXISTS_INDEX] == 1


def test_wait_in_timer_zero_blast_is_unsafe():
    field = create_field()
    state = create_state(
        field,
        position=(4, 6),
        bombs=[((4, 4), 0)],
    )
    features = get_features(state)

    assert features[WAIT_SAFE_INDEX] == 0


def test_exactly_one_safe_escape_direction():
    field = create_field()
    field[3, 6] = -1

    state = create_state(
        field,
        position=(4, 6),
        bombs=[((4, 4), 0)],
    )
    features = get_features(state)
    movement_safety = features[
        MOVE_SAFETY_START:MOVE_SAFETY_END
    ]

    assert features[safety_index("UP")] == 0
    assert features[safety_index("RIGHT")] == 1
    assert features[safety_index("DOWN")] == 0
    assert features[safety_index("LEFT")] == 0
    assert np.sum(movement_safety) == 1
    assert features[SAFE_MOVE_EXISTS_INDEX] == 1
    assert features[WAIT_SAFE_INDEX] == 0


def test_no_safe_escape_direction():
    field = create_field()
    field[3, 6] = -1
    field[5, 6] = -1

    state = create_state(
        field,
        position=(4, 6),
        bombs=[((4, 4), 0)],
    )
    features = get_features(state)
    movement_safety = features[
        MOVE_SAFETY_START:MOVE_SAFETY_END
    ]

    assert np.sum(movement_safety) == 0
    assert features[SAFE_MOVE_EXISTS_INDEX] == 0
    assert features[WAIT_SAFE_INDEX] == 0


def test_wait_on_own_bomb_with_escape_time_is_safe():
    field = create_field()
    state = create_state(
        field,
        position=(4, 4),
        bombs=[((4, 4), 3)],
    )
    features = get_features(state)

    assert features[WAIT_SAFE_INDEX] == 1


if __name__ == "__main__":
    test_feature_vector_shape()
    test_all_actions_safe_without_bombs()
    test_wall_marks_only_up_as_unsafe()
    test_wait_in_timer_zero_blast_is_unsafe()
    test_exactly_one_safe_escape_direction()
    test_no_safe_escape_direction()
    test_wait_on_own_bomb_with_escape_time_is_safe()

    print("All action-safety feature tests passed.")
