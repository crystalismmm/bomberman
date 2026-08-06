import numpy as np

try:
    from agent_code.q_learning_advanced_v4_agent.callbacks import (
        action_preserves_escape_route,
    )
except ModuleNotFoundError as exc:
    if exc.name != "agent_code.q_learning_advanced_v4_agent":
        raise

    from agent_code.q_learning_advanced_v3_agent.callbacks import (
        action_preserves_escape_route,
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


def test_free_movement_without_bombs():
    field = create_field()
    state = create_state(field, (4, 4))

    assert action_preserves_escape_route(state, "UP")


def test_wait_without_bombs():
    field = create_field()
    state = create_state(field, (4, 4))

    assert action_preserves_escape_route(state, "WAIT")


def test_movement_into_wall_is_invalid():
    field = create_field()
    field[4, 3] = -1
    state = create_state(field, (4, 4))

    assert not action_preserves_escape_route(state, "UP")


def test_movement_onto_bomb_is_invalid():
    field = create_field()
    state = create_state(
        field,
        position=(4, 4),
        bombs=[((4, 3), 3)],
    )

    assert not action_preserves_escape_route(state, "UP")


def test_movement_onto_other_agent_is_invalid():
    field = create_field()
    state = create_state(
        field,
        position=(4, 4),
        others=[("other", 0, True, (4, 3))],
    )

    assert not action_preserves_escape_route(state, "UP")


def test_wait_with_timer_zero_is_fatal():
    field = create_field()
    state = create_state(
        field,
        position=(4, 6),
        bombs=[((4, 4), 0)],
    )

    assert not action_preserves_escape_route(state, "WAIT")


def test_escape_from_timer_zero_blast():
    field = create_field()
    state = create_state(
        field,
        position=(4, 6),
        bombs=[((4, 4), 0)],
    )

    assert action_preserves_escape_route(state, "RIGHT")


def test_movement_inside_timer_zero_blast_is_fatal():
    field = create_field()
    state = create_state(
        field,
        position=(4, 6),
        bombs=[((4, 4), 0)],
    )

    assert not action_preserves_escape_route(state, "UP")


def test_movement_into_timer_zero_blast_is_fatal():
    field = create_field()
    state = create_state(
        field,
        position=(5, 5),
        bombs=[((4, 4), 0)],
    )

    assert not action_preserves_escape_route(state, "LEFT")


def test_wait_with_timer_one_and_short_escape_route():
    field = create_field()
    state = create_state(
        field,
        position=(4, 6),
        bombs=[((4, 4), 1)],
    )

    assert action_preserves_escape_route(state, "WAIT")


def test_wait_with_timer_one_and_long_escape_route():
    field = create_field()
    field[3, 6] = -1
    field[5, 6] = -1

    state = create_state(
        field,
        position=(4, 6),
        bombs=[((4, 4), 1)],
    )

    assert not action_preserves_escape_route(state, "WAIT")


def test_entering_timer_one_blast_with_escape_route():
    field = create_field()
    state = create_state(
        field,
        position=(5, 6),
        bombs=[((4, 4), 1)],
    )

    assert action_preserves_escape_route(state, "LEFT")


def test_wait_on_own_bomb_with_escape_time():
    field = create_field()
    state = create_state(
        field,
        position=(4, 4),
        bombs=[((4, 4), 3)],
    )

    assert action_preserves_escape_route(state, "WAIT")


def test_movement_into_active_explosion_is_fatal():
    field = create_field()
    explosion_map = np.zeros_like(field)
    explosion_map[4, 3] = 1

    state = create_state(
        field,
        position=(4, 4),
        explosion_map=explosion_map,
    )

    assert not action_preserves_escape_route(state, "UP")


if __name__ == "__main__":
    test_free_movement_without_bombs()
    test_wait_without_bombs()
    test_movement_into_wall_is_invalid()
    test_movement_onto_bomb_is_invalid()
    test_movement_onto_other_agent_is_invalid()

    test_wait_with_timer_zero_is_fatal()
    test_escape_from_timer_zero_blast()
    test_movement_inside_timer_zero_blast_is_fatal()
    test_movement_into_timer_zero_blast_is_fatal()

    test_wait_with_timer_one_and_short_escape_route()
    test_wait_with_timer_one_and_long_escape_route()
    test_entering_timer_one_blast_with_escape_route()
    test_wait_on_own_bomb_with_escape_time()

    test_movement_into_active_explosion_is_fatal()

    print("All action-safety tests passed.")
