import numpy as np
import pytest
from agent_code.q_learning_bomb_potential_agent.callbacks import bomb_potential, best_bombing_position

def test_bomb_potential_more_crates_is_better():
    assert bomb_potential(3, 1, True) > bomb_potential(1, 1, True)

def test_bomb_potential_cannot_escape():
    assert bomb_potential(5, 1, False) == 0

def test_bomb_potential_distance_decay():
    near = bomb_potential(2, 1, True)
    far = bomb_potential(2, 5, True)

    assert near > far

def make_game_state(field, position):
    return {
        "field": field,
        "self": ("me", 0, 1, position),
        "bombs": [],
        "others": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }

def test_best_bombing_position_already_there():

    field = -np.ones((7, 7), dtype=int)

    field[1:6, 1:6] = 0

    field[3, 2] = 1
    field[3, 4] = 1

    game_state = make_game_state(field, (3, 3))

    direction, score = best_bombing_position(game_state)

    assert direction is None
    assert score > 0


def test_best_bombing_position_no_crates():

    field = -np.ones((7, 7), dtype=int)
    field[1:6, 1:6] = 0

    game_state = make_game_state(field, (3, 3))

    direction, score = best_bombing_position(game_state)

    assert direction is None
    assert score == 0.0

