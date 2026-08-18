"""Training callbacks for the compact tabular Nash Q-learning agent."""

from __future__ import annotations

import pickle
from typing import List

import events as e
import numpy as np

from .callbacks import (
    ACTIONS,
    BOMB_UNSAFE,
    BOMB_USELESS,
    MODEL_FILE,
    MODEL_VERSION,
    MOVE_DELTAS,
    OPPOSITE_ACTION,
    TARGET_BOMBING,
    TARGET_COIN,
    WAIT_UNNECESSARY,
    StateKey,
    analyze_state,
    danger_time_map,
    infer_opponent_action,
    most_relevant_opponent,
    q_matrix_for_state,
    reachable_positions,
    solve_zero_sum_game,
    state_to_key
)


STEP_TAKEN = "STEP_TAKEN"
MOVED_TOWARDS_TARGET = "MOVED_TOWARDS_TARGET"
MOVED_AWAY_FROM_TARGET = "MOVED_AWAY_FROM_TARGET"
UNNECESSARY_WAIT = "UNNECESSARY_WAIT"
USELESS_BOMB = "USELESS_BOMB"
UNSAFE_BOMB = "UNSAFE_BOMB"
REVERSED_DIRECTION = "REVERSED_DIRECTION"


REWARDS = {
    e.COIN_COLLECTED: 2.0,
    e.CRATE_DESTROYED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.INVALID_ACTION: -1.0,
    e.KILLED_SELF: -7.0,
    e.GOT_KILLED: -5.0,
    STEP_TAKEN: -0.01,
    MOVED_TOWARDS_TARGET: 0.10,
    MOVED_AWAY_FROM_TARGET: -0.10,
    UNNECESSARY_WAIT: -0.50,
    USELESS_BOMB: -1.50,
    UNSAFE_BOMB: -5.00,
    REVERSED_DIRECTION: -0.25,
}

LEARNING_RATE = 0.10
DISCOUNT_FACTOR = 0.90
EPSILON_START = 0.20
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995


def setup_training(self):
    """Initialize variables needed only while training."""
    self.round_reward = 0.0
    self.epsilon = EPSILON_START

    if not hasattr(self, "q_table"):
        self.q_table = {}

    if not hasattr(self, "current_round"):
        self.current_round = None

    if not hasattr(self, "previous_action"):
        self.previous_action = None

    if not hasattr(self, "state_previous_action"):
        self.state_previous_action = None

    if not hasattr(self, "last_state_key"):
        self.last_state_key = None


def shortest_path_distance(game_state: dict, start: tuple[int, int], target: tuple[int, int]) -> float:
    """Return static distance to a fixed target."""
    distances, _ = reachable_positions_from(game_state, start)

    return float(distances.get(target, np.inf))


def reachable_positions_from(game_state: dict, start: tuple[int, int]) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int] | None]]:
    """Reuse the callbacks BFS with a substituted start position."""
    name, score, bombs_left, _ = game_state["self"]

    substituted = dict(game_state)

    substituted["self"] = (name, score, bombs_left, start)

    return reachable_positions(substituted)


def add_transition_events(self, old_game_state: dict, action: str, new_game_state: dict | None, events: List[str]):
    """Add state-aligned shaping events."""
    analysis = analyze_state(old_game_state, getattr(self, "state_previous_action", None))

    target_mode, _, _, old_bomb_status, old_wait_status, _ = analysis.key

    if e.WAITED in events and old_wait_status == WAIT_UNNECESSARY:
        events.append(UNNECESSARY_WAIT)

    if e.BOMB_DROPPED in events:
        if old_bomb_status == BOMB_UNSAFE:
            events.append(UNSAFE_BOMB)
        elif old_bomb_status == BOMB_USELESS:
            events.append(USELESS_BOMB)

    previous_action = getattr(self, "state_previous_action", None)

    if (
        new_game_state is not None
        and action in MOVE_DELTAS
        and previous_action in MOVE_DELTAS
        and action == OPPOSITE_ACTION[previous_action]
        and e.INVALID_ACTION not in events
    ):
        old_position = old_game_state["self"][3]
        new_position = new_game_state["self"][3]

        old_danger = danger_time_map(old_game_state)

        new_danger = danger_time_map(new_game_state)

        if (not np.isfinite(old_danger[old_position]) and not np.isfinite(new_danger[new_position])):
            events.append(REVERSED_DIRECTION)

    if (
        new_game_state is None
        or action not in MOVE_DELTAS
        or e.INVALID_ACTION in events
        or e.COIN_COLLECTED in events
        or analysis.target_position is None
        or target_mode not in (
            TARGET_COIN,
            TARGET_BOMBING
        )
    ):
        return

    old_position = old_game_state["self"][3]
    new_position = new_game_state["self"][3]

    old_danger = danger_time_map(old_game_state)

    new_danger = danger_time_map(new_game_state)

    if (np.isfinite(old_danger[old_position]) or np.isfinite(new_danger[new_position])):
        return

    old_distance = shortest_path_distance(old_game_state, old_position, analysis.target_position)

    new_distance = shortest_path_distance(old_game_state, new_position, analysis.target_position)

    if new_distance < old_distance:
        events.append(MOVED_TOWARDS_TARGET)

    elif new_distance > old_distance:
        events.append(MOVED_AWAY_FROM_TARGET)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """Learn from a non-terminal transition."""
    events.append(STEP_TAKEN)

    add_transition_events(self, old_game_state, self_action, new_game_state, events)

    reward = reward_from_events(events)

    self.round_reward += reward

    update_nash_q_table(self, old_game_state, self_action, new_game_state, reward)

    self.logger.debug(
        "Action %s produced reward %.3f from events %s.",
        self_action,
        reward,
        events
    )


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Learn the terminal transition and persist the table."""
    events.append(STEP_TAKEN)

    add_transition_events(self, last_game_state, last_action, None, events)

    reward = reward_from_events(events)

    self.round_reward += reward

    update_nash_q_table(self, last_game_state, last_action, None, reward)

    save_q_table(self.q_table, MODEL_FILE)

    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    self.logger.info(
        "Round finished with reward %.3f, epsilon %.4f, and %d known states.",
        self.round_reward,
        self.epsilon,
        len(self.q_table)
    )

    self.round_reward = 0.0
    self.current_round = None
    self.previous_action = None
    self.state_previous_action = None
    self.last_state_key = None


def reward_from_events(events: List[str]) -> float:
    """Convert events into their scalar reward sum."""
    return sum(REWARDS.get(event, 0.0) for event in events)


def table_entry(q_table: dict[StateKey, np.ndarray], state_key: StateKey) -> np.ndarray:
    """Return a writable Nash Q-matrix."""
    if state_key not in q_table:
        q_table[state_key] = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=np.float32)

    return q_table[state_key]


def update_nash_q_table(self, old_game_state: dict, action: str, new_game_state: dict | None, reward: float):
    """Apply one tabular Nash Q-learning update."""
    old_key = getattr(self, "last_state_key", None)

    if old_key is None:
        old_key = state_to_key(old_game_state, getattr(self, "state_previous_action", None))

    assert old_key is not None

    opponent = most_relevant_opponent(old_game_state)

    if opponent is None:
        return

    opponent_name = opponent[0]

    opponent_action = infer_opponent_action(old_game_state, new_game_state, opponent_name)

    my_action_index = ACTIONS.index(action)
    opponent_action_index = ACTIONS.index(opponent_action)

    old_matrix = table_entry(self.q_table, old_key)

    old_q_value = float(old_matrix[my_action_index, opponent_action_index])

    if new_game_state is None:
        future_value = 0.0

    else:
        new_key = state_to_key(new_game_state, action)

        assert new_key is not None

        new_matrix = q_matrix_for_state(self.q_table, new_key)

        _, _, future_value = solve_zero_sum_game(new_matrix)

    td_target = reward + DISCOUNT_FACTOR * future_value

    td_error = td_target - old_q_value

    old_matrix[my_action_index, opponent_action_index] += LEARNING_RATE * td_error

    self.logger.debug(
        "Updated state %s, joint action (%s, %s): "
        "old Q = %.3f, new Q = %.3f, reward = %.3f, "
        "Nash future value = %.3f, TD target = %.3f, TD error = %.3f",
        old_key,
        action,
        opponent_action,
        old_q_value,
        old_matrix[
            my_action_index,
            opponent_action_index
        ],
        reward,
        future_value,
        td_target,
        td_error
    )


def save_q_table(q_table: dict[StateKey, np.ndarray], path: Path)-> None:
    """Atomically persist the Nash Q-table."""
    payload = {"version": MODEL_VERSION, "actions": ACTIONS, "q_table": q_table}

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("wb") as model_file:
        pickle.dump(payload, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    temporary_path.replace(path)