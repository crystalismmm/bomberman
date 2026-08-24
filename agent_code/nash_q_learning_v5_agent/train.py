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
    TARGET_ESCAPE,
    WAIT_UNNECESSARY,
    WAIT_UNSAFE,
    StateKey,
    analyze_state,
    available_actions,
    danger_time_map,
    infer_opponent_action,
    policy_for_state,
    relevant_opponent,
    reachable_positions,
    state_to_key
)


STEP_TAKEN = "STEP_TAKEN"
MOVED_TOWARDS_TARGET = "MOVED_TOWARDS_TARGET"
MOVED_AWAY_FROM_TARGET = "MOVED_AWAY_FROM_TARGET"
UNNECESSARY_WAIT = "UNNECESSARY_WAIT"
USELESS_BOMB = "USELESS_BOMB"
UNSAFE_BOMB = "UNSAFE_BOMB"
UNSAFE_MOVE = "UNSAFE_MOVE"
UNSAFE_WAIT = "UNSAFE_WAIT"
ESCAPED_DANGER = "ESCAPED_DANGER"
REVERSED_DIRECTION = "REVERSED_DIRECTION"


REWARDS = {
    e.COIN_COLLECTED: 2.0,
    e.CRATE_DESTROYED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.INVALID_ACTION: -2.0,
    e.KILLED_SELF: -7.0,
    e.GOT_KILLED: -5.0,
    STEP_TAKEN: -0.01,
    MOVED_TOWARDS_TARGET: 0.10,
    MOVED_AWAY_FROM_TARGET: -0.10,
    UNNECESSARY_WAIT: -0.50,
    USELESS_BOMB: -1.50,
    UNSAFE_BOMB: -5.00,
    UNSAFE_MOVE: -2.00,
    UNSAFE_WAIT: -2.00,
    ESCAPED_DANGER: 0.75,
    REVERSED_DIRECTION: -0.25,
}

LEARNING_RATE = 0.10
DISCOUNT_FACTOR = 0.90
EPSILON_START = 0.30
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.9995


def setup_training(self):
    """Initialize training and continue the saved epsilon schedule."""
    self.round_reward = 0.0
    completed_rounds = getattr(self, "training_rounds", 0)
    self.epsilon = max(EPSILON_MIN, EPSILON_START * EPSILON_DECAY ** completed_rounds)

    for table_name in ("q_table", "visit_table", "solo_q_table", "solo_visit_table"):
        if not hasattr(self, table_name):
            setattr(self, table_name, {})

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

    target_mode, _, movement_mask, old_bomb_status, old_wait_status, _ = analysis.key

    if e.WAITED in events:
        if old_wait_status == WAIT_UNSAFE:
            events.append(UNSAFE_WAIT)
        elif old_wait_status == WAIT_UNNECESSARY:
            events.append(UNNECESSARY_WAIT)

    if e.BOMB_DROPPED in events:
        if old_bomb_status == BOMB_UNSAFE:
            events.append(UNSAFE_BOMB)
        elif old_bomb_status == BOMB_USELESS:
            events.append(USELESS_BOMB)

    if action in MOVE_DELTAS and e.INVALID_ACTION not in events:
        movement_index = list(MOVE_DELTAS).index(action)

        if not movement_mask & (1 << movement_index):
            events.append(UNSAFE_MOVE)

    if new_game_state is not None and target_mode == TARGET_ESCAPE:
        old_position = old_game_state["self"][3]
        new_position = new_game_state["self"][3]
        old_danger = danger_time_map(old_game_state)
        new_danger = danger_time_map(new_game_state)

        if np.isfinite(old_danger[old_position]) and not np.isfinite(new_danger[new_position]):
            events.append(ESCAPED_DANGER)

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

    self.training_rounds += 1
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    save_model(self, MODEL_FILE)

    self.logger.info(
        "Round %d finished with reward %.3f, epsilon %.4f, %d joint states, and %d solo states.",
        self.training_rounds,
        self.round_reward,
        self.epsilon,
        len(self.q_table),
        len(self.solo_q_table),
    )

    self.round_reward = 0.0
    self.current_round = None
    self.previous_action = None
    self.state_previous_action = None
    self.last_state_key = None


def reward_from_events(events: List[str]) -> float:
    """Convert events into their scalar reward sum."""
    return sum(REWARDS.get(event, 0.0) for event in events)


def table_entry(table: dict[StateKey, np.ndarray], state_key: StateKey, shape: tuple[int, ...], dtype) -> np.ndarray:
    """Return a writable state entry with the requested shape."""
    if state_key not in table:
        table[state_key] = np.zeros(shape, dtype=dtype)

    return table[state_key]


def update_nash_q_table(self, old_game_state: dict, action: str, new_game_state: dict | None, reward: float):
    """Update the robust solo baseline and one observed joint action."""
    old_key = getattr(self, "last_state_key", None)

    if old_key is None:
        old_key = state_to_key(old_game_state, getattr(self, "state_previous_action", None))

    assert old_key is not None

    my_action_index = ACTIONS.index(action)

    if new_game_state is None:
        solo_future_value = 0.0
        policy_future_value = 0.0

    else:
        new_analysis = analyze_state(new_game_state, action)
        allowed_indices = np.array(
            [ACTIONS.index(next_action) for next_action in available_actions(new_game_state)],
            dtype=int,
        )

        next_solo_q = self.solo_q_table.get(new_analysis.key)
        next_solo_visits = self.solo_visit_table.get(new_analysis.key)

        if next_solo_q is None or next_solo_visits is None:
            solo_future_value = 0.0
        else:
            candidate_values = np.asarray(next_solo_q[allowed_indices], dtype=np.float64).copy()
            candidate_values[next_solo_visits[allowed_indices] == 0] = 0.0
            solo_future_value = float(np.max(candidate_values))

        _, _, policy_future_value, _ = policy_for_state(self, new_game_state, new_analysis)

    solo_q = table_entry(self.solo_q_table, old_key, (len(ACTIONS),), np.float32)
    solo_visits = table_entry(self.solo_visit_table, old_key, (len(ACTIONS),), np.int32)

    solo_target = reward + DISCOUNT_FACTOR * solo_future_value
    old_solo_q = float(solo_q[my_action_index])
    solo_error = solo_target - old_solo_q
    solo_q[my_action_index] += LEARNING_RATE * solo_error
    solo_visits[my_action_index] += 1

    opponent, relevance = relevant_opponent(old_game_state)
    opponent_name = opponent[0] if opponent is not None else "NONE"
    opponent_action = (
        infer_opponent_action(old_game_state, new_game_state, opponent_name)
        if opponent is not None
        else None
    )

    joint_message = "joint update skipped"

    if opponent_action in ACTIONS:
        opponent_action_index = ACTIONS.index(opponent_action)
        matrix = table_entry(
            self.q_table,
            old_key,
            (len(ACTIONS), len(ACTIONS)),
            np.float32,
        )
        visits = table_entry(
            self.visit_table,
            old_key,
            (len(ACTIONS), len(ACTIONS)),
            np.int32,
        )

        joint_target = reward + DISCOUNT_FACTOR * policy_future_value
        old_joint_q = float(matrix[my_action_index, opponent_action_index])
        joint_error = joint_target - old_joint_q
        matrix[my_action_index, opponent_action_index] += LEARNING_RATE * joint_error
        visits[my_action_index, opponent_action_index] += 1
        joint_message = (
            f"joint ({action}, {opponent_action}) {old_joint_q:.3f} -> "
            f"{matrix[my_action_index, opponent_action_index]:.3f}"
        )

    self.logger.debug(
        "Updated state %s: solo %s %.3f -> %.3f; %s; "
        "opponent %s relevance %.3f; reward %.3f.",
        old_key,
        action,
        old_solo_q,
        float(solo_q[my_action_index]),
        joint_message,
        opponent_name,
        relevance,
        reward,
    )


def save_model(self, path: Path) -> None:
    """Atomically persist values, visit counts, and training progress."""
    payload = {
        "version": MODEL_VERSION,
        "actions": ACTIONS,
        "training_rounds": self.training_rounds,
        "q_table": self.q_table,
        "visit_table": self.visit_table,
        "solo_q_table": self.solo_q_table,
        "solo_visit_table": self.solo_visit_table,
    }

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("wb") as model_file:
        pickle.dump(payload, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    temporary_path.replace(path)


def save_q_table(q_table: dict[StateKey, np.ndarray], path: Path) -> None:
    """Legacy name kept only to fail clearly if external code still calls it."""
    raise RuntimeError("v4 requires save_model(self, path) so visit counts are not lost.")
