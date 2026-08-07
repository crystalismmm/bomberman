from collections import deque
from typing import List

import events as e
import numpy as np

from .callbacks import (
    ACTIONS,
    MODEL_FILE,
    MOVE_DELTAS,
    action_preserves_escape_route,
    available_actions,
    best_safe_bombing_position,
    calculate_q_values,
    can_escape_after_bomb,
    count_crates_in_blast,
    danger_time_map,
    state_to_features,
)


STEP_TAKEN = "STEP_TAKEN"
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
MOVED_TOWARDS_BOMBING_POSITION = (
    "MOVED_TOWARDS_BOMBING_POSITION"
)
MOVED_AWAY_FROM_BOMBING_POSITION = (
    "MOVED_AWAY_FROM_BOMBING_POSITION"
)
GOOD_BOMB_PLACEMENT = "GOOD_BOMB_PLACEMENT"
BAD_BOMB_PLACEMENT = "BAD_BOMB_PLACEMENT"
UNSAFE_BOMB_PLACEMENT = "UNSAFE_BOMB_PLACEMENT"
UNNECESSARY_WAIT = "UNNECESSARY_WAIT"
DANGEROUS_ACTION = "DANGEROUS_ACTION"
OSCILLATION = "OSCILLATION"


REWARDS = {
    e.COIN_COLLECTED: 2.0,
    e.CRATE_DESTROYED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.INVALID_ACTION: -1.0,
    e.KILLED_SELF: -7.0,
    e.GOT_KILLED: -5.0,
    STEP_TAKEN: -0.01,
    MOVED_TOWARDS_COIN: 0.1,
    MOVED_AWAY_FROM_COIN: -0.1,
    MOVED_TOWARDS_BOMBING_POSITION: 0.05,
    MOVED_AWAY_FROM_BOMBING_POSITION: -0.05,
    # Classification only: actual crate destruction supplies the reward.
    GOOD_BOMB_PLACEMENT: 0.0,
    BAD_BOMB_PLACEMENT: -1.0,
    UNSAFE_BOMB_PLACEMENT: -1.0,
    UNNECESSARY_WAIT: -0.75,
    DANGEROUS_ACTION: -0.5,
    OSCILLATION: -0.5,
}

LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.9
EPSILON_START = 0.2
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995


def setup_training(self):
    """Initialize variables that are only needed during training."""
    self.round_reward = 0.0
    self.epsilon = EPSILON_START
    self.position_history = deque(maxlen=4)
    self.position_history_round = None


def shortest_path_distance(
    game_state: dict,
    start: tuple[int, int],
    target: tuple[int, int],
) -> float:
    """Return the shortest walkable distance to a fixed target."""
    start = (int(start[0]), int(start[1]))
    target = (int(target[0]), int(target[1]))

    if start == target:
        return 0.0

    field = game_state["field"]
    blocked_positions = {
        position for position, _ in game_state["bombs"]
    }
    blocked_positions |= {
        other[3] for other in game_state["others"]
    }
    blocked_positions.discard(start)

    if target in blocked_positions or field[target] != 0:
        return np.inf

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        position, distance = queue.popleft()

        for dx, dy in MOVE_DELTAS.values():
            neighbor = (
                position[0] + dx,
                position[1] + dy,
            )

            if not (
                0 <= neighbor[0] < field.shape[0]
                and 0 <= neighbor[1] < field.shape[1]
                and field[neighbor] == 0
                and neighbor not in blocked_positions
                and neighbor not in visited
            ):
                continue

            if neighbor == target:
                return float(distance + 1)

            visited.add(neighbor)
            queue.append((neighbor, distance + 1))

    return np.inf


def nearest_reachable_coin(
    game_state: dict,
) -> tuple[tuple[int, int] | None, float]:
    """Return the nearest currently reachable coin and its distance."""
    start = game_state["self"][3]
    best_coin = None
    best_distance = np.inf

    for coin in game_state["coins"]:
        distance = shortest_path_distance(
            game_state,
            start,
            coin,
        )

        if distance < best_distance:
            best_coin = coin
            best_distance = distance

    return best_coin, best_distance


def add_action_safety_events(
    game_state: dict,
    action: str,
    events: List[str],
):
    """Add safety events for movement and WAIT actions."""
    if action not in MOVE_DELTAS and action != "WAIT":
        return

    action_is_safe = action_preserves_escape_route(
        game_state,
        action=action,
    )

    if not action_is_safe:
        events.append(DANGEROUS_ACTION)
        return

    if action != "WAIT":
        return

    safe_move_exists = any(
        action_preserves_escape_route(
            game_state,
            action=move,
        )
        for move in MOVE_DELTAS
    )

    if safe_move_exists:
        events.append(UNNECESSARY_WAIT)


def add_oscillation_event(
    self,
    old_game_state: dict,
    action: str,
    new_game_state: dict,
    events: List[str],
):
    """Penalize returning to the previous position while safe."""
    old_position = old_game_state["self"][3]
    new_position = new_game_state["self"][3]
    round_number = old_game_state["round"]

    if getattr(self, "position_history_round", None) != round_number:
        self.position_history.clear()
        self.position_history.append(old_position)
        self.position_history_round = round_number

    elif (
        not self.position_history
        or self.position_history[-1] != old_position
    ):
        self.position_history.clear()
        self.position_history.append(old_position)

    old_danger_map = danger_time_map(old_game_state)
    new_danger_map = danger_time_map(new_game_state)

    old_in_danger = np.isfinite(
        old_danger_map[old_position]
    )
    new_in_danger = np.isfinite(
        new_danger_map[new_position]
    )

    # Forget movements made during a bomb escape.
    if old_in_danger or new_in_danger:
        self.position_history.clear()
        self.position_history.append(new_position)
        return

    # WAIT, BOMB and invalid movements are handled elsewhere.
    if (
        action not in MOVE_DELTAS
        or new_position == old_position
    ):
        return

    if (
        len(self.position_history) >= 2
        and new_position == self.position_history[-2]
    ):
        events.append(OSCILLATION)

    self.position_history.append(new_position)


def add_movement_progress_events(
    old_game_state: dict,
    action: str,
    new_game_state: dict,
    events: List[str],
):
    """Reward progress towards a coin or a fixed bombing target."""
    if action not in MOVE_DELTAS:
        return

    old_position = old_game_state["self"][3]
    new_position = new_game_state["self"][3]

    old_danger_map = danger_time_map(old_game_state)
    new_danger_map = danger_time_map(new_game_state)

    if (
        np.isfinite(old_danger_map[old_position])
        or np.isfinite(new_danger_map[new_position])
    ):
        return

    coin_target, old_coin_distance = (
        nearest_reachable_coin(old_game_state)
    )

    if coin_target is not None:
        if e.COIN_COLLECTED in events:
            return

        new_coin_distance = shortest_path_distance(
            old_game_state,
            new_position,
            coin_target,
        )

        if new_coin_distance < old_coin_distance:
            events.append(MOVED_TOWARDS_COIN)
        elif new_coin_distance > old_coin_distance:
            events.append(MOVED_AWAY_FROM_COIN)

        return

    bombing_direction, bombing_distance, bombing_score = (
        best_safe_bombing_position(old_game_state)
    )

    if bombing_distance is None or bombing_score <= 0:
        return

    if bombing_distance > 0 and action == bombing_direction:
        events.append(MOVED_TOWARDS_BOMBING_POSITION)
    else:
        events.append(MOVED_AWAY_FROM_BOMBING_POSITION)


def add_bomb_placement_events(
    game_state: dict,
    events: List[str],
):
    """Evaluate a bomb that was successfully placed."""
    if e.BOMB_DROPPED not in events:
        return

    bomb_position = game_state["self"][3]

    if not can_escape_after_bomb(game_state):
        events.append(UNSAFE_BOMB_PLACEMENT)
        return

    crates_in_blast = count_crates_in_blast(
        game_state["field"],
        bomb_position,
    )

    if crates_in_blast == 0:
        events.append(BAD_BOMB_PLACEMENT)
        return

    events.append(GOOD_BOMB_PLACEMENT)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """Record the reward produced by one transition."""
    events.append(STEP_TAKEN)

    add_action_safety_events(
        old_game_state,
        self_action,
        events,
    )
    add_oscillation_event(
        self,
        old_game_state,
        self_action,
        new_game_state,
        events
    )
    add_movement_progress_events(
        old_game_state,
        self_action,
        new_game_state,
        events,
    )
    add_bomb_placement_events(
        old_game_state,
        events,
    )

    reward = reward_from_events(events)
    self.round_reward += reward

    update_model(
        self,
        old_game_state,
        self_action,
        new_game_state,
        reward,
    )

    self.logger.debug(
        "Action %s produced reward %.3f from events %s.",
        self_action,
        reward,
        events,
    )


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    """Finish the current training round."""
    events.append(STEP_TAKEN)

    add_action_safety_events(
        last_game_state,
        last_action,
        events,
    )
    add_bomb_placement_events(
        last_game_state,
        events,
    )

    reward = reward_from_events(events)
    self.round_reward += reward

    update_model(
        self,
        last_game_state,
        last_action,
        None,
        reward,
    )

    np.save(MODEL_FILE, self.model)
    self.logger.info("Saved model to %s.", MODEL_FILE)

    self.epsilon = max(
        EPSILON_MIN,
        self.epsilon * EPSILON_DECAY,
    )

    self.logger.info(
        "Round finished with reward %.3f and epsilon %.4f.",
        self.round_reward,
        self.epsilon,
    )

    self.position_history.clear()
    self.position_history_round = None
    self.round_reward = 0.0


def reward_from_events(events: List[str]) -> float:
    """Convert game events into a scalar reward."""
    return sum(
        REWARDS.get(event, 0.0)
        for event in events
    )


def update_model(
    self,
    old_game_state: dict,
    action: str,
    new_game_state: dict | None,
    reward: float,
):
    """Update the Q-learning model based on the transition."""
    old_features = state_to_features(old_game_state)
    assert old_features is not None, (
        "Old features should only be None if the old game state is None."
    )

    action_index = ACTIONS.index(action)
    old_q_value = calculate_q_values(
        self.model,
        old_features,
    )[action_index]

    if new_game_state is None:
        max_future_q_value = 0.0
    else:
        new_features = state_to_features(new_game_state)
        assert new_features is not None, (
            "New features should only be None if the new game state is None."
        )

        available = available_actions(
            new_game_state,
            allow_bomb=True,
            allow_wait=True,
        )
        new_q_values = calculate_q_values(
            self.model,
            new_features,
        )
        available_q_values = [
            new_q_values[ACTIONS.index(available_action)]
            for available_action in available
        ]
        max_future_q_value = (
            max(available_q_values)
            if available_q_values
            else 0.0
        )

    td_target = (
        reward
        + DISCOUNT_FACTOR * max_future_q_value
    )
    td_error = td_target - old_q_value

    self.model[action_index] += (
        LEARNING_RATE
        * td_error
        * old_features
    )

    new_q_value = calculate_q_values(
        self.model,
        old_features,
    )[action_index]

    self.logger.debug(
        "Updated Q-value for action '%s': old Q-value = %.3f, "
        "new Q-value = %.3f, reward = %.3f, "
        "max future Q-value = %.3f, TD target = %.3f, "
        "TD error = %.3f",
        action,
        old_q_value,
        new_q_value,
        reward,
        max_future_q_value,
        td_target,
        td_error,
    )
