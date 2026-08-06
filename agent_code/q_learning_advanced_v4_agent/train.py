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


REWARDS = {
    e.COIN_COLLECTED: 2.0,
    e.CRATE_DESTROYED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.INVALID_ACTION: -1.0,
    e.KILLED_SELF: -7.0,
    e.GOT_KILLED: -5.0,
    STEP_TAKEN: -0.01,
    MOVED_TOWARDS_BOMBING_POSITION: 0.05,
    MOVED_AWAY_FROM_BOMBING_POSITION: -0.05,
    GOOD_BOMB_PLACEMENT: 0.3,
    BAD_BOMB_PLACEMENT: -1.0,
    UNSAFE_BOMB_PLACEMENT: -1.0,
    UNNECESSARY_WAIT: -0.2,
    DANGEROUS_ACTION: -0.5,
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

    if crates_in_blast > 0:
        events.append(GOOD_BOMB_PLACEMENT)
    else:
        events.append(BAD_BOMB_PLACEMENT)


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

    if self_action in MOVE_DELTAS and e.CRATE_DESTROYED not in events:
        old_position = old_game_state["self"][3]
        new_position = new_game_state["self"][3]

        old_danger_map = danger_time_map(old_game_state)
        new_danger_map = danger_time_map(new_game_state)

        old_in_danger = np.isfinite(
            old_danger_map[old_position]
        )
        new_in_danger = np.isfinite(
            new_danger_map[new_position]
        )

        if not old_in_danger and not new_in_danger:
            _, _, old_bombing_score = (
                best_safe_bombing_position(old_game_state)
            )
            _, _, new_bombing_score = (
                best_safe_bombing_position(new_game_state)
            )

            if not np.isclose(
                old_bombing_score,
                new_bombing_score,
            ):
                if new_bombing_score > old_bombing_score:
                    events.append(
                        MOVED_TOWARDS_BOMBING_POSITION
                    )
                else:
                    events.append(
                        MOVED_AWAY_FROM_BOMBING_POSITION
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
