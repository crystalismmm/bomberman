from typing import List
import events as e
import numpy as np
from .callbacks import ACTIONS, MODEL_FILE, MOVE_DELTAS, available_actions, calculate_q_values, count_crates_in_blast, distance_to_nearest_bombing_position, state_to_features, danger_time_map, can_escape_after_bomb, wait_safety_features


STEP_TAKEN = "STEP_TAKEN"
MOVED_TOWARDS_BOMBING_POSITION = "MOVED_TOWARDS_BOMBING_POSITION"
MOVED_AWAY_FROM_BOMBING_POSITION = "MOVED_AWAY_FROM_BOMBING_POSITION"
GOOD_BOMB_PLACEMENT = "GOOD_BOMB_PLACEMENT"
BAD_BOMB_PLACEMENT = "BAD_BOMB_PLACEMENT"
UNSAFE_BOMB_PLACEMENT = "UNSAFE_BOMB_PLACEMENT"
UNNECESSARY_WAIT = "UNNECESSARY_WAIT"


REWARDS = {
    e.COIN_COLLECTED: 2.0,
    e.CRATE_DESTROYED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.INVALID_ACTION: -1.0,
    e.KILLED_SELF: -7.0,
    e.GOT_KILLED: -5.0,
    STEP_TAKEN: -0.01,  # small negative reward for taking a step to encourage efficiency
    MOVED_TOWARDS_BOMBING_POSITION: 0.05,  # small positive reward for moving towards a useful bombing position
    MOVED_AWAY_FROM_BOMBING_POSITION: -0.05,  # small negative reward for moving away from a useful bombing position
    GOOD_BOMB_PLACEMENT: 0.3,  # small positive reward for placing a bomb in a useful position
    BAD_BOMB_PLACEMENT: -0.3,  # small negative reward for placing a bomb in a useless position
    UNSAFE_BOMB_PLACEMENT: -1.0,
    UNNECESSARY_WAIT: -0.1,
}

LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.9
EPSILON_START = 0.2  # Exploration rate for epsilon-greedy policy
EPSILON_MIN = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay rate for exploration rate


def setup_training(self):
    """Initialize variables that are only needed during training."""
    self.round_reward = 0.0
    self.epsilon = EPSILON_START


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """Record the reward produced by one transition."""
    events.append(STEP_TAKEN)

    if e.WAITED in events:
        _, safe_move_exists = wait_safety_features(old_game_state)
        if safe_move_exists:
            events.append(UNNECESSARY_WAIT)


    # Reward movement towards useful bombing positions only while safe.
    if self_action in MOVE_DELTAS and e.CRATE_DESTROYED not in events:
        _, _, _, old_position = old_game_state["self"]
        _, _, _, new_position = new_game_state["self"]

        old_danger_map = danger_time_map(old_game_state)
        new_danger_map = danger_time_map(new_game_state)

        old_in_danger = np.isfinite(old_danger_map[old_position])
        new_in_danger = np.isfinite(new_danger_map[new_position])

        if not old_in_danger and not new_in_danger:
            old_distance = distance_to_nearest_bombing_position(
                old_game_state
            )
            new_distance = distance_to_nearest_bombing_position(
                new_game_state
            )

            if old_distance is not None and new_distance is not None:
                if new_distance < old_distance:
                    events.append(
                        MOVED_TOWARDS_BOMBING_POSITION
                    )
                elif new_distance > old_distance:
                    events.append(
                        MOVED_AWAY_FROM_BOMBING_POSITION
                    )

    # Evaluate a bomb only if it was actually placed.
    if e.BOMB_DROPPED in events:
        bomb_position = old_game_state["self"][3]

        if not can_escape_after_bomb(old_game_state):
            events.append(UNSAFE_BOMB_PLACEMENT)
        else:
            crates_in_blast = count_crates_in_blast(
                old_game_state["field"],
                bomb_position,
            )

            if crates_in_blast > 0:
                events.append(GOOD_BOMB_PLACEMENT)
            else:
                events.append(BAD_BOMB_PLACEMENT)

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

    if e.WAITED in events:
        _, safe_move_exists = wait_safety_features(
            last_game_state
        )

        if safe_move_exists:
            events.append(UNNECESSARY_WAIT)

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
    return sum(REWARDS.get(event, 0.0) for event in events)


def update_model(self, old_game_state: dict, action: str, new_game_state: dict | None, reward: float):
    """Update the Q-learning model based on the transition."""
    old_features = state_to_features(old_game_state)
    assert old_features is not None, "Old features should only be None if the old game state is None."

    action_index = ACTIONS.index(action)
    old_q_value = calculate_q_values(self.model, old_features)[action_index]

    if new_game_state is None:
        # If the new game state is None, we are at the end of the game
        max_future_q_value = 0.0
    else:
        new_features = state_to_features(new_game_state)
        assert new_features is not None, "New features should only be None if the new game state is None."

        available = available_actions(new_game_state, allow_bomb=True, allow_wait=True)
        new_q_values = calculate_q_values(self.model, new_features)
        # Only consider Q-values of available actions
        available_q_values = [new_q_values[ACTIONS.index(a)] for a in available]
        max_future_q_value = max(available_q_values) if available_q_values else 0

    # calculate the td_target and td_error
    td_target = reward + DISCOUNT_FACTOR * max_future_q_value
    td_error = td_target - old_q_value

    # update the model
    self.model[action_index] += LEARNING_RATE * td_error * old_features

    # calculate the new Q-value for logging
    new_q_value = calculate_q_values(self.model, old_features)[action_index]

    # Log the update for debugging
    self.logger.debug(
        "Updated Q-value for action '%s': old Q-value = %.3f, new Q-value = %.3f, reward = %.3f, "
        "max future Q-value = %.3f, TD target = %.3f, TD error = %.3f",
        action,
        old_q_value,
        new_q_value,
        reward,
        max_future_q_value,
        td_target,
        td_error,
    )