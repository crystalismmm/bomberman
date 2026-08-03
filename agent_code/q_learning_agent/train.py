from typing import List
import events as e
import numpy as np
from .callbacks import ACTIONS, MODEL_FILE, available_actions, calculate_q_values, state_to_features


REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.CRATE_DESTROYED: 0.2,
    e.KILLED_OPPONENT: 5.0,
    e.INVALID_ACTION: -1.0,
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0,
}

LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.9


def setup_training(self):
    """Initialize variables that are only needed during training."""
    self.round_reward = 0.0


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """Record the reward produced by one transition.
    """
    reward = reward_from_events(events)
    self.round_reward += reward

    # update the model based on the transition
    update_model(self, old_game_state, self_action, new_game_state, reward)

    self.logger.debug(
        f"Action {self_action} produced reward {reward} from events {events}."
    )


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    """Finish the current training round."""
    reward = reward_from_events(events)
    self.round_reward += reward

    # update the model based on the last transition
    update_model(self, last_game_state, last_action, None, reward)

    # Save the trained model
    np.save(MODEL_FILE, self.model)
    self.logger.info("Saved model to %s.", MODEL_FILE)

    self.logger.info(f"Round finished with training reward {self.round_reward}.")
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

        available = available_actions(new_game_state, allow_bomb=False, allow_wait=False)
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