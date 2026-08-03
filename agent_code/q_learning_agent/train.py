from typing import List
import events as e


REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.CRATE_DESTROYED: 0.2,
    e.KILLED_OPPONENT: 5.0,
    e.INVALID_ACTION: -1.0,
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0,
}


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

    The actual Q-learning update will be added after the state features have
    been designed.
    """
    reward = reward_from_events(events)
    self.round_reward += reward
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
    self.round_reward += reward_from_events(events)
    self.logger.info(f"Round finished with training reward {self.round_reward}.")
    self.round_reward = 0.0


def reward_from_events(events: List[str]) -> float:
    """Convert game events into a scalar reward."""
    return sum(REWARDS.get(event, 0.0) for event in events)
