import numpy as np
from collections import deque


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")

MOVE_DELTAS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}


def setup(self):
    """Initialize objects that are needed while the agent is running."""
    self.logger.info("Setting up q_learning_agent.")
    self.rng = np.random.default_rng()


def act(self, game_state: dict) -> str:
    """Choose an action.

    For now, this is a random baseline that only chooses immediately safe,
    executable movement actions. The Q-learning policy will replace this
    selection in the next development step.
    """
    actions = available_actions(game_state, allow_bomb=False)
    action = self.rng.choice(actions)
    self.logger.debug(f"Available actions: {actions}; selected: {action}")
    return str(action)


def available_actions(game_state: dict, allow_bomb: bool = True) -> list[str]:
    """Return actions that can be executed in the current state.

    Movement into a current explosion is excluded as well. Future bomb danger
    is intentionally not handled here yet.
    """
    field = game_state["field"]
    explosion_map = game_state["explosion_map"]
    _, _, bombs_left, (x, y) = game_state["self"]

    bomb_positions = {position for position, _ in game_state["bombs"]}
    other_positions = {other[3] for other in game_state["others"]}
    blocked_positions = bomb_positions | other_positions

    actions = []
    for action, (dx, dy) in MOVE_DELTAS.items():
        target = (x + dx, y + dy)
        if field[target] == 0 and target not in blocked_positions:
            if explosion_map[target] == 0:
                actions.append(action)

    actions.append("WAIT")

    if allow_bomb and bombs_left:
        actions.append("BOMB")

    return actions


def direction_to_nearest_coin(game_state: dict) -> str | None:
    """Return the direction to the nearest coin.

    If no coin is reachable, return None.
    """
    field = game_state["field"]
    coins = set(game_state["coins"])
    _, _, _, (x, y) = game_state["self"]

    if not coins:
        return None

    # Compute a Breadth-First Search (BFS) to find the nearest coin
    queue = deque([(x, y)])
    visited = set(queue)
    parent = {queue[0]: None}

    if (x, y) in coins:
        return None  # Already on a coin

    while queue:
        current = queue.popleft()
        if current in coins:
            # Backtrack to find the direction
            while parent[current] != (x, y):
                current = parent[current]
            dx, dy = current[0] - x, current[1] - y
            for action, (adx, ady) in MOVE_DELTAS.items():
                if (dx, dy) == (adx, ady):
                    return action
            return None  # Should not happen

        for _, (dx, dy) in MOVE_DELTAS.items():
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                0 <= neighbor[0] < field.shape[0]
                and 0 <= neighbor[1] < field.shape[1]
                and field[neighbor] == 0
                and neighbor not in visited
            ):
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    return None