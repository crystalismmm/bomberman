import numpy as np
from collections import deque
from pathlib import Path


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")

MOVE_DELTAS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}

N_FEATURES = 13  # 1 bias + 4 direction to nearest coin + 4 available actions + 4 directions to nearest coin cluster
COIN_DECAY = 0.8  # Decay factor for coin potential calculation

MODEL_FILE = Path(__file__).resolve().with_name("model.npy")

EPSILON_EVAL = 0.0


def setup(self):
    """Initialize the agent and load a trained model when available."""
    self.rng = np.random.default_rng()

    expected_shape = (len(ACTIONS), N_FEATURES)

    if self.train:
        self.logger.info("Starting training with a new model.")
        self.model = np.zeros(expected_shape, dtype=np.float32)
        return

    if MODEL_FILE.is_file():
        self.logger.info("Loading model from %s.", MODEL_FILE)

        model = np.load(MODEL_FILE, allow_pickle=False)

        if model.shape != expected_shape:
            raise ValueError(
                f"Saved model has shape {model.shape}, "
                f"but expected {expected_shape}."
            )

        self.model = model.astype(np.float32, copy=False)

    else:
        self.logger.warning(
            "No saved model found at %s. Using an untrained model.",
            MODEL_FILE,
        )
        self.model = np.zeros(expected_shape, dtype=np.float32)


def act(self, game_state: dict) -> str:
    """Choose an action.

    The action is chosen based on the current Q-values and an epsilon-greedy policy.
    """
    actions = available_actions(game_state, allow_bomb=False, allow_wait=False)

    features = state_to_features(game_state)
    assert features is not None, "Features should only be None if the game state is None."
    q_values = calculate_q_values(self.model, features)

    epsilon = self.epsilon if self.train else EPSILON_EVAL  # Use a very low epsilon during evaluation

    if self.rng.random() < epsilon:
        decision_type = "exploration"
        action = self.rng.choice(actions)
    else:
        decision_type = "exploitation"

        # Select the action with the highest Q-value among available actions
        # We don't consider the Q-values of the bomb action here since it's not available in this context
        # for equal Q-values, we select a random action among the best ones
        available_q_values = {action: q_values[idx] for idx, action in enumerate(ACTIONS) if action in actions}
        max_q_value = max(available_q_values.values())
        best_actions = [action for action, q in available_q_values.items() if np.isclose(q, max_q_value)]
        action = self.rng.choice(best_actions)

    self.logger.debug(
        "Round: %s | Step: %s | Mode: %s | Features: %s | "
        "Q-values: %s | Available: %s | Selected: %s",
        game_state["round"],
        game_state["step"],
        decision_type,
        features.tolist(),
        np.round(q_values, 3).tolist(),
        actions,
        action,
    )

    return str(action)


def available_actions(game_state: dict, allow_bomb: bool = True, allow_wait: bool = True) -> list[str]:
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

    if allow_wait or not actions:
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


def bfs_distance_map(field: np.ndarray, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    """Perform BFS to compute the distance from the start position to all reachable positions."""
    distance_map = {}
    queue = deque([start])
    distance_map[start] = 0

    while queue:
        current = queue.popleft()
        current_distance = distance_map[current] + 1

        for dx, dy in MOVE_DELTAS.values():
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                0 <= neighbor[0] < field.shape[0]
                and 0 <= neighbor[1] < field.shape[1]
                and field[neighbor] == 0
                and neighbor not in distance_map
            ):
                distance_map[neighbor] = current_distance
                queue.append(neighbor)

    return distance_map


def coin_potential(field: np.ndarray, start: tuple[int, int], coins: list[tuple[int, int]], decay: float = COIN_DECAY) -> float:
    """Return the discounted potential of all reachable coins."""
    distance_map = bfs_distance_map(field, start)
    potential = 0.0

    for coin in coins:
        if coin in distance_map:
            distance = distance_map[coin]
            potential += decay ** distance

    return potential


def state_to_features(game_state: dict) -> np.ndarray | None:
    """Convert the game state to a feature vector."""
    if game_state is None:
        return None

    # Feature vector is built as follows:
    # - 1 feature is always 1 (bias term)
    # - 4 features for the direction to the nearest coin (one-hot encoded)
    # - 4 features for the available actions (one-hot encoded)
    features = np.zeros(N_FEATURES, dtype=np.float32)

    # Bias term
    features[0] = 1.0

    # Direction to nearest coin
    direction = direction_to_nearest_coin(game_state)
    for idx, move in enumerate(MOVE_DELTAS.keys()):
        if move == direction:
            features[1 + idx] = 1.0

    # Available actions
    available = available_actions(game_state)
    for idx, move in enumerate(MOVE_DELTAS.keys()):
        if move in available:
            features[5 + idx] = 1.0

    # Directions to nearest coin cluster
    for idx, move in enumerate(MOVE_DELTAS.keys()):
        if move in available:
            dx, dy = MOVE_DELTAS[move]
            potential_position = (game_state["self"][3][0] + dx, game_state["self"][3][1] + dy)
            move_potential = coin_potential(game_state["field"], potential_position, game_state["coins"])
            features[9 + idx] = move_potential
    # normalize the coin potential features to be in the range [0, 1]
    max_potential = np.max(features[9:13])
    if max_potential > 0:
        features[9:13] /= max_potential

    return features


def calculate_q_values(model: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Calculate Q-values for all actions given the current features."""
    return model @ features  # Matrix multiplication to get Q-values