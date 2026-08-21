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
# 1 bias + 4 direction to nearest coin + 4 available actions + 4 coin potential in each direction +
# 1 agent is in danger + 4 directions to nearest safe tile + 1 can escape after bomb + 1 safe bomb quality here +
# 4 directions to the best safe bombing position + 1 whether waiting is safe +
# 1 whether at least one safe move exists + 1 score of the best safe bombing position + 4 directions that lead to a safe position
N_FEATURES = 1 + 4 + 4 + 4 + 1 + 4 + 1 + 1 + 4 + 1 + 1 + 1 + 4
COIN_DECAY = 0.8  # Decay factor for coin potential calculation
BOMB_DECAY = 0.8  # Decay factor for distant bombing positions
BOMB_POWER = 3  # Radius of bomb danger area
BOMB_TIMER = 4  # Number of turns before a bomb explodes

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
    actions = available_actions(game_state, allow_bomb=True, allow_wait=True)

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


def get_blast_tiles(field: np.ndarray, bomb_position: tuple[int, int], power: int = BOMB_POWER) -> set[tuple[int, int]]:
    """Return all tiles affected by a bomb at the given position with the specified power."""
    blast_tiles = {bomb_position}
    x, y = bomb_position

    for dx, dy in MOVE_DELTAS.values():
        for step in range(1, power + 1):
            nx, ny = x + dx * step, y + dy * step
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == -1:  # Wall
                    break
                blast_tiles.add((nx, ny))
            else:
                break

    return blast_tiles


def danger_time_map(game_state: dict) -> np.ndarray:
    """Return the earliest explosion time for every tile."""
    # Initialize the danger time map with infinity
    danger_time = np.full(game_state["field"].shape, np.inf, dtype=np.float32)
    
    # Update danger times for each bomb
    for bomb in game_state["bombs"]:
        bomb_position, bomb_time = bomb
        blast_tiles = get_blast_tiles(game_state["field"], bomb_position)
        for tile in blast_tiles:
            danger_time[tile] = min(danger_time[tile], bomb_time)

    explosion_map = game_state["explosion_map"]
    # Update danger times for tiles that are currently exploding
    danger_time[explosion_map > 0] = 0

    return danger_time

def direction_to_safety(game_state: dict) -> str | None:
    """Return the movement towards the nearest safe tile, or None if already safe."""
    field = game_state["field"]
    danger_map = danger_time_map(game_state)
    _, _, _, (x, y) = game_state["self"]

    bomb_positions = {
        position for position, _ in game_state["bombs"]
    }

    other_positions = {
        other[3] for other in game_state["others"]
    }

    blocked_positions = bomb_positions | other_positions

    queue = deque([((x, y), 0, None)])  # (position, arrival_time, first_move)
    visited = {(x, y)}

    while queue:
        current, arrival_time, first_move = queue.popleft()

        if not np.isfinite(danger_map[current]):
            return first_move

        for action, (dx, dy) in MOVE_DELTAS.items():
            neighbor = (current[0] + dx, current[1] + dy)

            if (0 <= neighbor[0] < field.shape[0] and 0 <= neighbor[1] < field.shape[1]
                and neighbor not in blocked_positions
                and field[neighbor] == 0
                and neighbor not in visited
                and danger_map[neighbor] > arrival_time + 1):

                visited.add(neighbor)
                queue.append((neighbor, arrival_time + 1, first_move or action))

    return None  # No safe tile found


def can_escape_after_bomb(game_state: dict) -> bool:
    """Check if the agent can escape after placing a bomb."""
    _, _, bombs_left, (x, y) = game_state["self"]

    if bombs_left == 0:
        return False

    # Simulate placing a bomb at the agent's current position
    simulated_bombs = list(game_state["bombs"]) + [
        ((x, y), BOMB_TIMER - 1)
    ]
    simulated_game_state = game_state.copy()
    simulated_game_state["bombs"] = simulated_bombs

    # Check if there's a safe direction to move to after placing the bomb
    escape_direction = direction_to_safety(simulated_game_state)
    return escape_direction is not None


def count_crates_in_blast(field: np.ndarray, bomb_position: tuple[int, int], power: int = BOMB_POWER) -> int:
    """Return the number of crates hit by a bomb."""
    blast_tiles = get_blast_tiles(field, bomb_position, power)
    return sum(1 for tile in blast_tiles if field[tile] == 1)


def can_escape_from_position(
    game_state: dict,
    bomb_position: tuple[int, int],
) -> bool:
    """Return whether the agent can escape after bombing at a position."""
    existing_bombs = {
        position for position, _ in game_state["bombs"]
    }

    if bomb_position in existing_bombs:
        return False

    name, score, bombs_left, _ = game_state["self"]

    simulated_game_state = game_state.copy()
    simulated_game_state["self"] = (
        name,
        score,
        bombs_left,
        bomb_position,
    )
    simulated_game_state["bombs"] = (
        list(game_state["bombs"])
        + [(bomb_position, BOMB_TIMER - 1)]
    )

    return direction_to_safety(simulated_game_state) is not None


def bomb_quality_at_position(
    game_state: dict,
    bomb_position: tuple[int, int],
) -> float:
    """Return the normalized quality of a useful and escapable bomb."""
    crates = count_crates_in_blast(
        game_state["field"],
        bomb_position,
    )

    if crates == 0:
        return 0.0

    if not can_escape_from_position(game_state, bomb_position):
        return 0.0

    return crates / (4 * BOMB_POWER)


def best_safe_bombing_position(
    game_state: dict,
    max_distance: int = 8,
) -> tuple[str | None, int | None, float]:
    """Return direction, distance and score of the best safe bombing position."""
    field = game_state["field"]
    _, _, _, start = game_state["self"]

    queue = deque([(start, 0, None)])
    visited = {start}

    bomb_positions = {
        position for position, _ in game_state["bombs"]
    }

    other_positions = {
        other[3] for other in game_state["others"]
    }

    blocked_positions = bomb_positions | other_positions

    best_direction = None
    best_distance = None
    best_score = 0.0

    while queue:
        current, distance, first_move = queue.popleft()

        quality = bomb_quality_at_position(
            game_state,
            current,
        )
        score = quality * BOMB_DECAY ** distance

        if score > best_score:
            best_direction = first_move
            best_distance = distance
            best_score = score

        if distance >= max_distance:
            continue

        for action, (dx, dy) in MOVE_DELTAS.items():
            neighbor = (current[0] + dx, current[1] + dy)

            if (
                0 <= neighbor[0] < field.shape[0]
                and 0 <= neighbor[1] < field.shape[1]
                and field[neighbor] == 0
                and neighbor not in blocked_positions
                and neighbor not in visited
            ):
                visited.add(neighbor)
                queue.append(
                    (
                        neighbor,
                        distance + 1,
                        first_move or action,
                    )
                )

    return best_direction, best_distance, best_score


def direction_to_bombing_position(game_state: dict) -> str | None:
    """Return the direction towards the best safe bombing position."""
    direction, _, _ = best_safe_bombing_position(game_state)
    return direction


def distance_to_nearest_bombing_position(game_state: dict) -> int | None:
    """Return the distance to the best safe bombing position."""
    _, distance, _ = best_safe_bombing_position(game_state)
    return distance


def wait_safety_features(game_state: dict) -> tuple[float, float]:
    """Return whether waiting is a safe and whether a safe movement exists."""
    field = game_state["field"]
    explosion_map = game_state["explosion_map"]
    danger_map = danger_time_map(game_state)

    _, _, _, (x, y) = game_state["self"]

    bomb_positions = {
        position for position, _ in game_state["bombs"]
    }
    other_positions = {
        other[3] for other in game_state["others"]
    }
    blocked_positions = bomb_positions | other_positions

    wait_is_safe = explosion_map[x, y] == 0 and danger_map[x, y] > 1

    safe_move_exists = False

    for dx, dy in MOVE_DELTAS.values():
        target = (x + dx, y + dy)

        if (
            0 <= target[0] < field.shape[0] and
            0 <= target[1] < field.shape[1] and
            field[target] == 0 and
            target not in blocked_positions and
            explosion_map[target] == 0 and
            danger_map[target] > 1
        ):
            safe_move_exists = True

    return float(wait_is_safe), float(safe_move_exists)


def action_preserves_escape_route(game_state: dict, action: str) -> bool:
    """Determines whether the agent will be able to still escape after a certain action."""
    # prevent a key error
    if not (action in MOVE_DELTAS.keys() or action == "WAIT"):
        return False
    
    field = game_state["field"]
    danger_map = danger_time_map(game_state)

    bomb_positions = {
        position for position, _ in game_state["bombs"]
    }
    other_positions = {
        other[3] for other in game_state["others"]
    }
    blocked_positions = bomb_positions | other_positions

    _, _, _, (x, y) = game_state["self"]

    # calculate position after action
    new_position = (x, y)
    if not action == "WAIT":
        dx, dy = MOVE_DELTAS[action]
        new_position = (x + dx, y + dy)

    # check if the new position is a valid action and not killing
    if not (
        0 <= new_position[0] < field.shape[0] and
        0 <= new_position[1] < field.shape[1] and
        field[new_position] == 0 and
        new_position not in other_positions and
        danger_map[new_position] >= 1 and
        (new_position not in bomb_positions or action == "WAIT")
    ):
        return False

    queue = deque([(new_position, 1)]) # (position, arrival time)
    visited = {new_position}

    while queue:
        pos, arrival_time = queue.popleft()

        # A safe way can be found
        if not np.isfinite(danger_map[pos]):
            return True

        # If we are still in danger look for neighbors
        for dx, dy in MOVE_DELTAS.values():
            neighbor = (pos[0] + dx, pos[1] + dy)
            neighbor_arrival_time = arrival_time + 1

            if (
                0 <= neighbor[0] < field.shape[0] and
                0 <= neighbor[1] < field.shape[1] and
                field[neighbor] == 0 and
                neighbor not in blocked_positions and
                danger_map[neighbor] >= neighbor_arrival_time and
                neighbor not in visited
            ):
                queue.append((neighbor, neighbor_arrival_time))
                visited.add(neighbor)

    return False

        


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

    danger_map = danger_time_map(game_state)
    # Feature for whether the agent is in danger
    _, _, _, (x, y) = game_state["self"]

    in_danger = np.isfinite(danger_map[x, y])
    features[13] = float(in_danger)

    escape_direction = direction_to_safety(game_state)

    for idx, move in enumerate(MOVE_DELTAS.keys()):
        if move == escape_direction:
            features[14 + idx] = 1.0

    # Feature for whether the agent can escape after placing a bomb
    can_escape = can_escape_after_bomb(game_state)
    features[18] = float(can_escape)

    # Feature for whether a bomb here is both useful and escapable
    features[19] = bomb_quality_at_position(
        game_state,
        (x, y),
    )

    # Features for the direction and score of the best safe bombing position
    bombing_direction, _, bombing_score = (
        best_safe_bombing_position(game_state)
    )

    for idx, move in enumerate(MOVE_DELTAS.keys()):
        if move == bombing_direction:
            features[20 + idx] = 1.0

    #wait_is_safe, safe_move_exists = wait_safety_features(game_state)
    wait_is_safe = action_preserves_escape_route(game_state, action="WAIT")
    safe_move_exists = 0.0
    for idx, move in enumerate(MOVE_DELTAS.keys()):
        if action_preserves_escape_route(game_state, action=move):
            features[27 + idx] = 1.0
            safe_move_exists = 1.0
    features[24] = float(wait_is_safe)
    features[25] = float(safe_move_exists)
    features[26] = bombing_score

    return features


def calculate_q_values(model: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Calculate Q-values for all actions given the current features."""
    return model @ features  # Matrix multiplication to get Q-values
