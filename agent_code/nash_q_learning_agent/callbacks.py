"""Callbacks for a compact tabular Nash Q-learning Bomberman agent."""

from __future__ import annotations

import itertools
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")

MOVE_DELTAS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}

OPPOSITE_ACTION = {
    "UP": "DOWN",
    "RIGHT": "LEFT",
    "DOWN": "UP",
    "LEFT": "RIGHT",
}

BOMB_POWER = 3
BOMB_TIMER = 4
BOMBING_DISTANCE_DECAY = 0.85

EPSILON_EVAL = 0.0
RNG_SEED = 0

MODEL_VERSION = 3
MODEL_FILE = Path(__file__).resolve().with_name("nash_q_table.pkl")

TARGET_ESCAPE = "ESCAPE"
TARGET_COIN = "COIN"
TARGET_BOMBING = "BOMBING"
TARGET_NONE = "NONE"

DIRECTION_HERE = "HERE"
DIRECTION_NONE = "NONE"

BOMB_UNAVAILABLE = "UNAVAILABLE"
BOMB_ATTACK = "ATTACK"
BOMB_GOOD_LOW = "GOOD_LOW"
BOMB_GOOD_HIGH = "GOOD_HIGH"
BOMB_USELESS = "USELESS"
BOMB_UNSAFE = "UNSAFE"

WAIT_NECESSARY = "NECESSARY"
WAIT_UNNECESSARY = "UNNECESSARY"
WAIT_UNSAFE = "UNSAFE"

PREVIOUS_NONE = "NONE"

StateKey: TypeAlias = tuple[str, str, int, str, str, str]


@dataclass(frozen=True)
class StateAnalysis:
    """Compact table key plus metadata used for reward shaping."""

    key: StateKey
    target_position: tuple[int, int] | None
    target_distance: int | None


# =====================================================================================================
# ======================================== Callback API ===============================================
# =====================================================================================================


def setup(self):
    """Initialize history and either create or load the Nash Q-table."""
    self.rng = np.random.default_rng(RNG_SEED)
    self.q_table: dict[StateKey, np.ndarray] = {}

    self.current_round = None
    self.previous_action = None
    self.state_previous_action = None
    self.last_state_key = None

    self.last_opponent_name = None
    self.last_opponent_action = None

    if self.train:
        self.logger.info("Starting Nash Q-learning with an empty Q-table.")
        return

    if not MODEL_FILE.is_file():
        self.logger.warning("No Nash Q-table found at %s. Evaluation starts untrained.", MODEL_FILE)
        return

    self.q_table = load_q_table(MODEL_FILE)

    self.logger.info("Loaded %d Nash Q-learning states from %s.", len(self.q_table), MODEL_FILE)


def act(self, game_state: dict) -> str:
    """Choose an action using the mixed Nash equilibrium."""
    round_number = game_state["round"]

    if self.current_round != round_number:
        self.current_round = round_number
        self.previous_action = None
        self.state_previous_action = None
        self.last_state_key = None
        self.last_opponent_name = None
        self.last_opponent_action = None

    analysis = analyze_state(game_state, self.previous_action)

    q_matrix = q_matrix_for_state(self.q_table, analysis.key)

    opponent = most_relevant_opponent(game_state)

    epsilon = self.epsilon if self.train else EPSILON_EVAL

    if opponent is None:
        opponent_name = "NONE"
        opponent_relevance = 0.0

        own_strategy, equilibrium_value = solve_single_agent_state(q_matrix)
        opponent_strategy = np.ones(len(ACTIONS), dtype=np.float32) / len(ACTIONS)

        if self.rng.random() < epsilon:
            decision_type = "exploration"
            action = str(self.rng.choice(ACTIONS))
        else:
            decision_type = "no-opponent"
            action = str(self.rng.choice(ACTIONS, p=own_strategy))

    else:
        opponent_name = opponent[0]
        opponent_position = opponent[3]

        opponent_relevance = opponent_impact_score(
            game_state,
            opponent_position
        )

        (
            own_strategy,
            opponent_strategy,
            equilibrium_value
        ) = solve_zero_sum_game(q_matrix)

        if self.rng.random() < epsilon:
            decision_type = "exploration"
            action = str(self.rng.choice(ACTIONS))
        else:
            decision_type = "nash"
            action = str(self.rng.choice(ACTIONS, p=own_strategy))

    self.last_state_key = analysis.key
    self.state_previous_action = self.previous_action
    self.previous_action = action
    self.last_opponent_name = opponent_name

    self.logger.debug(
        "Round: %s | Step: %s | Mode: %s | State: %s | "
        "Opponent: %s | Relevance: %.3f | Q-matrix: %s | "
        "Own Nash strategy: %s | Opponent Nash strategy: %s | "
        "Equilibrium value: %.3f | Selected: %s | Known states: %d",
        game_state["round"],
        game_state["step"],
        decision_type,
        analysis.key,
        opponent_name,
        opponent_relevance,
        np.round(q_matrix, 3).tolist(),
        np.round(own_strategy, 3).tolist(),
        np.round(opponent_strategy, 3).tolist(),
        equilibrium_value,
        action,
        len(self.q_table)
    )

    return action


# =====================================================================================================
# ======================================= Nash-Q helpers ==============================================
# =====================================================================================================


def most_relevant_opponent(game_state: dict):
    """Return the opponent with the highest current relevance."""
    if not game_state["others"]:
        return None

    return max(game_state["others"], key=lambda other: opponent_impact_score(game_state, other[3]))


def q_matrix_for_state(q_table: dict[StateKey, np.ndarray], state_key: StateKey)-> np.ndarray:
    """Return the Nash Q-matrix for a state."""
    values = q_table.get(state_key)

    if values is None:
        return np.zeros((len(ACTIONS), len(ACTIONS)), dtype=np.float32)

    return values


def solve_zero_sum_game(q_matrix: np.ndarray)-> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve a two-player zero-sum matrix game.

    Rows correspond to our actions and columns correspond to
    the opponent's actions.
    """
    matrix = np.asarray(q_matrix, dtype=np.float64)

    if not np.any(matrix):
        uniform_strategy = np.ones(len(ACTIONS), dtype=np.float32) / len(ACTIONS)

        return uniform_strategy, uniform_strategy.copy(), 0.0

    row_strategy, column_strategy, value = solve_zero_sum_game_by_support_enumeration(matrix)

    return row_strategy.astype(np.float32), column_strategy.astype(np.float32), float(value)


def solve_single_agent_state(q_matrix: np.ndarray)-> tuple[np.ndarray, float]:
    """Return a tie-aware greedy strategy when no opponent remains."""
    matrix = np.asarray(q_matrix, dtype=np.float64)

    row_values = np.mean(matrix, axis=1)
    best_value = float(np.max(row_values))

    best_rows = np.flatnonzero(np.isclose(row_values, best_value))

    strategy = np.zeros(len(ACTIONS), dtype=np.float32)
    strategy[best_rows] = 1.0 / len(best_rows)

    return strategy, best_value


def solve_zero_sum_game_by_support_enumeration(q_matrix: np.ndarray)-> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve a small zero-sum matrix game by support enumeration.

    For an equilibrium (p, q, v):

        A q <= v  for every row
        p A >= v  for every column

    because the row player maximizes and the column player minimizes.
    """
    n_rows, n_columns = q_matrix.shape

    best_solution = None

    max_support = min(n_rows, n_columns)

    for support_size in range(1, max_support + 1):
        for row_support in itertools.combinations(range(n_rows), support_size):
            for column_support in itertools.combinations(range(n_columns), support_size):
                row_support = list(row_support)
                column_support = list(column_support)

                submatrix = q_matrix[np.ix_(row_support, column_support)]

                row_strategy = solve_row_support(submatrix)
                column_strategy = solve_column_support(submatrix)

                if row_strategy is None or column_strategy is None:
                    continue

                full_row_strategy = np.zeros(n_rows)
                full_column_strategy = np.zeros(n_columns)

                full_row_strategy[row_support] = row_strategy
                full_column_strategy[column_support] = column_strategy

                value = float(full_row_strategy @ q_matrix @ full_column_strategy)

                row_payoffs = (q_matrix @ full_column_strategy)

                column_payoffs = (full_row_strategy @ q_matrix)

                # Row player maximizes:
                # no row may obtain MORE than the equilibrium value.
                if np.any(row_payoffs > value + 1e-7):
                    continue

                # Column player minimizes:
                # no column may obtain LESS than the equilibrium value.
                if np.any(column_payoffs < value - 1e-7):
                    continue

                best_solution = (full_row_strategy, full_column_strategy, value)

                return best_solution

    # This should only be reached because of numerical degeneracy.
    #
    # Use the maximin row and minimax column strategies independently.
    # This gives a safe deterministic approximation rather than pretending
    # that the resulting pair is an exact Nash equilibrium.
    row_index = int(np.argmax(np.min(q_matrix, axis=1)))

    column_index = int(np.argmin(np.max(q_matrix, axis=0)))

    row_strategy = np.zeros(n_rows)
    column_strategy = np.zeros(n_columns)

    row_strategy[row_index] = 1.0
    column_strategy[column_index] = 1.0

    value = float(row_strategy @ q_matrix @ column_strategy)

    return row_strategy, column_strategy, value


def solve_row_support(matrix: np.ndarray)-> np.ndarray | None:
    """
    Solve for the row player's probabilities on a given support.

    We require:

        p^T A_j = v

    for every column j in the opponent's support.
    """
    rows, columns = matrix.shape

    system = np.zeros((columns + 1, rows + 1), dtype=np.float64)

    system[:columns, :rows] = matrix.T
    system[:columns, rows] = -1.0
    system[columns, :rows] = 1.0

    rhs = np.zeros(columns + 1, dtype=np.float64)

    rhs[columns] = 1.0

    try:
        solution, residuals, rank, _ = np.linalg.lstsq(system, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if np.max(np.abs(system @ solution - rhs)) > 1e-7:
        return None

    strategy = solution[:rows]

    if np.any(strategy < -1e-7):
        return None

    strategy = np.maximum(strategy, 0.0)

    total = strategy.sum()

    if total <= 0:
        return None

    return strategy / total


def solve_column_support(matrix: np.ndarray)-> np.ndarray | None:
    """
    Solve for the column player's probabilities on a given support.

    We require:

        A q_i = v

    for every row i in the row player's support.
    """
    rows, columns = matrix.shape

    system = np.zeros((rows + 1, columns + 1), dtype=np.float64)

    system[:rows, :columns] = matrix
    system[:rows, columns] = -1.0
    system[rows, :columns] = 1.0

    rhs = np.zeros(rows + 1, dtype=np.float64)

    rhs[rows] = 1.0

    try:
        solution, residuals, rank, _ = np.linalg.lstsq(system, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if np.max(np.abs(system @ solution - rhs)) > 1e-7:
        return None

    strategy = solution[:columns]

    if np.any(strategy < -1e-7):
        return None

    strategy = np.maximum(strategy, 0.0)

    total = strategy.sum()

    if total <= 0:
        return None

    return strategy / total


# =====================================================================================================
# =================================== Opponent action inference =======================================
# =====================================================================================================


def infer_opponent_action(old_game_state: dict, new_game_state: dict | None, opponent_name: str)-> str | None:
    """
    Infer the opponent's action from two consecutive game states.

    Returns None when the action cannot be determined reliably.

    This is important for Nash-Q because the table represents:

        Q(s, our_action, opponent_action)

    We must not silently turn an unknown action into WAIT.
    """
    if new_game_state is None:
        return None

    old_opponent = next((other for other in old_game_state["others"] if other[0] == opponent_name), None)

    if old_opponent is None:
        return None

    new_opponent = next((other for other in new_game_state["others"] if other[0] == opponent_name), None)

    old_position = old_opponent[3]

    # If the opponent disappeared, we cannot know which action caused it.
    if new_opponent is None:
        return None

    new_position = new_opponent[3]

    dx = new_position[0] - old_position[0]
    dy = new_position[1] - old_position[1]

    # Movement.
    for action, delta in MOVE_DELTAS.items():
        if delta == (dx, dy):
            return action

    # Bomb placement.
    old_bombs = {position for position, _ in old_game_state["bombs"]}

    new_bombs = {position for position, _ in new_game_state["bombs"]}

    if (old_position in new_bombs and old_position not in old_bombs):
        return "BOMB"

    # Same position, no bomb: WAIT.
    if new_position == old_position:
        return "WAIT"

    # Anything else cannot be inferred safely.
    return None


# =====================================================================================================
# ======================================= State conversion ============================================
# =====================================================================================================


def state_to_key(game_state: dict, previous_action: str | None = None)-> StateKey | None:
    """Public compact state conversion used by tests and training."""
    if game_state is None:
        return None

    return analyze_state(game_state, previous_action).key


def load_q_table(path: Path)-> dict[StateKey, np.ndarray]:
    """Load and validate the Nash Q-table."""
    with path.open("rb") as model_file:
        payload = pickle.load(model_file)

    if not isinstance(payload, dict):
        raise ValueError("The saved Q-table payload is not a dictionary.")

    if payload.get("version") != MODEL_VERSION:
        raise ValueError(f"Unsupported Q-table version {payload.get('version')!r}; expected {MODEL_VERSION}.")

    if tuple(payload.get("actions", ())) != ACTIONS:
        raise ValueError("The saved Q-table uses a different action order.")

    raw_table = payload.get("q_table")

    if not isinstance(raw_table, dict):
        raise ValueError("The saved payload contains no valid Q-table.")

    table = {}

    expected_shape = (len(ACTIONS), len(ACTIONS))

    for key, raw_values in raw_table.items():
        if not isinstance(key, tuple) or len(key) != 6:
            raise ValueError(f"Invalid tabular state key: {key!r}")

        values = np.asarray(raw_values, dtype=np.float32)

        if values.shape != expected_shape:
            raise ValueError(f"Invalid Q-matrix shape {values.shape} for state {key!r}.")

        if not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid Q-values for state {key!r}.")

        table[key] = values

    return table


def available_actions(game_state: dict) -> list[str]:
    """Return the complete action space."""
    del game_state
    return list(ACTIONS)


# =====================================================================================================
# ======================================= Danger / movement ===========================================
# =====================================================================================================


def inside_field(field: np.ndarray, position: tuple[int, int])-> bool:
    """Return whether a coordinate lies inside the arena."""
    return (0 <= position[0] < field.shape[0] and 0 <= position[1] < field.shape[1])


def blocked_positions(game_state: dict) -> set[tuple[int, int]]:
    """Return positions occupied by bombs or other agents."""
    bombs = {position for position, _ in game_state["bombs"]}

    others = {other[3] for other in game_state["others"]}

    return bombs | others


def get_blast_tiles(field: np.ndarray, bomb_position: tuple[int, int], power: int = BOMB_POWER)-> set[tuple[int, int]]:
    """Return blast tiles using the game engine's explosion rules."""
    blast_tiles = {bomb_position}

    x, y = bomb_position

    for dx, dy in MOVE_DELTAS.values():
        for step in range(1, power + 1):
            tile = (x + dx * step, y + dy * step)

            if (not inside_field(field, tile) or field[tile] == -1):
                break

            blast_tiles.add(tile)

            # Crates stop the blast.
            if field[tile] == 1:
                break

    return blast_tiles


def danger_time_map(game_state: dict) -> np.ndarray:
    """Return the earliest bomb timer for every threatened tile."""
    field = game_state["field"]

    danger_time = np.full(field.shape, np.inf, dtype=np.float32)

    for bomb_position, bomb_timer in game_state["bombs"]:
        for tile in get_blast_tiles(field, bomb_position):
            danger_time[tile] = min(danger_time[tile], bomb_timer)

    danger_time[game_state["explosion_map"] > 0] = 0

    return danger_time


def _escape_distance_after_action(game_state: dict, action: str, danger_map: np.ndarray | None = None)-> int | None:
    """Return moves until permanent safety after an action."""
    if action not in MOVE_DELTAS and action != "WAIT":
        return None

    field = game_state["field"]

    if danger_map is None:
        danger_map = danger_time_map(game_state)

    blocked = blocked_positions(game_state)
    start = game_state["self"][3]

    if action == "WAIT":
        new_position = start
    else:
        dx, dy = MOVE_DELTAS[action]

        new_position = (start[0] + dx, start[1] + dy)

    if (
        not inside_field(field, new_position)
        or field[new_position] != 0
        or new_position in blocked
        or danger_map[new_position] < 1
    ):
        return None

    queue = deque([(new_position, 1)])

    visited = {new_position}

    while queue:
        position, arrival_time = queue.popleft()

        if not np.isfinite(danger_map[position]):
            return arrival_time

        for dx, dy in MOVE_DELTAS.values():
            neighbor = (position[0] + dx, position[1] + dy)

            neighbor_arrival = arrival_time + 1

            if (
                inside_field(field, neighbor)
                and field[neighbor] == 0
                and neighbor not in blocked
                and neighbor not in visited
                and danger_map[neighbor] >= neighbor_arrival
            ):
                visited.add(neighbor)

                queue.append((neighbor, neighbor_arrival))

    return None


def safe_move_mask(game_state: dict, danger_map: np.ndarray | None = None)-> int:
    """Encode route-preserving movements in four bits."""
    if danger_map is None:
        danger_map = danger_time_map(game_state)

    mask = 0

    for index, action in enumerate(MOVE_DELTAS):
        if _escape_distance_after_action(game_state, action, danger_map) is not None:
            mask |= 1 << index

    return mask


def direction_to_safety(game_state: dict, danger_map: np.ndarray | None = None)-> str | None:
    """Return the first action of the shortest escape route."""
    if danger_map is None:
        danger_map = danger_time_map(game_state)

    start = game_state["self"][3]

    if not np.isfinite(danger_map[start]):
        return None

    candidates = []

    for order, action in enumerate((*MOVE_DELTAS.keys(), "WAIT")):
        distance = _escape_distance_after_action(game_state, action, danger_map)

        if distance is not None:
            candidates.append((distance, order, action))

    if not candidates:
        return None

    return min(candidates)[2]


# =====================================================================================================
# ========================================== Bomb logic ===============================================
# =====================================================================================================


def count_crates_in_blast(field: np.ndarray, bomb_position: tuple[int, int])-> int:
    """Return the number of crates destroyed by a bomb."""
    return sum(field[tile] == 1 for tile in get_blast_tiles(field, bomb_position))


def bomb_hits_opponent(game_state: dict, bomb_position: tuple[int, int] | None = None)-> bool:
    """Return whether a bomb hits an opponent."""
    if bomb_position is None:
        bomb_position = game_state["self"][3]

    blast_tiles = get_blast_tiles(game_state["field"], bomb_position)

    return any(other[3] in blast_tiles for other in game_state["others"])


def can_escape_from_position(game_state: dict, bomb_position: tuple[int, int])-> bool:
    """Return whether a bomb placed at a position leaves an escape route."""
    existing_bombs = {position for position, _ in game_state["bombs"]}

    if bomb_position in existing_bombs:
        return False

    name, score, bombs_left, _ = game_state["self"]

    simulated_state = dict(game_state)

    simulated_state["self"] = name, score, bombs_left, bomb_position

    simulated_state["bombs"] = list(game_state["bombs"]) + [(bomb_position, BOMB_TIMER - 1)]

    return any(
        _escape_distance_after_action(simulated_state, action) is not None for action in MOVE_DELTAS)


def can_escape_after_bomb(game_state: dict)-> bool:
    """Return whether BOMB is escapable."""
    _, _, bombs_left, position = game_state["self"]

    return bool(bombs_left) and can_escape_from_position(game_state, position)


def bomb_status(game_state: dict) -> str:
    """Classify BOMB."""
    _, _, bombs_left, position = game_state["self"]

    if not bombs_left:
        return BOMB_UNAVAILABLE

    if not can_escape_after_bomb(game_state):
        return BOMB_UNSAFE

    if bomb_hits_opponent(game_state, position):
        return BOMB_ATTACK

    crates_in_blast = count_crates_in_blast(game_state["field"], position)

    if crates_in_blast == 0:
        return BOMB_USELESS

    if crates_in_blast == 1:
        return BOMB_GOOD_LOW

    return BOMB_GOOD_HIGH


def wait_status(game_state: dict, movement_mask: int | None = None, danger_map: np.ndarray | None = None)-> str:
    """Classify WAIT."""
    if danger_map is None:
        danger_map = danger_time_map(game_state)

    if movement_mask is None:
        movement_mask = safe_move_mask(game_state, danger_map)

    wait_is_safe = _escape_distance_after_action(game_state, "WAIT", danger_map) is not None

    if not wait_is_safe:
        return WAIT_UNSAFE

    if movement_mask == 0:
        return WAIT_NECESSARY

    return WAIT_UNNECESSARY


# =====================================================================================================
# ======================================== Navigation =================================================
# =====================================================================================================


def _first_direction(start: tuple[int, int], target: tuple[int, int], parents: dict[tuple[int, int], tuple[int, int] | None ])-> str:
    """Backtrack a BFS tree to obtain the first movement."""
    if target == start:
        return DIRECTION_HERE

    current = target

    while parents[current] != start:
        parent = parents[current]

        if parent is None:
            return DIRECTION_NONE

        current = parent

    delta = (current[0] - start[0], current[1] - start[1])

    for action, action_delta in MOVE_DELTAS.items():
        if action_delta == delta:
            return action

    return DIRECTION_NONE


def reachable_positions(game_state: dict)-> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int] | None]]:
    """Return static BFS distances and parents."""
    field = game_state["field"]
    start = game_state["self"][3]

    blocked = blocked_positions(game_state)
    blocked.discard(start)

    distances = {start: 0}
    parents = {start: None}

    queue = deque([start])

    while queue:
        position = queue.popleft()

        for dx, dy in MOVE_DELTAS.values():
            neighbor = (position[0] + dx, position[1] + dy)

            if (inside_field(field, neighbor) and field[neighbor] == 0 and neighbor not in blocked and neighbor not in distances):
                distances[neighbor] = distances[position] + 1

                parents[neighbor] = position
                queue.append(neighbor)

    return distances, parents


def nearest_reachable_coin(game_state: dict, distances: dict[tuple[int, int], int] | None = None, parents: dict[tuple[int, int], tuple[int, int] | None] | None = None)-> tuple[str | None, tuple[int, int] | None, int | None]:
    """Return direction, position, and distance of the nearest coin."""
    if distances is None or parents is None:
        distances, parents = reachable_positions(game_state)

    reachable_coins = [coin for coin in game_state["coins"] if coin in distances]

    if not reachable_coins:
        return None, None, None

    target = min(reachable_coins, key=lambda coin: (distances[coin], coin))

    direction = _first_direction(game_state["self"][3], target, parents)

    return direction, target, distances[target]


def nearest_safe_bombing_position(game_state: dict, distances: dict[tuple[int, int], int] | None = None, parents: dict[tuple[int, int], tuple[int, int] | None] | None = None)-> tuple[str | None, tuple[int, int] | None, int | None]:
    """Return the best useful and escapable bombing position."""
    if distances is None or parents is None:
        distances, parents = reachable_positions(game_state)

    field = game_state["field"]

    best_target = None
    best_distance = None
    best_rank = None

    for position, distance in distances.items():
        crate_count = count_crates_in_blast(field, position)

        if (crate_count == 0 or not can_escape_from_position(game_state, position)):
            continue

        score = crate_count * BOMBING_DISTANCE_DECAY ** distance

        rank = score, crate_count, -distance, -position[0], -position[1]

        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_target = position
            best_distance = distance

    if (best_target is None or best_distance is None):
        return None, None, None

    direction = _first_direction(game_state["self"][3], best_target, parents)

    return direction, best_target, best_distance


def analyze_state(game_state: dict, previous_action: str | None = None)-> StateAnalysis:
    """Convert a game state into the compact categorical table key."""
    danger_map = danger_time_map(game_state)

    movement_mask = safe_move_mask(game_state, danger_map)

    current_position = game_state["self"][3]

    target_mode = TARGET_NONE
    target_direction = DIRECTION_NONE
    target_position = None
    target_distance = None

    if np.isfinite(danger_map[current_position]):
        target_mode = TARGET_ESCAPE

        escape_action = direction_to_safety(game_state, danger_map)

        if escape_action == "WAIT":
            target_direction = DIRECTION_HERE
            target_position = current_position
            target_distance = 0

        elif escape_action is not None:
            target_direction = escape_action

            dx, dy = MOVE_DELTAS[escape_action]

            target_position = (current_position[0] + dx, current_position[1] + dy)

            target_distance = 1

    else:
        distances, parents = reachable_positions(game_state)

        direction, position, distance = nearest_reachable_coin(game_state, distances, parents)

        if position is not None:
            target_mode = TARGET_COIN
            target_direction = (direction or DIRECTION_NONE)
            target_position = position
            target_distance = distance

        else:
            direction, position, distance = nearest_safe_bombing_position(game_state, distances, parents)

            if position is not None:
                target_mode = TARGET_BOMBING
                target_direction = (direction or DIRECTION_NONE)
                target_position = position
                target_distance = distance

    key: StateKey = (
        target_mode,
        target_direction,
        movement_mask,
        bomb_status(game_state),
        wait_status(game_state, movement_mask, danger_map),
        (
            previous_action
            if previous_action in ACTIONS
            else PREVIOUS_NONE
        )
    )

    return StateAnalysis(key, target_position, target_distance)


# =====================================================================================================
# ================================== Opponent relevance ===============================================
# =====================================================================================================


W_THREAT = 1.0
W_COMPETITION = 1.0
RESOURCE_DECAY = 0.8


def bfs_distance_map(field: np.ndarray, start: tuple[int, int])-> dict[tuple[int, int], int]:
    """Return shortest-path distances from a position."""
    distance_map = {start: 0}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        current_distance = distance_map[current] + 1

        for dx, dy in MOVE_DELTAS.values():
            neighbor = (current[0] + dx, current[1] + dy)

            if (
                inside_field(field, neighbor)
                and field[neighbor] == 0
                and neighbor not in distance_map
            ):
                distance_map[neighbor] = current_distance

                queue.append(neighbor)

    return distance_map


def opponent_distance_relevance(field: np.ndarray, my_position: tuple[int, int], opponent_position: tuple[int, int], decay: float = 0.8)-> float:
    """Return relevance based on opponent distance."""
    distance_map = bfs_distance_map(field, my_position)

    distance = distance_map.get(opponent_position, np.inf)

    if not np.isfinite(distance):
        return 0.0

    return decay ** distance


def threat_from_bomb_position(game_state: dict, bomb_position: tuple[int, int], target_position: tuple[int, int])-> float:
    """Return whether a bomb can hit the target."""
    blast_tiles = get_blast_tiles(game_state["field"], bomb_position)

    return float(target_position in blast_tiles)


def opponent_threat_score(game_state: dict, opponent_position: tuple[int, int], max_distance: int = 8, decay: float = 0.8)-> float:
    """Estimate the strongest credible threat posed by an opponent."""
    field = game_state["field"]

    bomb_positions = {pos for pos, _ in game_state["bombs"]}

    other_positions = {other[3] for other in game_state["others"] if other[3] != opponent_position}

    blocked = bomb_positions | other_positions

    queue = deque([(opponent_position, 0)])

    visited = {opponent_position}
    best_threat = 0.0

    while queue:
        current, distance = queue.popleft()

        if field[current] == 0:
            can_escape = can_escape_from_position(
                game_state,
                current
            )

            if can_escape:
                local_threat = threat_from_bomb_position(game_state, current, game_state["self"][3])

                discounted_threat = local_threat * decay ** distance

                best_threat = max(best_threat, discounted_threat)

        if distance >= max_distance:
            continue

        for dx, dy in MOVE_DELTAS.values():
            neighbor = (
                current[0] + dx,
                current[1] + dy
            )

            if (
                inside_field(field, neighbor)
                and field[neighbor] == 0
                and neighbor not in blocked
                and neighbor not in visited
            ):
                visited.add(neighbor)

                queue.append((neighbor, distance + 1))

    return best_threat


def resource_competition(game_state: dict, opponent_position: tuple[int, int], max_distance: int = 8, decay: float = RESOURCE_DECAY)-> float:
    """Estimate competition over coins and bombing opportunities."""
    field = game_state["field"]

    my_position = game_state["self"][3]

    my_distances = bfs_distance_map(
        field,
        my_position
    )

    opponent_distances = bfs_distance_map(
        field,
        opponent_position
    )

    competition = 0.0

    for coin in game_state["coins"]:
        my_distance = my_distances.get(coin)
        opponent_distance = opponent_distances.get(coin)

        if (
            my_distance is None
            or opponent_distance is None
        ):
            continue

        if (
            my_distance > max_distance
            and opponent_distance > max_distance
        ):
            continue

        resource_value = decay ** min(my_distance, opponent_distance)

        contest_strength = np.exp(-0.5 * abs(my_distance - opponent_distance))

        competition += resource_value * contest_strength

    for position, my_distance in my_distances.items():
        if my_distance > max_distance:
            continue

        if field[position] != 0:
            continue

        crates = count_crates_in_blast(
            field,
            position,
        )

        if crates <= 0:
            continue

        opponent_distance = opponent_distances.get(
            position
        )

        if opponent_distance is None:
            continue

        if opponent_distance > max_distance:
            continue

        resource_value = crates * decay ** min(my_distance, opponent_distance)

        contest_strength = np.exp(-0.5 * abs(my_distance - opponent_distance))

        competition += resource_value * contest_strength

    return float(competition)


def opponent_impact_score(game_state: dict, opponent_position: tuple[int, int])-> float:
    """Estimate how relevant an opponent is to the current game state."""
    field = game_state["field"]
    my_position = game_state["self"][3]

    distance_relevance = opponent_distance_relevance(
        field,
        my_position,
        opponent_position,
    )

    threat = opponent_threat_score(
        game_state,
        opponent_position,
    )

    competition = resource_competition(
        game_state,
        opponent_position,
    )

    return (W_THREAT * threat + W_COMPETITION * competition) * distance_relevance
