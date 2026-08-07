"""Callbacks for a compact tabular Q-learning Bomberman agent.

The state deliberately contains only categorical information.  This keeps the
table small enough to learn while separating situations that the former linear
model mixed together (especially GOOD/USELESS/UNSAFE bombs and different WAIT
situations).
"""

from __future__ import annotations

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

MODEL_VERSION = 2
MODEL_FILE = Path(__file__).resolve().with_name("q_table.pkl")

TARGET_ESCAPE = "ESCAPE"
TARGET_COIN = "COIN"
TARGET_BOMBING = "BOMBING"
TARGET_NONE = "NONE"

DIRECTION_HERE = "HERE"
DIRECTION_NONE = "NONE"

BOMB_UNAVAILABLE = "UNAVAILABLE"
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
    """The table key plus metadata used for reward shaping and logging."""

    key: StateKey
    target_position: tuple[int, int] | None
    target_distance: int | None


def setup(self):
    """Initialize history and either create or load the Q-table."""
    self.rng = np.random.default_rng(RNG_SEED)
    self.q_table: dict[StateKey, np.ndarray] = {}

    self.current_round = None
    self.previous_action = None
    self.state_previous_action = None
    self.last_state_key = None

    if self.train:
        self.logger.info("Starting tabular training with an empty Q-table.")
        return

    if not MODEL_FILE.is_file():
        self.logger.warning(
            "No Q-table found at %s. Evaluation starts untrained.",
            MODEL_FILE,
        )
        return

    self.q_table = load_q_table(MODEL_FILE)
    self.logger.info(
        "Loaded %d tabular states from %s.",
        len(self.q_table),
        MODEL_FILE,
    )


def act(self, game_state: dict) -> str:
    """Choose one of all six actions with an epsilon-greedy policy.

    There is intentionally no action-quality or safety mask here.  Even WAIT,
    BOMB, dangerous moves, and currently invalid actions remain learnable.
    """
    round_number = game_state["round"]
    if self.current_round != round_number:
        self.current_round = round_number
        self.previous_action = None
        self.state_previous_action = None
        self.last_state_key = None

    analysis = analyze_state(game_state, self.previous_action)
    q_values = q_values_for_state(self.q_table, analysis.key)

    epsilon = self.epsilon if self.train else EPSILON_EVAL
    if self.rng.random() < epsilon:
        decision_type = "exploration"
        action = str(self.rng.choice(ACTIONS))
    else:
        decision_type = "exploitation"
        maximum = np.max(q_values)
        best_indices = np.flatnonzero(np.isclose(q_values, maximum))
        action = ACTIONS[int(self.rng.choice(best_indices))]

    # The training callback needs the exact state used for this decision.  The
    # previous action is then advanced here as evaluation has no train callback.
    self.last_state_key = analysis.key
    self.state_previous_action = self.previous_action
    self.previous_action = action

    self.logger.debug(
        "Round: %s | Step: %s | Mode: %s | State: %s | "
        "Q-values: %s | Selected: %s | Known states: %d",
        game_state["round"],
        game_state["step"],
        decision_type,
        analysis.key,
        np.round(q_values, 3).tolist(),
        action,
        len(self.q_table),
    )

    return action


def q_values_for_state(
    q_table: dict[StateKey, np.ndarray],
    state_key: StateKey,
) -> np.ndarray:
    """Return Q-values without inserting unseen evaluation states."""
    values = q_table.get(state_key)
    if values is None:
        return np.zeros(len(ACTIONS), dtype=np.float32)
    return values


def load_q_table(path: Path) -> dict[StateKey, np.ndarray]:
    """Load and validate a table written by :func:`save_q_table`."""
    with path.open("rb") as model_file:
        payload = pickle.load(model_file)

    if not isinstance(payload, dict):
        raise ValueError("The saved Q-table payload is not a dictionary.")
    if payload.get("version") != MODEL_VERSION:
        raise ValueError(
            f"Unsupported Q-table version {payload.get('version')!r}; "
            f"expected {MODEL_VERSION}."
        )
    if tuple(payload.get("actions", ())) != ACTIONS:
        raise ValueError("The saved Q-table uses a different action order.")

    raw_table = payload.get("q_table")
    if not isinstance(raw_table, dict):
        raise ValueError("The saved payload contains no valid Q-table.")

    table: dict[StateKey, np.ndarray] = {}
    for key, raw_values in raw_table.items():
        if not isinstance(key, tuple) or len(key) != 6:
            raise ValueError(f"Invalid tabular state key: {key!r}")
        values = np.asarray(raw_values, dtype=np.float32)
        if values.shape != (len(ACTIONS),) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid Q-values for state {key!r}.")
        table[key] = values

    return table


def available_actions(game_state: dict) -> list[str]:
    """Return the complete action space without policy-side restrictions."""
    del game_state
    return list(ACTIONS)


def inside_field(field: np.ndarray, position: tuple[int, int]) -> bool:
    """Return whether a coordinate lies inside the arena."""
    return (
        0 <= position[0] < field.shape[0]
        and 0 <= position[1] < field.shape[1]
    )


def blocked_positions(game_state: dict) -> set[tuple[int, int]]:
    """Return positions occupied by bombs or other agents."""
    bombs = {position for position, _ in game_state["bombs"]}
    others = {other[3] for other in game_state["others"]}
    return bombs | others


def get_blast_tiles(
    field: np.ndarray,
    bomb_position: tuple[int, int],
    power: int = BOMB_POWER,
) -> set[tuple[int, int]]:
    """Return blast tiles using this framework's exact explosion rules.

    Solid walls stop a blast.  Crates do not stop it in the supplied game
    engine, so all crates along a ray can be destroyed.
    """
    blast_tiles = {bomb_position}
    x, y = bomb_position

    for dx, dy in MOVE_DELTAS.values():
        for step in range(1, power + 1):
            tile = (x + dx * step, y + dy * step)
            if not inside_field(field, tile) or field[tile] == -1:
                break
            blast_tiles.add(tile)

    return blast_tiles


def danger_time_map(game_state: dict) -> np.ndarray:
    """Return the earliest bomb timer for every threatened tile.

    A timer of zero means that the tile explodes after the current action.  A
    value of infinity means no currently visible bomb threatens the tile.
    """
    field = game_state["field"]
    danger_time = np.full(field.shape, np.inf, dtype=np.float32)

    for bomb_position, bomb_timer in game_state["bombs"]:
        for tile in get_blast_tiles(field, bomb_position):
            danger_time[tile] = min(danger_time[tile], bomb_timer)

    danger_time[game_state["explosion_map"] > 0] = 0
    return danger_time


def _escape_distance_after_action(
    game_state: dict,
    action: str,
    danger_map: np.ndarray | None = None,
) -> int | None:
    """Return moves until permanent safety after an action, or ``None``."""
    if action not in MOVE_DELTAS and action != "WAIT":
        return None

    field = game_state["field"]
    danger_map = danger_time_map(game_state) if danger_map is None else danger_map
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
        or (new_position in blocked and not (action == "WAIT" and new_position == start))
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


def action_preserves_escape_route(game_state: dict, action: str) -> bool:
    """Return whether an action leaves a timer-valid route to safety."""
    return _escape_distance_after_action(game_state, action) is not None


def safe_move_mask(
    game_state: dict,
    danger_map: np.ndarray | None = None,
) -> int:
    """Encode route-preserving movements in four bits."""
    danger_map = danger_time_map(game_state) if danger_map is None else danger_map
    mask = 0
    for index, action in enumerate(MOVE_DELTAS):
        if _escape_distance_after_action(game_state, action, danger_map) is not None:
            mask |= 1 << index
    return mask


def direction_to_safety(
    game_state: dict,
    danger_map: np.ndarray | None = None,
) -> str | None:
    """Return the first action of the shortest timer-valid escape route."""
    danger_map = danger_time_map(game_state) if danger_map is None else danger_map
    start = game_state["self"][3]
    if not np.isfinite(danger_map[start]):
        return None

    candidates: list[tuple[int, int, str]] = []
    for order, action in enumerate((*MOVE_DELTAS.keys(), "WAIT")):
        distance = _escape_distance_after_action(game_state, action, danger_map)
        if distance is not None:
            candidates.append((distance, order, action))

    if not candidates:
        return None
    return min(candidates)[2]


def count_crates_in_blast(
    field: np.ndarray,
    bomb_position: tuple[int, int],
) -> int:
    """Return how many currently present crates a bomb would destroy."""
    return sum(field[tile] == 1 for tile in get_blast_tiles(field, bomb_position))


def can_escape_from_position(
    game_state: dict,
    bomb_position: tuple[int, int],
) -> bool:
    """Return whether a newly placed bomb leaves an escape route."""
    if bomb_position in {position for position, _ in game_state["bombs"]}:
        return False

    name, score, bombs_left, _ = game_state["self"]
    simulated_state = dict(game_state)
    simulated_state["self"] = (name, score, bombs_left, bomb_position)
    simulated_state["bombs"] = list(game_state["bombs"]) + [
        (bomb_position, BOMB_TIMER - 1)
    ]

    return any(
        _escape_distance_after_action(simulated_state, action) is not None
        for action in MOVE_DELTAS
    )


def can_escape_after_bomb(game_state: dict) -> bool:
    """Return whether BOMB is available and escapable at the current tile."""
    _, _, bombs_left, position = game_state["self"]
    return bool(bombs_left) and can_escape_from_position(game_state, position)


def bomb_status(game_state: dict) -> str:
    """Classify BOMB without removing it from the action space."""
    _, _, bombs_left, position = game_state["self"]
    if not bombs_left:
        return BOMB_UNAVAILABLE
    if not can_escape_after_bomb(game_state):
        return BOMB_UNSAFE
    crates_in_blast = count_crates_in_blast(game_state["field"], position)
    if crates_in_blast == 0:
        return BOMB_USELESS
    elif crates_in_blast == 1:
        return BOMB_GOOD_LOW
    return BOMB_GOOD_HIGH


def wait_status(
    game_state: dict,
    movement_mask: int | None = None,
    danger_map: np.ndarray | None = None,
) -> str:
    """Classify WAIT as necessary, unnecessary, or unsafe."""
    danger_map = danger_time_map(game_state) if danger_map is None else danger_map
    movement_mask = (
        safe_move_mask(game_state, danger_map)
        if movement_mask is None
        else movement_mask
    )
    wait_is_safe = (
        _escape_distance_after_action(game_state, "WAIT", danger_map) is not None
    )
    if not wait_is_safe:
        return WAIT_UNSAFE
    if movement_mask == 0:
        return WAIT_NECESSARY
    return WAIT_UNNECESSARY


def _first_direction(
    start: tuple[int, int],
    target: tuple[int, int],
    parents: dict[tuple[int, int], tuple[int, int] | None],
) -> str:
    """Backtrack a BFS tree to obtain the first movement action."""
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


def reachable_positions(
    game_state: dict,
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], tuple[int, int] | None],
]:
    """Return static BFS distances and parents from the agent position."""
    field = game_state["field"]
    start = game_state["self"][3]
    blocked = blocked_positions(game_state)
    blocked.discard(start)

    distances = {start: 0}
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = deque([start])

    while queue:
        position = queue.popleft()
        for dx, dy in MOVE_DELTAS.values():
            neighbor = (position[0] + dx, position[1] + dy)
            if (
                inside_field(field, neighbor)
                and field[neighbor] == 0
                and neighbor not in blocked
                and neighbor not in distances
            ):
                distances[neighbor] = distances[position] + 1
                parents[neighbor] = position
                queue.append(neighbor)

    return distances, parents


def nearest_reachable_coin(
    game_state: dict,
    distances: dict[tuple[int, int], int] | None = None,
    parents: dict[tuple[int, int], tuple[int, int] | None] | None = None,
) -> tuple[str | None, tuple[int, int] | None, int | None]:
    """Return direction, position, and distance of the nearest visible coin."""
    if distances is None or parents is None:
        distances, parents = reachable_positions(game_state)

    reachable_coins = [coin for coin in game_state["coins"] if coin in distances]
    if not reachable_coins:
        return None, None, None

    target = min(reachable_coins, key=lambda coin: (distances[coin], coin))
    direction = _first_direction(game_state["self"][3], target, parents)
    return direction, target, distances[target]


def nearest_safe_bombing_position(
    game_state: dict,
    distances: dict[tuple[int, int], int] | None = None,
    parents: dict[tuple[int, int], tuple[int, int] | None] | None = None,
) -> tuple[str | None, tuple[int, int] | None, int | None]:
    """Return the best useful and escapable bombing position.

    More crates increase the score while distance discounts it.  The score is
    used only to select a categorical target direction; it is not part of the
    tabular state and therefore cannot cause linear feature cross-talk.
    """
    if distances is None or parents is None:
        distances, parents = reachable_positions(game_state)

    field = game_state["field"]
    best_target = None
    best_distance = None
    best_rank = None

    for position, distance in distances.items():
        crate_count = count_crates_in_blast(field, position)
        if crate_count == 0 or not can_escape_from_position(game_state, position):
            continue

        score = crate_count * BOMBING_DISTANCE_DECAY ** distance
        rank = (
            score,
            crate_count,
            -distance,
            -position[0],
            -position[1],
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_target = position
            best_distance = distance

    if best_target is None or best_distance is None:
        return None, None, None

    direction = _first_direction(game_state["self"][3], best_target, parents)
    return direction, best_target, best_distance


def analyze_state(
    game_state: dict,
    previous_action: str | None = None,
) -> StateAnalysis:
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
        direction, position, distance = nearest_reachable_coin(
            game_state,
            distances,
            parents,
        )
        if position is not None:
            target_mode = TARGET_COIN
            target_direction = direction or DIRECTION_NONE
            target_position = position
            target_distance = distance
        else:
            direction, position, distance = nearest_safe_bombing_position(
                game_state,
                distances,
                parents,
            )
            if position is not None:
                target_mode = TARGET_BOMBING
                target_direction = direction or DIRECTION_NONE
                target_position = position
                target_distance = distance

    key: StateKey = (
        target_mode,
        target_direction,
        movement_mask,
        bomb_status(game_state),
        wait_status(game_state, movement_mask, danger_map),
        previous_action if previous_action in ACTIONS else PREVIOUS_NONE,
    )
    return StateAnalysis(key, target_position, target_distance)


def state_to_key(
    game_state: dict,
    previous_action: str | None = None,
) -> StateKey | None:
    """Public compact state conversion used by tests and training."""
    if game_state is None:
        return None
    return analyze_state(game_state, previous_action).key
