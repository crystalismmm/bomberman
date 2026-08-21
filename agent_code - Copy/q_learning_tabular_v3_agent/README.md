# Q-learning tabular V3 agent

V3 is a controlled multiplayer extension of the tested V2 tabular agent.
Its state key remains compact:

```text
(target_mode, target_direction, safe_move_mask,
 bomb_status, wait_status, previous_action)
```

The only semantic extension is the new bomb status `ATTACK`. A bomb is
classified in this order:

1. `UNAVAILABLE` if no bomb is available.
2. `UNSAFE` if placing it leaves no escape route.
3. `ATTACK` if its current blast line contains an opponent.
4. `USELESS` if it hits neither opponent nor crate.
5. `GOOD_LOW` for one crate.
6. `GOOD_HIGH` for at least two crates.

An `ATTACK` bomb gets no direct shaping reward. Instead, the real
`KILLED_OPPONENT` event teaches its value through Q-learning. Crucially, it no
longer receives the `USELESS_BOMB` penalty merely because it hits no crate.

All six actions (`UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, `BOMB`) remain fully
available in action selection and in the TD target. Safety information is part
of the state; it is not used as an action mask.

The included `q_table_classic_v2_train1000_seed42.pkl` is an archived V2
reference only. It is intentionally not named `q_table.pkl` and is incompatible
with V3's model version. V3 therefore starts with a fresh table when trained.
