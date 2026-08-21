# Compact tabular Q-learning agent

This agent replaces the former linear feature model with a sparse Q-table.  A
state key contains only six categorical components:

1. target mode (`ESCAPE`, `COIN`, `BOMBING`, `NONE`),
2. target direction,
3. a four-bit mask of route-preserving moves,
4. bomb status (`UNAVAILABLE`, `GOOD`, `USELESS`, `UNSAFE`),
5. wait status (`NECESSARY`, `UNNECESSARY`, `UNSAFE`),
6. the previous action.

All six game actions stay available to the policy.  The agent learns from
rewards that an invalid, unsafe, or unproductive action is undesirable; there
is no policy-side safety or bomb-quality action mask.

The trained model is saved as `q_table.pkl` in this folder.  The supplied table
was trained for 1000 rounds on `loot-crate` with seed 42.  An unchanged backup
is included as `q_table_train1000_seed42.pkl`.

Starting a command with `--train 1` deliberately creates a new empty table and
will eventually overwrite `q_table.pkl`.  Evaluation loads `q_table.pkl`.
Restore the supplied benchmark table in PowerShell with:

```powershell
Copy-Item agent_code/q_learning_tabular_agent/q_table_train1000_seed42.pkl agent_code/q_learning_tabular_agent/q_table.pkl -Force
```

Suggested reproducibility experiment from the project root:

```powershell
python main.py play --agents q_learning_tabular_agent --train 1 --scenario loot-crate --n-rounds 1000 --no-gui --seed 42 --save-stats results/tabular_train_1000_seed42.json
python main.py play --agents q_learning_tabular_agent --scenario loot-crate --n-rounds 100 --no-gui --seed 123 --save-stats results/tabular_eval_100_seed123.json
```

For the supplied table, that evaluation produced 3277 coins, 8612 destroyed
crates, 3691 placed bombs, 2 suicides, and 723 WAIT actions in 39516 steps.
