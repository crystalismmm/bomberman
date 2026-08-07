# Compact tabular Q-learning agent V2

This agent replaces the former linear feature model with a sparse Q-table.  A
state key contains only six categorical components:

1. target mode (`ESCAPE`, `COIN`, `BOMBING`, `NONE`),
2. target direction,
3. a four-bit mask of route-preserving moves,
4. bomb status (`UNAVAILABLE`, `GOOD_LOW`, `GOOD_HIGH`, `USELESS`, `UNSAFE`),
5. wait status (`NECESSARY`, `UNNECESSARY`, `UNSAFE`),
6. the previous action.

All six game actions stay available to the policy.  The agent learns from
rewards that an invalid, unsafe, or unproductive action is undesirable; there
is no policy-side safety or bomb-quality action mask.

`GOOD_LOW` means that a bomb would hit exactly one crate; `GOOD_HIGH` means at
least two.  The bombing target maximizes
`crate_count * 0.85 ** distance`, so a somewhat more distant position is chosen
only when its expected crate yield compensates for the additional movement.

The trained model is saved as `q_table.pkl`.  V2 uses model format version 2
because its state keys differ from V1.  The archived
`q_table_classic_v1_train1000_seed42.pkl` is provided only as a comparison and
cannot be loaded by the V2 callbacks.

Starting a command with `--train 1` deliberately creates a new empty V2 table.
Evaluation loads `q_table.pkl` after V2 has been trained.

Suggested reproducibility experiment from the project root:

```powershell
python main.py play --agents q_learning_tabular_v2_agent --train 1 --scenario classic --n-rounds 1000 --no-gui --seed 42 --save-stats results/tabular_v2_classic_train_1000_seed42.json
python main.py play --agents q_learning_tabular_v2_agent --scenario classic --n-rounds 100 --no-gui --seed 123 --save-stats results/tabular_v2_classic_eval_100_seed123.json
```
