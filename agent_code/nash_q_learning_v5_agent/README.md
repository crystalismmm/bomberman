# Nash Q-learning V5 Agent – unrestricted actions

V5 is a separate successor to the safety-masked V4 agent. It is intended for
experiments in which the agent must learn to avoid bad actions instead of being
prevented from selecting them.

## Action policy

At every step the complete action space is returned:

```python
["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
```

This applies to exploration, the solo Q-policy, Nash play, and future-value
calculation. Moves into walls, unavailable bombs, unnecessary waiting, unsafe
moves, and unsafe bombs are therefore still possible.

Safety calculations remain part of the state representation and reward
shaping. They inform learning but never remove an action.

## Main changes compared with the original Nash agent

- A separate solo Q-table provides a stable baseline when no opponent is
  relevant or the joint matrix is still sparse.
- Nash play is used only for a nearby or directly threatening opponent.
- Joint-action visit counts determine when a Nash matrix is reliable enough.
- Unknown opponent actions do not update all opponent-action columns.
- Untried own actions retain a neutral value so negative experience encourages
  the agent to test alternatives.
- Invalid, unsafe movement, unsafe waiting, and unsafe bombing actions receive
  explicit negative rewards.
- Successful escape from danger receives a positive shaping reward.
- Training continues an existing V5 table and epsilon schedule automatically.

V4 tables are intentionally incompatible with V5 because the action policy and
reward system differ.

## PowerShell commands

Run the commands from the Bomberman project directory.

Smoke training:

```powershell
python main.py play --agents nash_q_learning_v5_agent rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --seed 42 --n-rounds 10 --no-gui --save-stats nash_v5_smoke_train_10_seed42.json
```

Pilot training:

```powershell
python main.py play --agents nash_q_learning_v5_agent rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --seed 42 --n-rounds 1000 --no-gui --save-stats nash_v5_pilot_train_1000_seed42.json
```

Main training:

```powershell
python main.py play --agents nash_q_learning_v5_agent rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --seed 43 --n-rounds 10000 --no-gui --save-stats nash_v5_train_10000_seed43.json
```

The commands continue the same model. To start completely from scratch, delete
only the V5 table:

```powershell
Remove-Item .\agent_code\nash_q_learning_v5_agent\nash_q_table.pkl
```

Evaluation:

```powershell
python main.py play --agents nash_q_learning_v5_agent rule_based_agent rule_based_agent rule_based_agent --train 0 --scenario classic --seed 123 --n-rounds 500 --no-gui --save-stats nash_v5_eval_500_seed123.json
```
