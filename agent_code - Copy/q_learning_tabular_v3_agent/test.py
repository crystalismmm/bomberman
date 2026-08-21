import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Einstellungen
# --------------------------------------------------

FILE_PATH = "agent_code/q_learning_tabular_v3_agent/q_table.pkl"

N_HEATMAP_STATES = 50


# --------------------------------------------------
# Datei laden
# --------------------------------------------------

with open(FILE_PATH, "rb") as f:
    data = pickle.load(f)

print("Version:", data["version"])

ACTIONS = list(data["actions"])
q_table = data["q_table"]

print("Actions:", ACTIONS)
print("Anzahl States:", len(q_table))


# --------------------------------------------------
# Q-Tabelle in DataFrame umwandeln
# --------------------------------------------------

rows = []

for state, q_values in q_table.items():

    row = {
        "state": str(state),

        # einzelne Bestandteile des States
        "feature_1": state[0],
        "feature_2": state[1],
        "feature_3": state[2],
        "feature_4": state[3],
        "feature_5": state[4],
        "feature_6": state[5],
    }

    # Q-Werte den Actions zuordnen
    for action, q_value in zip(ACTIONS, q_values):
        row[action] = q_value

    rows.append(row)


df = pd.DataFrame(rows)


# --------------------------------------------------
# Beste Aktion bestimmen
# --------------------------------------------------

df["best_action"] = df[ACTIONS].idxmax(axis=1)
df["best_q_value"] = df[ACTIONS].max(axis=1)


# --------------------------------------------------
# Tabelle anzeigen
# --------------------------------------------------

print("\nErste 20 States:\n")

print(
    df[
        [
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
            "feature_6",
            *ACTIONS,
            "best_action",
            "best_q_value",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# --------------------------------------------------
# Tabelle als CSV speichern
# --------------------------------------------------

df.to_csv(
    "q_table_visualized.csv",
    index=False
)

print("\nGespeichert als q_table_visualized.csv")


# --------------------------------------------------
# Interessanteste States bestimmen
# --------------------------------------------------

# größter absoluter Q-Wert jedes States
df["max_abs_q"] = df[ACTIONS].abs().max(axis=1)

# States mit den stärksten Q-Werten auswählen
df_plot = (
    df
    .sort_values("max_abs_q", ascending=False)
    .head(N_HEATMAP_STATES)
    .copy()
)


# --------------------------------------------------
# Heatmap vorbereiten
# --------------------------------------------------

heatmap_data = df_plot[ACTIONS].to_numpy()


# kurze State-Namen für y-Achse
state_labels = []

for _, row in df_plot.iterrows():

    label = (
        f"{row['feature_1']} | "
        f"{row['feature_2']} | "
        f"{row['feature_3']} | "
        f"{row['feature_4']} | "
        f"{row['feature_5']} | "
        f"{row['feature_6']}"
    )

    state_labels.append(label)


# --------------------------------------------------
# Heatmap
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 12))

image = ax.imshow(
    heatmap_data,
    aspect="auto",
)

fig.colorbar(
    image,
    ax=ax,
    label="Q-value",
)

ax.set_xticks(
    np.arange(len(ACTIONS))
)

ax.set_xticklabels(
    ACTIONS
)

ax.set_yticks(
    np.arange(len(state_labels))
)

ax.set_yticklabels(
    state_labels,
    fontsize=7
)

ax.set_xlabel("Action")
ax.set_ylabel("State")

ax.set_title(
    f"Top {len(df_plot)} States nach größtem |Q|-Wert"
)

plt.tight_layout()
plt.show()


FILE_PATH = "agent_code/q_learning_tabular_v3_agent/q_table.pkl"
OUTPUT_FILE = "q_table.csv"


with open(FILE_PATH, "rb") as f:
    data = pickle.load(f)

actions = list(data["actions"])
q_table = data["q_table"]

rows = []

for state, q_values in q_table.items():
    row = {
        "state": str(state),
        "feature_1": state[0],
        "feature_2": state[1],
        "feature_3": state[2],
        "feature_4": state[3],
        "feature_5": state[4],
        "feature_6": state[5],
    }

    for action, q_value in zip(actions, q_values):
        row[action] = q_value

    rows.append(row)

df = pd.DataFrame(rows)

# optional: beste Aktion hinzufügen
df["best_action"] = df[actions].idxmax(axis=1)
df["best_q_value"] = df[actions].max(axis=1)

df.to_csv(OUTPUT_FILE, index=False)

print(f"CSV gespeichert: {OUTPUT_FILE}")
print(f"Anzahl States: {len(df)}")