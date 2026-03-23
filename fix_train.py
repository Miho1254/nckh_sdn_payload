import sys

with open('ai_model/train_actor_critic.py', 'r') as f:
    text = f.read()

# 1. Update load_v3_data
text = text.replace(
    "y_path = os.path.join(DATA_DIR, 'y_v3.npy')",
    "y_path = os.path.join(DATA_DIR, 'y_v3.npy')\n    s_path = os.path.join(DATA_DIR, 'scenarios_v3.npy')"
)
text = text.replace(
    "y = np.load(y_path)",
    "y = np.load(y_path)\n    scen = np.load(s_path) if os.path.exists(s_path) else np.array(['UNKNOWN']*len(y))"
)
text = text.replace(
    "return X, y, metadata",
    "return X, y, scen, metadata"
)

# 2. Update __main__
text = text.replace(
    "X, y, metadata = load_v3_data()",
    "X, y, scen, metadata = load_v3_data()"
)
text = text.replace(
    "phase2_offline_training(\n            agent, X, y, env",
    "phase2_offline_training(\n            agent, X, y, scen, env"
)
text = text.replace(
    "phase3_constraint_tuning(\n            agent, X, y, env",
    "phase3_constraint_tuning(\n            agent, X, y, scen, env"
)

with open('ai_model/train_actor_critic.py', 'w') as f:
    f.write(text)
print("Applied phase 1 replacements")
