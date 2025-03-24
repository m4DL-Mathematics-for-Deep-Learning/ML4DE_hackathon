import numpy as np
import os

TEAM_FOLDER = "team1"  # Adjust if needed
TRUTH_FILE = os.path.join(TEAM_FOLDER, "prediction.npy") # truth.npy

# Ensure the team folder exists
os.makedirs(TEAM_FOLDER, exist_ok=True)

# Generate a random matrix as the "truth" data
rows, cols = 100, 200  # Adjust size as needed
truth_data = np.random.rand(rows, cols)

# Save to `truth.npy`
np.save(TRUTH_FILE, truth_data)

print(f"Generated random truth file: {TRUTH_FILE}")
