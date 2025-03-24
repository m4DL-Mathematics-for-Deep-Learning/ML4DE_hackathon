from flask import Flask, request, jsonify, send_from_directory, send_file, render_template, make_response
import os
import numpy as np
from ks_scoring import scoring  # ✅ Import scoring function

app = Flask(__name__)

# Define paths
TEAM_FOLDER = '../team1/'
TRUTH_FILE = os.path.join(TEAM_FOLDER, 'truth.npy')
PREDICTION_FILE = os.path.join(TEAM_FOLDER, 'prediction.npy')
TEAMNAME_FILE = os.path.join(TEAM_FOLDER, 'teamname.txt')

# Parameters for scoring
k = 20   # Number of snapshots
modes = 100

@app.route("/")
def index():
    """Loads truth & prediction, runs scoring, and returns results."""
    try:
        if not os.path.exists(TRUTH_FILE) or not os.path.exists(PREDICTION_FILE):
            return "Missing truth or prediction file in ../team1/", 500
        
        # Load data
        truth = np.load(TRUTH_FILE)
        prediction = np.load(PREDICTION_FILE)

        # Run scoring
        E1, E2 = scoring(truth, prediction, k, modes)

        response = make_response(render_template("index.html", E1=E1, E2=E2))
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response

    except Exception as e:
        return str(e), 500

@app.route("/get_teamname")
def get_teamname():
    """Returns the team name from teamname.txt"""
    try:
        return send_file(TEAMNAME_FILE, as_attachment=False)
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)
