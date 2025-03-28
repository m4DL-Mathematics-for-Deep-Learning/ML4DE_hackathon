from flask import Flask, render_template
import numpy as np
from ks_scoring import scoring  #  Import scoring functions
app = Flask(__name__)
import os


# Define paths
TEAM_FOLDER = 'team1/'
TRUTH_FILE = os.path.join(TEAM_FOLDER, 'truth.npy')
PREDICTION_FILE = os.path.join(TEAM_FOLDER, 'prediction.npy')
TEAMNAME_FILE = os.path.join(TEAM_FOLDER, 'teamname.txt')

@app.route('/')
def index():
    """Loads truth & prediction, runs scoring, and returns results."""
    #try:
    #    if not os.path.exists(TRUTH_FILE) or not os.path.exists(PREDICTION_FILE):
    #        return "Missing truth or prediction file in ../team1/", 500
        
    # Load data
    truth = np.load(TRUTH_FILE)
    prediction = np.load(PREDICTION_FILE)

    print(np.random.rand(1,1))
    # Parameters for scoring
    k = 20   # Number of snapshots
    modes = 100

    # Run scoring
    E1, E2 = scoring(truth, prediction, k, modes)

    # Read team name from file
    with open("team1/teamname.txt", "r") as file:
        team1 = file.read()
    return render_template("index.html", name=team1, E1=E1, E2=E2)
#except:
#    return "Error!" # TODO: Handle error