from flask import Flask
app = Flask(__name__)
import os


# Define paths
TEAM_FOLDER = '../team1/'
TRUTH_FILE = os.path.join(TEAM_FOLDER, 'truth.npy')
PREDICTION_FILE = os.path.join(TEAM_FOLDER, 'prediction.npy')
TEAMNAME_FILE = os.path.join(TEAM_FOLDER, 'teamname.txt')

@app.route('/')
def hello_world():
    return 'Hello, World!'