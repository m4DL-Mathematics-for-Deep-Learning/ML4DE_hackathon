from flask import Flask, render_template
app = Flask(__name__)
import os


# Define paths
TEAM_FOLDER = '../team1/'
TRUTH_FILE = os.path.join(TEAM_FOLDER, 'truth.npy')
PREDICTION_FILE = os.path.join(TEAM_FOLDER, 'prediction.npy')
TEAMNAME_FILE = os.path.join(TEAM_FOLDER, 'teamname.txt')

@app.route('/')
def hello_world():
    return 'Hello, World!123213'

def index():
    return render_template("index.html", E1=1.0, E2=1.0)