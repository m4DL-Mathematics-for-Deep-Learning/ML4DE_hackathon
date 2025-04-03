from flask import Flask, render_template
import numpy as np
from ks_scoring import scoring  #  Import scoring functions
app = Flask(__name__)
import os


def get_team_scores(team_folder):
    """Get scores for a single team"""
    try:
        # Define paths for this team
        truth_file = os.path.join('data', 'truth.npy')
        prediction_file = os.path.join(team_folder, 'prediction.npy')
        teamname_file = os.path.join(team_folder, 'teamname.txt')
        
        # Skip if prediction doesn't exist
        if not os.path.exists(prediction_file):
            return None
            
        # Load data
        truth = np.load(truth_file)
        prediction = np.load(prediction_file)
        
        # Parameters for scoring
        k = 20   # Number of snapshots
        modes = 20  # Need modes strictly less than m/2
        
        # Run scoring
        E1, E2 = scoring(truth, prediction, k, modes)
        
        # Read team name from file
        with open(teamname_file, "r") as file:
            team_name = file.read().strip()
            
        return {
            'name': team_name,
            'folder': team_folder,
            'E1': E1,
            'E2': E2,
            'total': E1 + E2  # Combined score for ranking
        }
    except Exception as e:
        print(f"Error processing team {team_folder}: {str(e)}")
        return None

@app.route('/')
def index():
    """Load all team results and display ranking"""
    # Find all team folders
    team_folders = [d for d in os.listdir('.') if d.startswith('team') and os.path.isdir(d)]
    
    # Get scores for each team
    team_scores = []
    for folder in team_folders:
        scores = get_team_scores(folder)
        if scores:
            team_scores.append(scores)
    
    # Sort teams by total score (lower is better)
    team_scores.sort(key=lambda x: x['total'])
    
    return render_template(
        "index.html",
        teams=team_scores,
        timestamp=np.datetime64('now')
    )