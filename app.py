from flask import Flask, render_template
import numpy as np
from aux.ks_eval import scoring_ks  #  Import KS scoring functions
from aux.lorenz_eval import scoring_lorenz  # Import Lorenz scoring functions
app = Flask(__name__)
import os


def get_team_scores(team_folder):
    """Get scores for a single team"""
    try:
        # Define paths for this team
        ks_truth_file = os.path.join('data', 'ks_truth.npy')
        ks_prediction_file = os.path.join(team_folder, 'ks_prediction.npy')
        lorenz_truth_file = os.path.join('data', 'lorenz_truth.npy')
        lorenz_prediction_file = os.path.join(team_folder, 'lorenz_prediction.npy')
        teamname_file = os.path.join(team_folder, 'teamname.txt')
        
        # Skip if predictions don't exist
        if not os.path.exists(ks_prediction_file) or not os.path.exists(lorenz_prediction_file):
            return None
            
        # Load data
        ks_truth = np.load(ks_truth_file)
        ks_prediction = np.load(ks_prediction_file)
        lorenz_truth = np.load(lorenz_truth_file)
        lorenz_prediction = np.load(lorenz_prediction_file)

        # Parameters for scoring
        k = 20   # Number of snapshots
        ks_modes = 20  # Need modes strictly less than m/2 for KS
        lorenz_modes = 1000  # For Lorenz
        
        # Run scoring for both KS and Lorenz
        ks_E1, ks_E2 = scoring_ks(ks_truth, ks_prediction, k, ks_modes)
        lorenz_E1, lorenz_E2 = scoring_lorenz(lorenz_truth, lorenz_prediction, k, lorenz_modes)
        
        # Read team name from file
        with open(teamname_file, "r") as file:
            team_name = file.read().strip()
            
        return {
            'name': team_name,
            'folder': team_folder,
            'ks_E1': ks_E1,
            'ks_E2': ks_E2,
            'lorenz_E1': lorenz_E1,
            'lorenz_E2': lorenz_E2,
            'total': ks_E1 + ks_E2 + lorenz_E1 + lorenz_E2  # Combined score for ranking
        }
    except Exception as e:
        print(f"Error processing team {team_folder}: {str(e)}")
        return None

@app.route('/')
def index():
    """Load all team results and display ranking"""
    # Find all team folders
    team_folders = [d for d in os.listdir('./team_entries') if d.startswith('team') and os.path.isdir(os.path.join('./team_entries', d))]

    # Get scores for each team
    team_scores = []
    for folder in team_folders:
        scores = get_team_scores(os.path.join('./team_entries', folder))
        if scores:
            team_scores.append(scores)
    
    # Sort teams by total score (higher is better)
    team_scores.sort(key=lambda x: x['total'], reverse=True)
    
    return render_template(
        "index.html",
        teams=team_scores,
        timestamp=np.datetime64('now') + np.timedelta64(1, 'h')
    )