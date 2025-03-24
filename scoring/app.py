from flask import Flask, request, jsonify, send_from_directory
import os
import json
import subprocess

app = Flask(__name__)

# Define folder locations
UPLOAD_FOLDER = 'uploads'
LEADERBOARD_FILE = 'leaderboard.json'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize leaderboard file if not exists
if not os.path.exists(LEADERBOARD_FILE):
    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump([], f)

# Dummy scoring function (example: count number of lines in the file)
def score_file(filepath):
    return len(open(filepath).readlines())  # Example scoring method

# Load leaderboard
def load_leaderboard():
    with open(LEADERBOARD_FILE, 'r') as f:
        return json.load(f)

# Save leaderboard and push to GitHub
def save_leaderboard(data):
    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(data, f, indent=4)

    os.sync()  # Ensure data is written to disk

    # GitHub push setup
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Token must be set in Render environment variables
    REPO_URL = f"https://oauth2:{GITHUB_TOKEN}@github.com/YOUR_USERNAME/YOUR_REPO.git"

    subprocess.run(['git', 'config', '--global', 'user.email', 'your-email@example.com'])
    subprocess.run(['git', 'config', '--global', 'user.name', 'Your Render Bot'])

    # Pull, commit, and push updates
    subprocess.run(['git', 'pull', REPO_URL])
    subprocess.run(['git', 'add', LEADERBOARD_FILE])
    subprocess.run(['git', 'commit', '-m', 'Updated leaderboard'])
    subprocess.run(['git', 'push', REPO_URL])

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'category' not in request.form:
        return jsonify({'error': 'Missing file or category'}), 400
    
    file = request.files['file']
    category = request.form['category']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Process the file
    score = score_file(filepath)

    # Update leaderboard
    leaderboard = load_leaderboard()
    leaderboard.append({'filename': file.filename, 'score': score, 'category': category})
    leaderboard = sorted(leaderboard, key=lambda x: x['score'], reverse=True)[:10]  # Keep top 10
    save_leaderboard(leaderboard)

    return jsonify({'filename': file.filename, 'score': score, 'category': category})

@app.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    leaderboard = load_leaderboard()
    return jsonify(leaderboard)

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)
