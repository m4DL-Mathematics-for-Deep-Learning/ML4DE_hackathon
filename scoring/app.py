from flask import Flask, request, jsonify, send_from_directory
import os
import json
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
LEADERBOARD_FILE = 'leaderboard.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize leaderboard file if not exists
if not os.path.exists(LEADERBOARD_FILE):
    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump([], f)

# Dummy scoring function
def score_file(filepath):
    return len(open(filepath).readlines())  # Example: Count lines in file

# Load leaderboard
def load_leaderboard():
    with open(LEADERBOARD_FILE, 'r') as f:
        return json.load(f)

# Save leaderboard and push to GitHub
def save_leaderboard(data):
    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    
    # Git commands to push changes
    subprocess.run(['git', 'pull'])  # Ensure latest version
    subprocess.run(['git', 'add', LEADERBOARD_FILE])
    subprocess.run(['git', 'commit', '-m', 'Updated leaderboard'])
    subprocess.run(['git', 'push'])

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Process the file
    score = score_file(filepath)
    
    # Update leaderboard
    leaderboard = load_leaderboard()
    leaderboard.append({'filename': file.filename, 'score': score})
    leaderboard = sorted(leaderboard, key=lambda x: x['score'], reverse=True)[:10]  # Keep top 10
    save_leaderboard(leaderboard)
    
    return jsonify({'filename': file.filename, 'score': score})

@app.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    leaderboard = load_leaderboard()
    return jsonify(leaderboard)

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)