from flask import Flask, request, jsonify, send_from_directory, send_file

app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory('', 'index.html')

@app.route("/get_teamname")
def get_teamname():
    try:
        return send_file("../team1/teamname.txt", as_attachment=False)
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)
