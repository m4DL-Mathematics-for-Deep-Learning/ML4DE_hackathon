from flask import Flask, render_template, send_file

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_teamname")
def get_teamname():
    try:
        return send_file("team1/teamname.txt", as_attachment=False)
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)
