from flask import *

app=Flask(__name__)

@app.route("/")
def upload(user):
    return render_template("accident_detected.html")

if __name__ == "__main__":
    app.run()