import os
from flask import Flask
from flask_session import Session
from server.routes import register_routes

application_path = os.path.dirname(os.path.realpath(__file__))
template_dir = os.path.join(application_path, 'server', 'frontend')

app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.urandom(24)

# --- SERVER SESSION CONFIGURATION ---
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session/"
Session(app)
# -------------------------------------------

register_routes(app)

if __name__ == '__main__':
    if not os.path.exists("./flask_session/"):
        os.makedirs("./flask_session/")

    # If you want to implement new things or patch bugs, set debugging to True.
    # The performance will be slower, but you will get more detailed error messages and automatic reloading of the
    # server on code changes.
    app.run(debug=True, host='127.0.0.1', port=5000)
