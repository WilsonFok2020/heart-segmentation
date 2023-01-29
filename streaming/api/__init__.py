from flask import Flask
from flask import current_app, render_template, stream_template, Response, request

app = Flask(__name__)
app.debug=True

with app.app_context():
    """
    Default setting
    """
    current_app.a = 1.0
    current_app.b = 100.0
    current_app.lower = -4.0
    current_app.upper = 4.0
    current_app.maxiter = 10
    current_app.initialGuess = "0.5,0.5"

from api import routes

