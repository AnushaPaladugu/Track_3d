"""
Simple script to test if the Flask server can start properly
"""
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Flask server is working!"

if __name__ == '__main__':
    print("Starting test server on http://127.0.0.1:5000/")
    print("Press Ctrl+C to stop the server")
    app.run(host="127.0.0.1", port=5000, debug=True)
