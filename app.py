"""
Crop Choice Intelligence - Web Application Entrypoint for Vercel / Flask
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from flask_app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
