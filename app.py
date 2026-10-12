"""
Root entrypoint for deployment (Render, Railway, etc.).
Exports the Flask application instance from backend.app.
"""
from backend.app import app

if __name__ == "__main__":
    app.run()
