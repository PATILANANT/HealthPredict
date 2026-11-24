"""Start the web server:  python run.py   (then open http://127.0.0.1:5000)

For first-time setup use start.bat (Windows) or ./start.sh (macOS/Linux) instead.
"""
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1", threaded=True)
