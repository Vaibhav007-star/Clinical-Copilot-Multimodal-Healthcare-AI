"""
Vercel Serverless Function Entrypoint for Clinical Copilot.
Directs ASGI requests to the FastAPI application handler.
"""
import sys
from pathlib import Path

# Add both root directory and api directory to Python path
API_DIR = Path(__file__).resolve().parent
ROOT_DIR = API_DIR.parent

for p in (str(ROOT_DIR), str(API_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from api.server import app

# Top-level ASGI exports for Vercel
handler = app
app = app
