"""
Vercel Serverless Function Entrypoint for Clinical Copilot.
Directs ASGI requests to the FastAPI application handler.
"""
import sys
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.server import app

# Handler export for Vercel serverless ASGI execution
handler = app
app = app
