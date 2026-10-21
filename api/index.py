"""
Vercel Serverless Function Entrypoint for Clinical Copilot.
Directs ASGI requests to the FastAPI application handler with top-level diagnostic safety.
"""
import sys
import traceback
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# Add both root directory and api directory to Python path
API_DIR = Path(__file__).resolve().parent
ROOT_DIR = API_DIR.parent

for p in (str(ROOT_DIR), str(API_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import_error = None
try:
    from api.server import app
except Exception as e:
    import_error = traceback.format_exc()
    app = FastAPI()

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    def catch_all(path: str):
        return PlainTextResponse(f"Startup Crash in api/index.py:\n{import_error}", status_code=500)

handler = app
