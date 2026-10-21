"""
Vercel Serverless Function Entrypoint for Clinical Copilot.
Directs ASGI requests to the FastAPI application handler with diagnostic fallback.
"""
import sys
import traceback
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from api.server import app
    handler = app
except Exception as e:
    err_tb = traceback.format_exc()
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse
    err_app = FastAPI()

    @err_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    def catch_all(path: str):
        return PlainTextResponse(f"Startup Crash in api/index.py:\n{err_tb}", status_code=500)

    handler = err_app
    app = err_app
