"""
Vercel Serverless Function Entrypoint for Clinical Copilot.
Directs ASGI requests to the FastAPI application handler with diagnostic introspection.
"""
import sys
import os
import traceback
from pathlib import Path
from fastapi import FastAPI, Response

API_DIR = Path(__file__).resolve().parent
ROOT_DIR = API_DIR.parent

for p in (str(ROOT_DIR), str(API_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

app = FastAPI()

@app.get("/test-health")
def test_health():
    return {
        "status": "ok",
        "python": sys.version,
        "cwd": os.getcwd(),
        "files": os.listdir("."),
        "sys_path": sys.path[:5]
    }

@app.get("/debug-import")
def debug_import():
    try:
        import api.server
        return {"status": "import_success", "routes": [r.path for r in api.server.app.routes]}
    except Exception as e:
        return Response(content=traceback.format_exc(), media_type="text/plain", status_code=500)

# Try mounting the real app
try:
    from api.server import app as real_app
    app = real_app
except Exception as e:
    startup_error = traceback.format_exc()
    @app.get("/{path:path}")
    def fallback(path: str):
        return Response(content=f"Startup Crash in api/server:\n{startup_error}", media_type="text/plain", status_code=500)

handler = app
