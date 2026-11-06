"""Local web server for the ball simulator. Standard library only.

    uv run python webapp/server.py [--port 8000]

Serves webapp/frontend/ and a small JSON API backed by sim_api.py.
"""

import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import sim_api

FRONTEND = Path(__file__).resolve().parent / "frontend"
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8"}
POST_ROUTES = {"/api/room": sim_api.preview_room, "/api/simulate": sim_api.simulate_runs,
               "/api/save": sim_api.save_dataset}
MAX_REQUEST_BYTES = 1_000_000


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict):
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/init":
            return self._send_json(200, sim_api.init())
        target = (FRONTEND / (path.lstrip("/") or "index.html")).resolve()
        if not target.is_relative_to(FRONTEND) or not target.is_file():
            return self._send_json(404, {"error": "not found"})
        content_type = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), content_type)

    def do_POST(self):
        route = POST_ROUTES.get(urlsplit(self.path).path)
        if route is None:
            return self._send_json(404, {"error": "not found"})
        # Requiring JSON stops other websites from posting here through the browser.
        if self.headers.get("Content-Type") != "application/json":
            return self._send_json(415, {"error": "expected application/json"})
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_REQUEST_BYTES:
            return self._send_json(413, {"error": "request too large"})
        try:
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise sim_api.ApiError("expected a JSON object")
            self._send_json(200, route(request))
        except (sim_api.ApiError, json.JSONDecodeError) as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"error": f"server error: {e}"})

    def log_message(self, format, *args):
        pass  # the UI reports errors; per-request logs would just be noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        sys.exit(f"Could not listen on port {args.port} ({e}). Try --port with another number.")
    print(f"Ball simulator running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
