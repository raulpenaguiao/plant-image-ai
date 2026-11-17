"""Tiny stdlib-only server that answers every request with the 'not maintained' page.

Replaces the FastAPI app after an undeploy. No third-party dependencies, so it
keeps working after the virtualenv has been deleted.
"""
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

with open(os.path.join(HERE, "index.html"), "rb") as f:
    PAGE = f.read()


class Handler(BaseHTTPRequestHandler):
    def _respond(self, with_body=True):
        # 410 Gone: the resource is intentionally and permanently unavailable.
        self.send_response(410)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        if with_body:
            self.wfile.write(PAGE)

    def do_HEAD(self):
        self._respond(with_body=False)

    def _any(self):
        # Drain a small request body (e.g. an upload to /predict) before replying.
        try:
            n = min(int(self.headers.get("Content-Length") or 0), 1 << 20)
            if n:
                self.rfile.read(n)
        except (ValueError, OSError):
            pass
        self._respond()

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _any

    def log_message(self, fmt, *args):
        pass  # journald already records the unit's lifecycle; skip per-request noise


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
