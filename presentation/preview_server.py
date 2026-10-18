"""Read-only local preview of the built UI with its existing backend."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError

DIST = Path(__file__).resolve().parents[1] / "Frontend/dist"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/"):
            try:
                response = urlopen("http://127.0.0.1:8000" + self.path[4:], timeout=45)
            except HTTPError as error:
                response = error
            self.send_response(response.status)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            self.end_headers()
            self.wfile.write(response.read())
        else:
            if not (DIST / self.path.split("?")[0].lstrip("/")).is_file():
                self.path = "/index.html"
            super().do_GET()


ThreadingHTTPServer(("127.0.0.1", 5174), Handler).serve_forever()
