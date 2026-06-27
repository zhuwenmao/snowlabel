"""Tiny CORS-enabled static server for the demo images.

Serves demo_images/ on :8081 so both the Label Studio frontend (<img>) and the
SnowLabel ML backend (requests.get) can fetch task images by http URL, with
Access-Control-Allow-Origin:* so LS brush canvas can read pixels cross-origin.
"""
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo_images"))
PORT = 8081


class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass  # quiet


# Threading server: LS fires many concurrent image fetches (one per task when
# auto-predict runs); a single-threaded server serializes them and times out.
httpd = ThreadingHTTPServer(("", PORT), CORSHandler)
print(f"Serving demo_images/ on http://localhost:{PORT} (threaded)")
httpd.serve_forever()
