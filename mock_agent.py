"""Mock agent for integration testing.

GET /stt-context  → returns plain text context
POST /chat        → prints received payload, returns 200
"""

import argparse
import json
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler


class Handler(BaseHTTPRequestHandler):
    context = ""

    def do_GET(self):
        if self.path == "/stt-context":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(self.context.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode()
            print(f"\n=== RECEIVED CHAT ===\n{body}\n=====================")
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # suppress request logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8123)
    parser.add_argument("-c", "--context", default="Sand Hollow, Southern Utah")
    args = parser.parse_args()

    Handler.context = args.context
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Mock agent on http://127.0.0.1:{args.port}")
    print(f"  GET /stt-context → \"{args.context}\"")
    print(f"  POST /chat → prints payload")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
