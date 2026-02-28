"""Integration test: wav file → hotword → STT → mock agent."""

import json
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

_DIR = Path(__file__).resolve().parent
PYTHON = str(_DIR / ".venv" / "bin" / "python")
PORT = 8123


class MockAgent(BaseHTTPRequestHandler):
    received = []
    context_requests = 0

    def do_GET(self):
        if self.path == "/stt-context":
            MockAgent.context_requests += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Sand Hollow, Southern Utah")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode()
            MockAgent.received.append(body)
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


def run_server(server):
    server.serve_forever()


def main():
    wav = _DIR / "test_audio.wav"
    config = _DIR / "agent_config.test.yaml"
    if not wav.exists():
        print(f"FAIL: {wav} not found. Run: make record")
        sys.exit(1)
    if not config.exists():
        print(f"FAIL: {config} not found")
        sys.exit(1)

    # Start mock agent
    import socket
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("127.0.0.1", PORT), MockAgent)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    t = threading.Thread(target=run_server, args=(server,), daemon=True)
    t.start()

    # Run main.py with wav
    proc = subprocess.run(
        [PYTHON, str(_DIR / "main.py"),
         "--audio-file", str(wav),
         "--config", str(config)],
        capture_output=True, text=True, timeout=120,
    )

    server.shutdown()

    stdout = proc.stdout
    stderr = proc.stderr
    ok = True

    # 1. Hotword detected
    if "Wake word detected" in stdout:
        print("PASS: hotword detected")
    else:
        print("FAIL: hotword not detected")
        ok = False

    # 2. Context fetched
    if MockAgent.context_requests > 0:
        print(f"PASS: context fetched ({MockAgent.context_requests} request(s))")
    else:
        print("FAIL: context not fetched")
        ok = False

    # 3. STT produced transcription
    if ">> " in stdout:
        lines = [l.strip() for l in stdout.splitlines() if l.strip().startswith(">> ")]
        text = " ".join(l[3:] for l in lines)
        print(f"PASS: STT transcribed: \"{text}\"")
    else:
        print("FAIL: no STT transcription")
        ok = False

    # 4. Chat message posted to agent
    if MockAgent.received:
        print(f"PASS: chat posted ({len(MockAgent.received)} message(s))")
        for msg in MockAgent.received:
            print(f"      payload: {msg}")
    else:
        print("FAIL: no chat message posted")
        ok = False

    # 5. Clean shutdown
    if proc.returncode == 0:
        print("PASS: clean exit")
    else:
        print(f"FAIL: exit code {proc.returncode}")
        ok = False

    if not ok:
        print(f"\n--- stdout ---\n{stdout}")
        print(f"\n--- stderr ---\n{stderr}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
