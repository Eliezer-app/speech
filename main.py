"""
Speech assistant orchestrator.

Captures mic audio, publishes to a unix socket for consumers.
Runs hotword detection as a subprocess. On detection, transitions
to conversation mode (STT placeholder).

Architecture:
  main.py (audio server + state machine)
    └─ unix socket: raw float32 PCM @ 16kHz mono
         ├─ hotword/detect.py (subprocess, always connected)
         └─ stt (future)

Usage: python main.py
"""

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

_DIR = Path(__file__).resolve().parent
SOCKET_PATH = "/tmp/speech-audio.sock"
SAMPLE_RATE = 16000
CHUNK_MS = 80
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)


class AudioServer:
    """Captures mic audio and fans out to connected unix socket clients."""

    def __init__(self, sock_path):
        self.sock_path = sock_path
        self.clients = []
        self.lock = threading.Lock()
        self._cleanup_socket()
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(sock_path)
        self.server.listen(4)
        self.server.settimeout(0.5)

    def _cleanup_socket(self):
        try:
            os.unlink(self.sock_path)
        except FileNotFoundError:
            pass

    def accept_loop(self):
        """Run in a thread — accepts new client connections."""
        while True:
            try:
                conn, _ = self.server.accept()
                with self.lock:
                    self.clients.append(conn)
            except socket.timeout:
                continue
            except OSError:
                break

    def broadcast(self, data):
        """Send audio bytes to all connected clients. Drop dead ones."""
        with self.lock:
            alive = []
            for c in self.clients:
                try:
                    c.sendall(data)
                    alive.append(c)
                except (BrokenPipeError, OSError):
                    try:
                        c.close()
                    except OSError:
                        pass
            self.clients = alive

    def close(self):
        with self.lock:
            for c in self.clients:
                try:
                    c.close()
                except OSError:
                    pass
            self.clients.clear()
        self.server.close()
        self._cleanup_socket()


class HotwordProcess:
    """Manages the hotword detector subprocess."""

    def __init__(self, sock_path):
        self.sock_path = sock_path
        self.proc = None

    def start(self):
        hotword_dir = _DIR / "hotword"
        detect_py = hotword_dir / "detect.py"
        hotword_python = hotword_dir / ".venv" / "bin" / "python"
        self.proc = subprocess.Popen(
            [str(hotword_python), "-u", str(detect_py), "--audio-source", self.sock_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def poll_detection(self):
        """Non-blocking check for detection output. Returns confidence or None."""
        if self.proc is None or self.proc.poll() is not None:
            return None
        import select
        ready, _, _ = select.select([self.proc.stdout], [], [], 0)
        if ready:
            line = self.proc.stdout.readline().decode().strip()
            if line.startswith("DETECTED"):
                try:
                    conf = float(line.split(":")[-1].strip(" )"))
                except (ValueError, IndexError):
                    conf = 1.0
                return conf
        return None

    def drain(self):
        """Discard any buffered detection lines."""
        import select
        while True:
            ready, _, _ = select.select([self.proc.stdout], [], [], 0)
            if not ready:
                break
            self.proc.stdout.readline()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        self.proc = None


class STTProcess:
    """Manages the mock STT subprocess."""

    def __init__(self, sock_path):
        self.sock_path = sock_path
        self.proc = None

    def start(self):
        listen_py = _DIR / "stt" / "listen.py"
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(listen_py), "--audio-source", self.sock_path],
            stdout=None,
            stderr=None,
        )

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        self.proc = None


def main():
    print("=== Speech Assistant ===\n")

    # Start audio server
    audio = AudioServer(SOCKET_PATH)
    accept_thread = threading.Thread(target=audio.accept_loop, daemon=True)
    accept_thread.start()
    print(f"Audio server: {SOCKET_PATH}")

    # Give the socket a moment, then start hotword detector
    time.sleep(0.2)
    hotword = HotwordProcess(SOCKET_PATH)
    hotword.start()
    print("Hotword detector: started")

    state = "idle"  # idle | conversing
    stt = None

    running = True

    def handle_signal(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    def shutdown():
        print("\nShutting down...")
        if stt:
            stt.stop()
        hotword.stop()
        audio.close()

    # Quick health check
    time.sleep(1)
    rc = hotword.proc.poll()
    if rc is not None:
        print(f"ERROR: hotword subprocess exited with code {rc}")
        # drain stdout/stderr
        out = hotword.proc.stdout.read()
        if out:
            print(f"  stdout: {out}")
        err = hotword.proc.stderr.read()
        if err:
            print(f"  stderr: {err}")
        shutdown()
    print(f"\nState: {state} — listening for wake word\n")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while running:
                chunk, _ = stream.read(CHUNK_SAMPLES)
                audio.broadcast(chunk[:, 0].tobytes())

                if state == "idle":
                    conf = hotword.poll_detection()
                    if conf is not None:
                        print(f"\n*** Wake word detected (conf={conf:.3f}) ***")
                        state = "conversing"
                        stt = STTProcess(SOCKET_PATH)
                        stt.start()

                elif state == "conversing":
                    if not stt.is_running():
                        stt = None
                        state = "idle"
                        print(f"\nState: {state} — listening for wake word\n")

    except KeyboardInterrupt:
        pass
    shutdown()


if __name__ == "__main__":
    main()
