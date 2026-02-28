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

import json
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib import request, error

import numpy as np
import sounddevice as sd
import yaml

_DIR = Path(__file__).resolve().parent
SOCKET_PATH = "/tmp/speech-audio.sock"
SAMPLE_RATE = 16000
CHUNK_MS = 80
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
PLAYBACK_RATE = 48000  # Mac speakers default


def make_tone(freq, duration, volume=0.3):
    t = np.linspace(0, duration, int(PLAYBACK_RATE * duration), dtype=np.float32)
    fade = int(PLAYBACK_RATE * 0.01)
    tone = np.sin(2 * np.pi * freq * t) * volume
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    return tone


def play_beep():
    """Single rising beep — hotword detected."""
    sd.play(make_tone(880, 0.15), samplerate=PLAYBACK_RATE)


def play_idle_beep():
    """Two-tone tee-taa — back to idle."""
    gap = np.zeros(int(PLAYBACK_RATE * 0.05), dtype=np.float32)
    sound = np.concatenate([make_tone(660, 0.1), gap, make_tone(440, 0.12)])
    sd.play(sound, samplerate=PLAYBACK_RATE)


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
        self.proc = subprocess.Popen(
            [str(_DIR / "hotword" / "run"), "--audio-source", self.sock_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def poll_detection(self):
        """Non-blocking check for detection output. Returns confidence or None."""
        if self.proc is None or self.proc.poll() is not None:
            return None
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
    """Manages the persistent STT subprocess.

    Starts once at boot (model loads once). Controlled via stdin:
      send "START\\n" → STT begins listening and printing transcriptions
      STT prints "END\\n" on stdout → silence timeout, back to idle
      send "STOP\\n"  → force stop current session
    """

    def __init__(self, sock_path):
        self.sock_path = sock_path
        self.proc = None

    def start(self):
        """Launch the STT process (loads models, then waits for START)."""
        self.proc = subprocess.Popen(
            [str(_DIR / "stt" / "run"), "--audio-source", self.sock_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def activate(self, context=""):
        """Send START to begin a listening session, with optional context."""
        if self.proc and self.proc.poll() is None:
            msg = json.dumps({"cmd": "START", "context": context})
            self.proc.stdin.write(f"{msg}\n".encode())
            self.proc.stdin.flush()

    def poll_output(self):
        """Non-blocking read of stdout. Returns line or None."""
        if self.proc is None or self.proc.poll() is not None:
            return None
        ready, _, _ = select.select([self.proc.stdout], [], [], 0)
        if ready:
            line = self.proc.stdout.readline().decode().strip()
            return line if line else None
        return None

    def drain_stderr(self):
        """Non-blocking drain of stderr, print to our stderr."""
        if self.proc is None:
            return
        ready, _, _ = select.select([self.proc.stderr], [], [], 0)
        while ready:
            line = self.proc.stderr.readline().decode().rstrip()
            if line:
                print(f"  [stt] {line}", file=sys.stderr, flush=True)
            ready, _, _ = select.select([self.proc.stderr], [], [], 0)

    def is_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        self.proc = None


def load_agent_config():
    """Load agent_config.yaml if it exists. Returns dict."""
    path = _DIR / "agent_config.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def fetch_stt_context(url):
    """GET stt context from agent. Returns context string or ""."""
    try:
        req = request.Request(url)
        with request.urlopen(req, timeout=3) as resp:
            return resp.read().decode().strip()
    except Exception as e:
        print(f"  [agent] context fetch failed: {e}", file=sys.stderr, flush=True)
        return ""


def post_chat_message(url, text, payload_template=None):
    """POST accumulated transcript to agent."""
    try:
        if payload_template:
            body = payload_template.replace("%text%", text.replace('"', '\\"')).encode()
        else:
            body = json.dumps({"text": text}).encode()
        req = request.Request(url, data=body,
                              headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=10) as resp:
            resp.read()
    except Exception as e:
        print(f"  [agent] chat post failed: {e}", file=sys.stderr, flush=True)


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
    utterances = []  # accumulated transcriptions for current session

    # Agent config
    agent_cfg = load_agent_config()
    stt_initial_prompt = agent_cfg.get("stt_initial_prompt", "")
    ctx_url = agent_cfg.get("stt_context_url")
    chat_url = agent_cfg.get("chat_message_url")
    chat_payload = agent_cfg.get("chat_message_payload")
    if ctx_url:
        print(f"Agent context: {ctx_url}")
    if chat_url:
        print(f"Agent chat: {chat_url}")

    # Start STT once (model loads in background)
    stt = STTProcess(SOCKET_PATH)
    stt.start()
    print("STT: starting (loading models)...")

    running = True

    def handle_signal(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    def shutdown():
        print("\nShutting down...")
        stt.stop()
        hotword.stop()
        audio.close()

    # Quick health check
    time.sleep(1)
    rc = hotword.proc.poll()
    if rc is not None:
        print(f"ERROR: hotword subprocess exited with code {rc}")
        out = hotword.proc.stdout.read()
        if out:
            print(f"  stdout: {out}")
        err = hotword.proc.stderr.read()
        if err:
            print(f"  stderr: {err}")
        shutdown()
        return
    print(f"\nState: {state} — listening for wake word\n")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while running:
                chunk, _ = stream.read(CHUNK_SAMPLES)
                audio.broadcast(chunk[:, 0].tobytes())

                stt.drain_stderr()

                if state == "idle":
                    conf = hotword.poll_detection()
                    if conf is not None:
                        play_beep()
                        print(f"\n*** Wake word detected (conf={conf:.3f}) ***")
                        agent_ctx = fetch_stt_context(ctx_url) if ctx_url else ""
                        parts = [p for p in [stt_initial_prompt, agent_ctx] if p]
                        context = "\n".join(parts)
                        if context:
                            print(f"  [context] {context}")
                        state = "conversing"
                        utterances = []
                        stt.activate(context)
                        hotword.drain()

                elif state == "conversing":
                    line = stt.poll_output()
                    if line == "END":
                        if utterances and chat_url:
                            text = " ".join(utterances)
                            print(f"  [sending] {text}")
                            post_chat_message(chat_url, text, chat_payload)
                        state = "idle"
                        play_idle_beep()
                        print(f"\nState: {state} — listening for wake word\n")
                    elif line:
                        print(f"  >> {line}")
                        utterances.append(line)

                    if not stt.is_alive():
                        print("ERROR: STT process died", file=sys.stderr)
                        state = "idle"

    except KeyboardInterrupt:
        pass
    shutdown()


if __name__ == "__main__":
    main()
