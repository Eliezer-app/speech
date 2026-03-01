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

import argparse
import json
import os
import queue
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
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
        self.muted = False
        self._silence = None
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
        """Send audio bytes to all connected clients. Drop dead ones.
        When muted, sends silence instead."""
        if self.muted:
            if self._silence is None or len(self._silence) != len(data):
                self._silence = b'\x00' * len(data)
            data = self._silence
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

    def wait_ready(self, timeout=30):
        """Block until hotword process prints 'ready' on stderr."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                err = self.proc.stderr.read().decode()
                raise RuntimeError(f"Hotword process died during startup:\n{err}")
            ready, _, _ = select.select([self.proc.stderr], [], [], 0.5)
            if ready:
                line = self.proc.stderr.readline().decode().rstrip()
                if line:
                    print(f"  [hotword] {line}", file=sys.stderr, flush=True)
                if "ready" in line.lower():
                    return
        raise RuntimeError("Hotword process did not become ready in time")

    def drain(self):
        """Discard any buffered detection lines."""
        while True:
            ready, _, _ = select.select([self.proc.stdout], [], [], 0)
            if not ready:
                break
            self.proc.stdout.readline()

    def is_alive(self):
        return self.proc is not None and self.proc.poll() is None

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

    def wait_ready(self, timeout=120):
        """Block until STT process prints 'ready' on stderr."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                err = self.proc.stderr.read().decode()
                raise RuntimeError(f"STT process died during startup:\n{err}")
            ready, _, _ = select.select([self.proc.stderr], [], [], 0.5)
            if ready:
                line = self.proc.stderr.readline().decode().rstrip()
                if line:
                    print(f"  [stt] {line}", file=sys.stderr, flush=True)
                if "ready" in line.lower():
                    return
        raise RuntimeError("STT process did not become ready in time")

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


class TTSProcess:
    """Manages the persistent TTS subprocess.

    Starts once at boot. Send text on stdin, it speaks and prints DONE.
    """

    def __init__(self):
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen(
            [str(_DIR / "tts" / "run")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def speak(self, text):
        """Send text to TTS. Non-blocking — poll for DONE."""
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.write(f"{text}\n".encode())
            self.proc.stdin.flush()

    def poll_done(self):
        """Non-blocking check if TTS finished speaking."""
        if self.proc is None or self.proc.poll() is not None:
            return False
        ready, _, _ = select.select([self.proc.stdout], [], [], 0)
        if ready:
            line = self.proc.stdout.readline().decode().strip()
            return line == "DONE"
        return False

    def drain_stderr(self):
        if self.proc is None:
            return
        ready, _, _ = select.select([self.proc.stderr], [], [], 0)
        while ready:
            line = self.proc.stderr.readline().decode().rstrip()
            if line:
                print(f"  [tts] {line}", file=sys.stderr, flush=True)
            ready, _, _ = select.select([self.proc.stderr], [], [], 0)

    def is_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        self.proc = None


def load_agent_config(path):
    """Load agent config YAML if it exists. Returns dict."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
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


speak_queue = queue.Queue()


class SpeakHandler(BaseHTTPRequestHandler):
    """HTTP handler for /speak endpoint."""

    def do_POST(self):
        if self.path == "/speak":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode()
            try:
                data = json.loads(body)
                text = data.get("text", "")
            except json.JSONDecodeError:
                text = body.strip()
            if text:
                speak_queue.put(text)
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


def start_http_server(port):
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("127.0.0.1", port), SpeakHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def audio_chunks_from_file(path):
    """Yield float32 mono 16kHz chunks from a wav file at real-time pace."""
    import scipy.io.wavfile as wavfile
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype != np.float32:
        data = data.astype(np.float32) / 32768.0
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sr}Hz")
    start = time.monotonic()
    for i in range(0, len(data), CHUNK_SAMPLES):
        chunk = data[i:i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        target = start + (i + CHUNK_SAMPLES) / SAMPLE_RATE
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)
        yield chunk


def main():
    parser = argparse.ArgumentParser(description="Speech assistant orchestrator")
    parser.add_argument("--audio-file", help="WAV file instead of mic")
    parser.add_argument("--config", default=str(_DIR / "agent_config.yaml"),
                        help="Agent config YAML (default: agent_config.yaml)")
    args = parser.parse_args()

    print("=== Speech Assistant ===\n")

    # Preflight checks
    classifier = _DIR / "hotword" / "output" / "classifier.onnx"
    classifier_data = _DIR / "hotword" / "output" / "classifier.onnx.data"
    missing = [f for f in [classifier, classifier_data] if not f.exists()]
    if missing:
        names = ", ".join(f.name for f in missing)
        print(f"ERROR: Missing hotword model files: {names}\n"
              "Train the hotword model (cd hotword && make train) "
              "or copy classifier.onnx + classifier.onnx.data into hotword/output/.",
              file=sys.stderr, flush=True)
        return

    if not args.audio_file:
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
                audio_test, _ = stream.read(CHUNK_SAMPLES * 5)
                if np.all(audio_test == 0):
                    raise RuntimeError(
                        "Microphone returns silence — no audio access "
                        "(running via SSH?). Use --audio-file or run locally.")
        except sd.PortAudioError as e:
            print(f"ERROR: No microphone available: {e}",
                  file=sys.stderr, flush=True)
            return
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr, flush=True)
            return

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

    state = "idle"  # idle | conversing | speaking
    utterances = []  # accumulated transcriptions for current session

    # Agent config
    agent_cfg = load_agent_config(args.config)
    stt_initial_prompt = agent_cfg.get("stt_initial_prompt", "")
    ctx_url = agent_cfg.get("stt_context_url")
    chat_url = agent_cfg.get("chat_message_url")
    chat_payload = agent_cfg.get("chat_message_payload")
    listen_port = agent_cfg.get("listen_port", 8124)
    if ctx_url:
        print(f"Agent context: {ctx_url}")
    if chat_url:
        print(f"Agent chat: {chat_url}")

    # Start STT once (model loads in background)
    stt = STTProcess(SOCKET_PATH)
    stt.start()

    # Start TTS
    tts = TTSProcess()
    tts.start()

    # Start HTTP server for /speak
    http_server = start_http_server(listen_port)
    print(f"Listening on http://127.0.0.1:{listen_port}/speak")

    running = True

    def handle_signal(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    prev_state_before_speak = "idle"

    def shutdown():
        print("\nShutting down...")
        http_server.shutdown()
        tts.stop()
        stt.stop()
        hotword.stop()
        audio.close()

    # Wait for subprocesses to be ready
    try:
        hotword.wait_ready()
        stt.wait_ready()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        shutdown()
        return
    print(f"\nState: {state} — listening for wake word\n")

    def mic_chunks():
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while running:
                chunk, _ = stream.read(CHUNK_SAMPLES)
                yield chunk[:, 0]

    if args.audio_file:
        print(f"Audio source: {args.audio_file}")
        chunks = audio_chunks_from_file(args.audio_file)
    else:
        chunks = mic_chunks()

    try:
        for chunk in chunks:
            if not running:
                break
            audio.broadcast(chunk.tobytes())

            stt.drain_stderr()
            tts.drain_stderr()

            # Check speak queue — agent can push text at any time
            try:
                text = speak_queue.get_nowait()
                print(f"  [speak] {text}")
                if state == "conversing":
                    # Interrupt STT session
                    stt.proc.stdin.write(b'{"cmd":"STOP"}\n')
                    stt.proc.stdin.flush()
                prev_state_before_speak = state
                state = "speaking"
                audio.muted = True
                tts.speak(text)
            except queue.Empty:
                pass

            # TTS finished speaking
            if state == "speaking" and tts.poll_done():
                audio.muted = False
                hotword.drain()
                state = prev_state_before_speak
                if state == "conversing":
                    # Resume STT
                    agent_ctx = fetch_stt_context(ctx_url) if ctx_url else ""
                    parts = [p for p in [stt_initial_prompt, agent_ctx] if p]
                    context = "\n".join(parts)
                    stt.activate(context)
                print(f"  [speak] done, state: {state}")

            if state == "idle":
                if not hotword.is_alive():
                    print("ERROR: hotword process died", file=sys.stderr, flush=True)
                    running = False
                    break
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
                    print("ERROR: STT process died", file=sys.stderr, flush=True)
                    running = False
                    break

        # File ended — keep sending silence so STT's silence timeout fires
        if args.audio_file and state == "conversing":
            print("Audio file ended, waiting for STT...")
            silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            deadline = time.time() + 15
            while state == "conversing" and time.time() < deadline:
                audio.broadcast(silence.tobytes())
                stt.drain_stderr()
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
                else:
                    time.sleep(CHUNK_MS / 1000)
            sd.wait()  # let beep finish before shutdown

    except KeyboardInterrupt:
        pass
    shutdown()


if __name__ == "__main__":
    main()
