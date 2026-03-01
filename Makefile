PYTHON ?= python3

.PHONY: install prepare run record mock-agent test clean

# Clone subprojects and prepare all venvs
install:
	test -d hotword || git clone https://github.com/Eliezer-app/hotword.git
	test -d stt || git clone https://github.com/Eliezer-app/stt.git
	test -d tts || git clone https://github.com/Eliezer-app/tts.git
	$(MAKE) prepare

# Create venv and install deps for all projects
prepare:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	$(MAKE) -C hotword prepare PYTHON=$(PYTHON)
	$(MAKE) -C stt prepare PYTHON=$(PYTHON)
	$(MAKE) -C tts prepare PYTHON=$(PYTHON)

# Run with mic (default)
run:
	.venv/bin/python main.py

# Record test audio
record:
	.venv/bin/python record.py

# Start mock agent standalone
mock-agent:
	.venv/bin/python mock_agent.py

# Full integration test
test: test_audio.wav
	@lsof -ti:8123 | xargs kill 2>/dev/null || true
	.venv/bin/python test_integration.py

test_audio.wav:
	@echo "No test_audio.wav found. Run: make record"
	@exit 1

clean:
	rm -f test_audio_pcm.wav
