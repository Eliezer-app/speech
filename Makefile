PYTHON = .venv/bin/python

.PHONY: run record mock-agent test clean

# Run with mic (default)
run:
	$(PYTHON) main.py

# Record test audio
record:
	$(PYTHON) record.py

# Start mock agent standalone
mock-agent:
	$(PYTHON) mock_agent.py

# Full integration test
test: test_audio.wav
	@lsof -ti:8123 | xargs kill 2>/dev/null || true
	$(PYTHON) test_integration.py

test_audio.wav:
	@echo "No test_audio.wav found. Run: make record"
	@exit 1

clean:
	rm -f test_audio_pcm.wav
