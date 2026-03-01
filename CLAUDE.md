## Rules

- Multi-repo project: speech/, hotword/, stt/, tts/ each have their own .git. NEVER run git commands without `cd /absolute/path &&` prefix. Example: `cd /Users/victor/projects/speech/stt && git status`
- Always use `.venv/bin/pip` for installs, never bare `pip`
