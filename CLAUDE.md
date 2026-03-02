## CRITICAL — READ FIRST

**NEVER `git push` unless the user EXPLICITLY says "push" in that message.** "commit and push" = push. "commit" = do NOT push. This is non-negotiable — multiple deployments depend on controlled pushes. Reverts are hard.

## Rules

- Multi-repo project: speech/, hotword/, stt/, tts/ each have their own .git. NEVER run git commands without `cd /absolute/path &&` prefix. Example: `cd /Users/victor/projects/speech/stt && git status`
- Always use `.venv/bin/pip` for installs, never bare `pip`
