## CRITICAL — READ FIRST

**NEVER `git push` unless the user EXPLICITLY says "push" in that message.** "commit and push" = push. "commit" = do NOT push. This is non-negotiable — multiple deployments depend on controlled pushes. Reverts are hard.

## Rules

- Multi-repo project: speech/, hotword/, stt/, tts/ each have their own .git. Always use `git -C /absolute/path` for git commands.
- Always use `.venv/bin/pip` for installs, never bare `pip`
- **Before committing:** run `git status` on ALL repos. Review every outstanding change (staged, unstaged, untracked). Do NOT commit if anything is unclear — ask first. Every change must be accounted for in the commit.
