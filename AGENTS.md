# HBC Lab Agent Instructions

## Subagent Delegation

- When the runtime provides subagent/model delegation, prefer `gpt5.6-luna-max` for repetitive, well-scoped work such as code-review passes, static configuration comparisons, log monitoring, training-progress summaries, and checklist verification.
- Give the subagent a narrow task, explicit files or log directories, and a concrete expected output. Do not delegate ambiguous architecture decisions or destructive repository operations.
- Treat subagent output as supporting evidence. The primary agent remains responsible for checking the relevant code, resolving contradictions, running validation, and reporting the final conclusion.
- If subagent delegation is unavailable, continue the task directly instead of blocking or pretending that a subagent was called.

## Repository Priorities

- Preserve deployability: actor observations must be obtainable on the real robot; simulator-only signals belong in rewards, diagnostics, or the critic.
- Preserve task success before tightening motion-quality regularization.
- Keep the low-level HBC interface and high-level task interface frame-consistent.
- Do not commit training logs, exported checkpoints, caches, or generated visualization artifacts unless explicitly requested.
