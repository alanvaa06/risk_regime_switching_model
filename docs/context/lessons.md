# Lessons

- Read the artifact before trusting a summary; a binary 8/8 gate over a window that excludes an event is a spec bug, not a model failure.
- A plan step that says "pre-existing golden must show no diff" must list which files are schema unions (e.g. alerts.csv) — otherwise the executor stops on an expected change.
- Subagents hand back while waiting on long pytest runs: tell implementers to run only the new fast tests, and have the controller run slow regressions in the background.
