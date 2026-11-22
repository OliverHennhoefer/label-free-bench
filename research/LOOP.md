# Research loop

Each session plays exactly one role. `python -m benchmarks next` reads the repository and names the role, the experiment and the instructions file. Nothing else carries state between sessions: the repository, `HANDOFF.md` and `loop.jsonl` do.

## One session

1. Work on the `lab` branch: `git fetch origin main lab`, check out `lab` (create it from `origin/main` if it does not exist), and merge `origin/main` so held-out ledger rows from CI arrive.
2. `python -m benchmarks next`. If the role is `halt`, stop and report what it is waiting on.
3. Follow `research/roles/<role>.md`. How you do the work is up to you.
4. `python -m benchmarks log --role <role> [--slug <slug>] --summary "..." [--friction "..."]`. Friction is a rule or step that got in the way, for the observer.
5. Overwrite `HANDOFF.md`, unless you are the verifier or observer. Commit and push to `lab`. A human merges `lab` into `main`, which a held-out look needs first.

## Order

`next` picks the first that applies: halt on a flag at the top of `HANDOFF.md` or `Halt: yes` in the latest observer report; setup once, as iteration 0; a final observer report and then halt at `max_iterations`; gardener when `check-memory` fails; observer on its cadence; gardener every `gardener_every` iterations; verifier for a graded experiment; runner for a committed prediction; otherwise proposer. Cadence lives in `loop.toml`.

## Guardrails

- **Fresh session per role.** `log` refuses a session that already played another role on the same experiment, and `check-memory` fails a `result.md` without a separate verifier session.
- **No held-out looks from the loop.** Roles grade only on development. A held-out look is a `heldout-request.md` that a human dispatches through CI.

Everything else, including how many ideas to keep open and how to search, is the roles' call.

## Drivers

- **Scheduled cloud trigger** (recommended). One wake runs one session with the prompt in `scripts/wake-prompt.md`. Each wake is a fresh session by construction and leaves a commit.
- **Local.** `scripts/lab-loop.sh N` runs N sessions with `claude -p`, each with a new `LAB_SESSION`.
