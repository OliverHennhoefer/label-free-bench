# Gardening

Run a gardening session on the cadence in `loop.toml` (default every 10 iterations), when `check-memory` fails, or when the index nears its limit. Use a fresh session that did not run the experiments.

1. Run `python -m benchmarks check-memory` and fix every failure.
2. Merge duplicate or overlapping claims into one. Mark the old ones `superseded by C-NNN` and `git mv` them to `archive/`.
3. Promote a pattern seen in three or more claims or negative rows to a new claim that cites them.
4. Compress each closed experiment: its claim or negative row must stand alone, then move the experiment folder to `archive/`.
5. Turn finished literature notes into claims or queue items where they change a decision.
6. Rewrite the index so it is current. Do not change any status without a new ledger run.
7. Commit with a message listing every merge, retirement and promotion.
