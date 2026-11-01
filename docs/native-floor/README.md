# Native prefill floor: analysis and L1/L2 design (snapshots)

These are snapshots, taken 1 Nov 2026 ~15:05Z, of the hub working copies in `tmp/native-floor-analysis/`. The working copies remain the files the agents edit. Main gets a fresh snapshot at each milestone.

- `20261101-analysis.md`: the per-chunk command and round-trip anatomy of the native split prefill, and why it is far from the GPU floor.
- `20261101-design-l1-l2.md`: steps A (one command per dense QSA layer), B (grouped GDN/QSA commands) and C0/C1 (two chunks in flight across PP), with the proofs, risk register and kill criteria.

**Amendments from the independent review** (`docs/reviews/20261101-prefill-progress/`), not yet folded into the text:
- **C0/C1, at design :113–118:** a stage-local command's apply must not wait on the other stage's ranks (D1).
- **R10:** after Step A, a per-stage service checkpoint against 256/183 ms per 1,024 rows (M2).
- **Step A cancel proof:** it must cover a cancel arriving between the two TP peers' acceptance.
