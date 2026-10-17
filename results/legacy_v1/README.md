# Legacy results: SSPO v1 code package (superseded protocol)

These are the results of the first SSPO experiment package (October 15-16, 2026), kept for reference. They were
produced with a **different protocol** and are not comparable with the numbers of the standard protocol of this
repository.

| item | v1 package | this repository (standard protocol) |
|---|---|---|
| policy model | Qwen3-4B, LoRA r = 64 | Qwen3.5-9B, LoRA r = 64 (MLP + full attention) |
| context / generation | 4,096 / 1,536 tokens | 262,144 / 16,384 tokens |
| problems | 100 test problems from earlier execution-selection experiments (AtCoder, LeetCode, USACO, Codeforces) | CodeContests, contest-disjoint splits |
| checker | reused caches built with Qwen3.5-35B-A3B helper programs | built with the policy's own base model (Qwen3.5-9B, thinking off) |
| training | 30 steps x 16 problems x 16 rollouts, lr 1e-6 | 100 steps x 32 problems x 16 rollouts, lr 1e-5 |
| evaluation | 16 samples per problem, Monte-Carlo select@k | 32 samples per problem, exact select@k |

Main table (test, best checkpoint by test score, which is optimistic; `step_0030` = final):

| arm | seeds | best select@8 | final select@8 | best pass@8 |
|---|---:|---:|---:|---:|
| sspo (2 turns) | 1 | 19.88 | 15.56 | 27.03 |
| pkpo_pass8 | 2 | 18.45 | 18.03 | 25.85 |
| drgrpo_pass1 | 2 | 18.44 | 16.54 | 24.74 |
| sspo | 3 | 18.21 | 17.34 | 25.16 |
| checker_reward | 1 | 18.12 | 15.68 | 24.35 |
| frozen | 1 | 16.84 | 16.84 | 23.00 |

All paired differences between trained arms were within noise (e.g. sspo vs drgrpo_pass1 select@8 −0.23 points,
95% CI [−2.20, +1.98], 100 problems). The training-free diagnosis (`diag_summary.json`) explains why the setup
could not separate the methods: the selection gap cov@8 − select@8 was only 1.15 points, critical problems
(where the selector loses a covered correct program) were 5%, and only 19 problems had a base pass@8 in (0, 1).
The standard protocol here was designed against exactly these limits (larger model, long generation budget,
own-model checkers on CodeContests, a larger active training pool, exact select@k).

Files: `tables/` (all CSV tables of the v1 report), `report_v1.md` (the v1 auto-generated report),
`runs/<arm>_seed<k>/{summary.json,manifest.json,metrics.jsonl}`, `diag_summary.json`, `probe_summary.json`.
