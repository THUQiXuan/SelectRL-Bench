# Reference results

## Standard protocol (`selectrl-standard-v1`)

| run | status | val select@8 | test select@8 | test pass@1 | test pass@8 |
|---|---|---:|---:|---:|---:|
| `frozen_base` (Qwen3.5-9B, no training) | see `baselines/frozen_base/` | – | – | – | – |
| `sspo_seed0` | to be produced on 8 x H200 (`bash scripts/run_baselines.sh`) | – | – | – | – |
| `grpo_seed0` | to be produced | – | – | – | – |
| `passk_seed0` | to be produced | – | – | – | – |

Numbers are in percent; select@8 is the exact expectation over all 8-subsets of 32 samples per problem.
`python -m selectrl.eval.report --runs "runs/*" "results/baselines/*"` regenerates the table with paired,
contest-clustered bootstrap intervals.

## Legacy

`legacy_v1/` holds the results of the first SSPO code package (Qwen3-4B, 4K context, different problems); they
are not comparable with the standard protocol. See `legacy_v1/README.md`.
