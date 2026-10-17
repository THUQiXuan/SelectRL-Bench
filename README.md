# SelectRL-Bench

**RL algorithm discovery for deployment-time execution selection on code.**

A locked, reproducible reinforcement-learning harness in which an *algorithm* is the only free variable.
`Qwen/Qwen3.5-9B` (thinking off, 262,144-token context) is fine-tuned with LoRA on CodeContests problems; the
benchmark measures how often the **deployed system** — sample 8 programs, filter them with a frozen execution
checker, vote by output signature, submit one — solves hidden-test problems (**select@8**). An algorithm is one
Python file implementing `compute_advantages` (and optionally `policy_loss`) against a small contract; everything
else (model, data, sampling, optimizer, compute budget, checker, selector, evaluation) is fixed and hash-locked.

The repository ships three reference algorithms:

| name | idea |
|---|---|
| `grpo` | standard GRPO on the hidden-test reward (group mean/std normalization) |
| `passk` | pass@k training with the PKPO unbiased closed form and a leave-one-out baseline |
| `sspo` | **SSPO**, selection-aware set policy optimization: exact leave-one-out credit for the deployed selector's success (J⁻ with known-correct-signature penalty) plus a coverage term, computed by a dynamic program |

Chinese run guide: [README_zh.md](README_zh.md). Task statement for agents: [AGENT_TASK.md](AGENT_TASK.md).
Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Changes from the first SSPO code package and the answers to
the questions raised when running it: [docs/V1_REVIEW.md](docs/V1_REVIEW.md).

## Quick start (one node, 1-8 GPUs with >= 80 GB; H200 recommended)

```bash
bash scripts/setup_env.sh            # inside a vLLM 0.29 environment (or: --new-venv)
bash scripts/download_model.sh       # Qwen/Qwen3.5-9B @ pinned revision -> models/ (19 GB)
bash scripts/build_data.sh           # once: active training pool + checker cache (8 x H200: ~30-60 min)
bash scripts/smoke.sh                # ~20-30 min: 2 real training steps + resume + tiny eval -> "SMOKE PASS"
bash scripts/run_baselines.sh        # frozen base + sspo, grpo, passk (seed 0), with val/test evaluation
bash scripts/collect_results.sh      # results_bundle.tar.gz (small files only)
```

Every script skips finished work; re-running the same command after an interruption resumes (training resumes
from the last finished step, evaluation from the last finished problem).

## Protocol (configs/protocol/standard.yaml, locked)

| item | value |
|---|---|
| model | `Qwen/Qwen3.5-9B` @ `c2022362`, text tower only, bf16, `max_model_len` 262,144, thinking off |
| sampling | temperature 1.0, top-p 1.0, no generation cap (up to the 262,144-token context), identical for training rollouts and evaluation |
| training | 100 steps x 32 problems x 16 rollouts, 2 PPO mini-batches/step, LoRA r=64 / alpha=128 on MLP + full-attention projections, AdamW lr 1e-5, grad-norm clip 1.0 |
| data | CodeContests; contest-disjoint splits: train = active pool (base solves 1-7 of 8 samples), val = CodeContests valid + held-out contests, test (hidden) = CodeContests test + held-out contests |
| checker | per problem: 4 input generators, 4 brute-force solvers, 2 validators written by the base model (thinking off); Chk = public examples pass ∧ stress inputs finish in 6 s CPU ∧ agreement with confirmed brute-force outputs; signature = hash of normalized outputs on all check inputs |
| metric | exact select@8 over all 8-subsets of 32 samples per problem (also pass@k unbiased, select@k for k in 1..16) |

## Hardware

One vLLM server per GPU serves rollouts; one training rank per GPU trains LoRA with a token-balanced shard.

| GPUs | placement | notes |
|---|---|---|
| >= 120 GB (H200) | colocate: server (40% of memory) + trainer on every GPU | default target; 1-8 GPUs |
| 80 GB (H100 / A100-80G) | colocate with 42% for the server, smaller micro-batches | 1-8 GPUs |
| 48 GB | split: first half of the GPUs serve, second half train | >= 2 GPUs; used for validation runs |

Estimated wall time of one standard training run on 8 x H200: 8-10 hours (generation dominates). The bundled
`SELECTRL_*` environment variables override any preset value (see `selectrl/config.py`).

## Using it as a benchmark

```bash
bash scripts/make_agent_workspace.sh ../agent_ws        # copy without hidden test data, all but algorithms/ read-only
# ... the agent edits ../agent_ws/algorithms/candidate/ following AGENT_TASK.md ...
python -m selectrl.bench.workspace collect --workspace ../agent_ws --name my_algo
bash scripts/score.sh submissions/my_algo               # integrity + guard + train + eval -> runs/<name>/score.json
```

Anti-gaming measures: sha256 lock of the harness, protocol, data and checker cache (`bench/LOCK.json`,
`cache/checkers/MANIFEST.json`); an AST guard on algorithm code (no I/O, processes, network, reflection, autograd
graph access); algorithms only ever receive training-split signals; evaluation runs in a separate process that
never imports algorithm code; fixed evaluation seeds and an exact (sampling-free) select@k; the run manifest
records the protocol hash and the algorithm's sha256, and scoring rejects mismatches.

## Results

See [results/README.md](results/README.md) for the reference numbers and their status.

## Layout

```
selectrl/            locked harness (algo_api, algolib, sandbox, data, checker, rollout, train, eval, bench)
reference_algorithms/ grpo, passk, sspo
algorithms/candidate/ the file a benchmark agent edits (starts as GRPO)
configs/protocol/     standard.yaml (benchmark), smoke.yaml (installation check)
scripts/              bash entry points
data/, cache/         problems, splits, checker cache (released as an asset)
results/              reference results
```

## License and data

Code: Apache-2.0. CodeContests (DeepMind) is CC BY 4.0; problem statements and tests in the released assets are
derived from it. Qwen3.5 weights are downloaded from Hugging Face under their own license.
