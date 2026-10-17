# Review of the first SSPO code package (v1) and what changed

The first package (`sspo_v1`, October 2026) was written without GPU access and handed to a collaborator to run.
Running it on an 8-GPU node with vLLM 0.29 exposed problems; this file lists what a code review found and how
this repository addresses each item.

## Findings

| # | area | v1 behaviour | consequence | here |
|---|---|---|---|---|
| 1 | environment | `requirements-gpu.txt` pins vLLM 0.6.6 / torch 2.5.1 / transformers 4.48 / verl 0.2; the reported runs were actually produced in a vLLM 0.29.0 / torch 2.13 / transformers 5.12.1 container | the lock file was never the tested environment; vLLM 0.6.6 cannot run Qwen3.5 | `scripts/setup_env.sh` installs only a few extras on top of an existing vLLM 0.29 environment (or builds one), pins what is installed via constraints, and prints the verified versions |
| 2 | multi-GPU | the README's per-GPU presets (FSDP full fine-tuning on 4-16 GPUs, TP=2, timing table) were not implemented; the trainer was a single process (vLLM on GPU 0, HF + LoRA on GPU 1) | extra GPUs idle; table not meaningful | one vLLM server per GPU + one torchrun rank per GPU (token-balanced shards, gradient all-reduce); presets derived from measured memory |
| 3 | packaging | `results/` shipped with `done` markers of every stage | `run_all.sh` on a new machine would skip diagnosis and all training arms | runs live in `runs/`, which is never shipped; reference numbers live in `results/` without markers |
| 4 | checker stage | the authors' runs did not execute `ec_gen`: they linked checker caches from earlier experiments (built with a 35B helper model on other problem sets) | the checker-building path (`build_checkers.sh`, offline `LLM()`) had never run | helper generation goes through the same vLLM servers as training; `scripts/build_data.sh` builds the training pool and the whole checker cache in one resumable command |
| 5 | checker model | `ec_gen` defaulted to `Qwen/Qwen3-8B` regardless of the policy model chosen by the preset | helpers from a model that was not downloaded and not the policy | helpers are written by the policy's base model (Qwen3.5-9B, thinking off) |
| 6 | vLLM start | offline `LLM(gpu_memory_utilization=0.85)` inside the generation script | fails with "Engine core initialization failed" as soon as anything else holds more than ~15% of the GPU (the root cause is printed above that line) | servers are started by a supervisor that logs each server's output, checks readiness, prints the log tail on failure, and uses a preset memory fraction (0.85 only when nothing else runs on the GPU) |
| 7 | sandbox threads | no OpenMP/BLAS thread limits (the risk notes claimed otherwise) | on a 384-core host `import numpy` in a 2 GiB sandbox stalls or fails | the pool exports `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/RAYON_NUM_THREADS=1` before the worker starts, the worker re-asserts them and preloads numpy before forking |
| 8 | sandbox limits | wall-clock timeouts, 12 workers max, silent output truncation, lost output of threaded programs, a stuck worker could block forever | flaky verdicts under load; slow labeling | CPU-time verdicts (wall guard 3T+2 s), up to 256 workers, output over the cap is an error, non-daemon threads are joined, per-job deadlines with worker replacement |
| 9 | evaluation | Monte-Carlo select@k seeded with Python's randomized `hash()`, and reported as "exact" for impure clusters | irreproducible numbers | exact select@k by a dynamic program for every cluster structure; fixed evaluation seeds |
| 10 | data splits | dev sampled per problem (contests leak across splits), valid and test merged, only the first rows of the stream read | leakage and a biased problem sample | contest-disjoint splits over the whole dataset, documented hold-out rule, filter statistics |
| 11 | budget | 4,096-token context and 1,536 generated tokens | many truncated programs | 262,144-token context window, 16,384 generated tokens |
| 12 | statistical power | 19 active training problems, selection gap 1.15 points, 5% critical problems, lr 1e-6 for 30 steps | trained arms indistinguishable from each other (see `results/legacy_v1/`) | active pool from 4,000 candidates, own-model checkers, lr 1e-5 for 100 steps, 32 eval samples per problem |
| 13 | LoRA serving | (new risk found while porting) vLLM 0.29 accepts a PEFT adapter saved from the text-only `Qwen3_5ForCausalLM` but applies nothing, because the served model nests the text tower under `model.language_model` | training would silently sample from the base model forever | adapters are exported with the served module names (verified: vLLM vs HF log-prob difference with the adapter equals the base-model difference); the trainer aborts at step 2 if the inference/training log-prob gap exceeds 0.08 |
| 14 | proxies | (new) cluster nodes with `http_proxy` set route requests to 127.0.0.1 through the proxy | local servers unreachable | every local request bypasses proxies (`no_proxy`, proxy-free openers) |
| 15 | credentials | a personal access token was hard-coded as a default | not appropriate for a public repository | no credentials in the repository; `HF_TOKEN` is read from the environment when present (the model is public) |

## Answers to the questions raised when running v1 (summary)

1. *Which vLLM was validated?* v1's lock file said 0.6.6, but the reported runs used a vLLM 0.29.0 container
   (torch 2.13.0+cu130, transformers 5.12.1, peft 0.19.1, fla 0.5.2); the checker-building engine was never run.
   This repository is validated on vLLM 0.29.0 with exactly those versions (`requirements/extra.txt`).
2. *Single-GPU checker building?* v1 started one offline engine per split, serially (the 4/4/2 helper programs
   were the `n` of one request each), with 85% of the GPU. Here helper generation uses the per-GPU servers; on one
   GPU nothing else may run on that GPU during this stage.
3. *BLAS threads = 1 in the sandbox?* Correct and harmless (sandboxed programs are single-threaded by design);
   v1 lacked it. It is now built in.
4. *A verified minimal real configuration?* `bash scripts/smoke.sh` (configs/protocol/smoke.yaml) — validated end
   to end on 2 x 48 GB GPUs (split placement).
5. *Which data was the bundled cache?* Not CodeContests: problem pools and checker caches of earlier experiments.
   Real CodeContests needs its own checker cache; `scripts/build_data.sh` builds it (validated on vLLM 0.29.0).
