# SelectRL-Bench 中文运行说明

这份说明写给在 GPU 机器上跑实验的人：照着做即可，不需要读论文。目标机器：单机 1–8 张 ≥80 GB 显卡（推荐 8×H200 140 GB），已有 vLLM 0.29 环境（或允许新建虚拟环境），普通用户权限即可。

## 0. 这次和第一版代码包的区别（先看）

- 模型统一为 `Qwen/Qwen3.5-9B`（固定版本 `c2022362`），关闭思考，vLLM 上下文窗口用模型默认的 262,144 token；生成不设单独上限（提示 + 回答最多用满 262,144 token）。
- 推理侧每张卡起一个 vLLM OpenAI 服务（同一个代码路径负责训练采样、建检查器和评测），训练侧每张卡一个进程，LoRA 训练，每步把新 adapter 热加载进 vLLM。
- CodeContests 的切分（训练候选 9967 题 / 验证 266 题 / 隐藏测试 422 题，比赛互不重叠）随包附带；训练池筛选和检查器缓存用一条命令在你的机器上建（第 3 步），走的是和训练相同的 vLLM 服务路径，不再是旧版的离线引擎。
- 判题沙箱已在 worker 启动前把 OpenMP/BLAS 线程锁成 1，并按 CPU 时间判超时；384 核机器上不需要再手动设置。
- 所有命令都能续跑：中断后重新执行同一条命令即可。

## 1. 环境（约 5–15 分钟）

已有 vLLM 0.29 环境（例如官方镜像 `vllm/vllm-openai:v0.29.0`）时，在该环境里执行：

```bash
bash scripts/setup_env.sh
```

它只会补装 `peft==0.19.1`、`flash-linear-attention==0.5.2` 等少量包，并用约束文件锁住已装的 torch / vLLM / transformers 版本，不会升级或降级它们；若当前 transformers 不支持 Qwen3.5，会装 5.12.1（已与 vLLM 0.29.0 一起验证）。没有现成环境时：

```bash
bash scripts/setup_env.sh --new-venv
```

最后会打印各包版本、GPU 列表，以及沙箱能否用 `unshare -rn` 断网（不能的话只做 Python 层面的网络屏蔽，不影响结果）。

## 2. 模型（约 19 GB）

```bash
bash scripts/download_model.sh
```

下载到 `models/Qwen3.5-9B`。模型是公开的，`HF_TOKEN` 可设可不设（只影响匿名限速）；国内网络可先 `export HF_ENDPOINT=https://hf-mirror.com`。

## 3. 训练池与检查器缓存（一次，8×H200 约 30–60 分钟）

```bash
bash scripts/build_data.sh
```

依次：每卡起一个 vLLM 服务（推理专用，占 85% 显存）→ 从训练候选里按哈希取 4000 题、每题基座采 8 条，保留做对 1–7 条的题作为训练池 `data/problems/train.jsonl.gz` → 为训练/验证/测试题生成检查器辅助程序（每题 4 个输入生成器、4 个暴力解、2 个校验器，关闭思考）→ CPU 上建检查用例 `cache/checkers/` 并写清单。每一步都可续跑。

## 4. 冒烟测试（约 20–30 分钟，先跑这个）

```bash
bash scripts/smoke.sh
```

真实模型、真实沙箱：GRPO 训练 2 步（第 1 步后故意退出、自动续跑第 2 步），再在 8 道验证题上评测。最后打印 `SMOKE PASS`。失败时看 `runs/smoke_grpo/logs/` 下的 `launcher.log`、`vllm_gpu*.log`、`train.log`。

## 5. 正式实验

一条命令跑完全部基线（冻结基座评测 + SSPO / GRPO / Pass@k 各训练 100 步并在 val、test 上评测），已完成的阶段自动跳过：

```bash
bash scripts/run_baselines.sh
```

8×H200 上每个算法训练约 8–10 小时，评测约 1 小时；全部约 1.5–2 天。也可以分开跑：

```bash
bash scripts/eval.sh base --name frozen_base
bash scripts/train.sh reference_algorithms/sspo --seed 0
bash scripts/train.sh reference_algorithms/grpo --seed 0
bash scripts/train.sh reference_algorithms/passk --seed 0
```

有余力时加种子：`bash scripts/run_baselines.sh --seeds "0 1"`。只想用部分卡：加 `--gpus 0,1,2,3`。

## 6. 输出在哪里

每个运行在 `runs/<名字>/`：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 协议名与哈希、算法名与 sha256、种子、卡数、开始时间 |
| `metrics.jsonl` | 每步一行：正确率 `g_rate`、检查器通过率 `chk_rate`、保留序列数、优势统计、损失、梯度范数、vLLM 与训练端 logprob 差 `mismatch_abs_lp`、各阶段耗时 |
| `eval/{val,test}_final.jsonl` | 每题一行：pass@k、select@k、cov@k（k=1,2,4,8,16）、正确数、检查器通过数、签名数等 |
| `eval/{val,test}_final.summary.json` | 汇总：`select@8`（主指标）、`pass@1`、`pass@8` 等 |
| `ckpt/step_XXXX/adapter/` | 每 10 步与最后一步的 LoRA（可直接被 vLLM 加载） |
| `logs/` | `train.log`、`eval.log`、`launcher.log`、`vllm_gpu*.log` |

汇总表：`python -m selectrl.eval.report --runs "runs/*"`（写到 `results/leaderboard.md`，含按比赛聚类的配对 bootstrap 区间）。

## 7. 中断与续跑

直接重新执行原命令。训练从最后完成的一步继续（`runs/<名字>/resume/` 每步原子更新，包括 adapter、优化器状态、数据游标、算法状态），已经采样但没来得及更新的那一步会复用已保存的采样结果；评测跳过已完成的题目。`train.sh` 自己也会在异常退出后自动重启最多 4 次。

## 8. 回传结果

```bash
bash scripts/collect_results.sh
```

生成 `results_bundle.tar.gz`（只含小文件，不含模型和检查点），发回即可。

## 9. 常见问题

- **vLLM 起不来**：看 `runs/<名字>/logs/vllm_gpu*.log` 末尾；最常见的是同卡上有别的进程占显存。推理专用阶段默认每卡占 85%，训练阶段 vLLM 占 40%（H200）或 42%（80 GB），可用 `SELECTRL_MEM_UTIL`、`SELECTRL_EVAL_MEM_UTIL` 调整。
- **训练端显存不够**：调小 micro-batch 的打包预算 `SELECTRL_MICRO_TOKENS`（只决定一次前向放几条序列，**不缩短上下文**；超过预算的长序列总是单独成批，上下文始终是 262,144）。
- **机器有 HTTP 代理**：脚本已把 127.0.0.1 加入 `no_proxy`，本地服务不会走代理。
- **`mismatch_abs_lp` 报错退出**：说明 vLLM 没有用上训练出的 LoRA（第 2 步就会检查），请把 `logs/` 发回。
