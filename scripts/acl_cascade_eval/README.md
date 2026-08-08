# ACL6060 级联评测与 v4 训练脚本

这些脚本此前只存在于 hyper00 容器和 Tilde 的家目录里，现收进 Git 作为
source of truth。路径都是**容器内/集群内**的绝对路径，换机器需要改开头的
常量。详细结论见
[`docs/experiment_ledger_moss_tts_cascade_20260808.md`](../../docs/experiment_ledger_moss_tts_cascade_20260808.md)。

## hyper00（评测侧，容器 `sglang-omni-jaxan`）

| 脚本 | 作用 |
| --- | --- |
| `run_eval_queue.sh <gpu> <tag> <ckpt> <mode> <talk...>` | 通用 TTS 队列。`mode` 取 `reset`（每 11 turn 一个会话，输入 `rows`）、`sliding`（w=11）或 `slidingN`（任意窗口，输入 `swrow` 整场一行）。带 `.done` 标记，可断点重跑 |
| `run_base_queue.sh <gpu> <talk...>` | 原生未微调 MOSS 的 baseline 队列 |
| `score_generic.py <tag> <mode>` | 拼接整场 wav + 自建 Qwen3-ASR 转写，产出 run dir 的 `instances.log` |
| `score_base.py` | baseline 专用版（额外记录崩溃 session 数到 `session_qa.json`） |
| `score_chain_generic.sh <tag> <mode>` | 完整链：ASR → SEGALE inputs → SEGALE 对齐 → BLEU → XCOMET-XL |

打分链依赖容器内的 `/data/venvs/segale_eval2` 与
`/data/speech-to-speech-latency`（pinned `d0041438`），以及 47500 端口上的
自建 Qwen3-ASR：

```bash
CUDA_VISIBLE_DEVICES=5 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 47500 --mem-fraction-static 0.45
```

**注意**：`score_*.py` 对缺失/失败的 session 是静默跳过的。曾因此踩坑
（`sliding6` 因 `rows_suffix` 精确匹配 bug 读错输入文件，拼出 0 秒音频却
一路跑到 SEGALE 才报错）。用前请确认日志里的 `missing_rows` 为 0。

## Tilde（Slurm 训练侧）

| 脚本 | 作用 |
| --- | --- |
| `tilde_setup.sh` | 一次性环境搭建：clone S2S_omni + MOSS-TTS 并打 `third_party/moss_tts` 补丁，建 conda env（torch 2.11.0+cu130 / transformers 5.6.0） |
| `prep_rows.py <train_v3.jsonl> <out_dir>` | 从 v3 数据集切出干净行（无 `context_only`），分成 originals 与 long sessions |
| `v4_selfhist.sbatch` | 8 卡闭环自生成历史，按 row-id 跳过已完成，支持抢占重排队 |
| `v4_train.sbatch` | 同节点并排训练处理组与对照组 |
| `upload_v4_ckpts.py` | checkpoint 经 HF 中转到 hyper（两机不可达） |

Tilde 的硬约束：QoS `guest-dev` 上限 8 GPU、墙钟 24h、`PreemptMode=REQUEUE`，
所以每个作业都必须可断点续跑，状态写 `/home/guests/zhen` 持久盘。

**两组必须用相同的进程数**：accelerate 每个 optimizer step 会把 lr scheduler
推进 `num_processes` 次而 `sft.py` 没有乘回来，进程数不同会导致有效学习步数
不同，直接污染 A/B。详见台账 6.2。
