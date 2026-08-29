# InfiniSST 短语边界策略：流水线脚本

配套交接文档：[`docs/handoff_codex_20260830.md`](../../docs/handoff_codex_20260830.md)。
这些脚本在 aries 容器 `infinisst-phrase-jaxan-1` 内运行，工作副本在
`/mnt/gemini/data2/jiaxuanluo/phrase_pipeline/`。

| 脚本 | 作用 |
| --- | --- |
| `run_infer.sh <tag> <lora> [source_list]` | 单次受控 SimulEval 推理。已带 `--model-type w2v2_qwen25`（默认值是 llama31，用错会静默输出乱码），并在跑之前调 `check_lora_keys.py`。目录已存在时拒绝运行，避免新旧产物混淆。 |
| `check_lora_keys.py <lora> <reference>` | 校验 LoRA 键名与参考一致，不一致非零退出。**必要**：agent 用 `strict=False` 加载且丢弃 missing/unexpected，键名对不上会静默跑一个没有适配器的模型。 |
| `strip_lightning_prefix.py <in> <out> <reference>` | 剥掉 Lightning checkpoint 的 `model.` 前缀，并强制结果键集与参考逐键相同。 |
| `interp_lora.py <base> <phrase> <alpha> <out>` | 在底座与短语 LoRA 之间线性插值，零训练成本扫「结构收益 vs 鲁棒性」曲线。 |
| `instances_to_turns.py <instances.log> <outdir> <tag>` | SimulEval 产物转 turn 流。三条硬断言：字符数守恒、delay 单调、拼接文本逐字相等。 |
| `cmp_turns.py <名称=目录> ...` | 比较各档 turn 结构（turn 数、中位长度、碎 turn 占比、结尾落标点率）。**标签里不要出现 `=`**，解析按第一个等号切分。 |
| `text_latency.py` / `lag2.py` 思路 | 直接由 turn 流的 `delay_ms` 计算延迟，不依赖 ASR 与外部 API。合并档与 baseline 文本逐字相同，可精确相减。 |
| `finalize_v2.sh` / `finalize_v3.sh` | 训练收尾：等 checkpoint 写稳（NFS 上 16GB 要约 7 分钟，中途读会拿到撕裂档）→ 转换 → 剥前缀 → 打印权重位移 → 双档并行推理 → 结构判据。 |

`infinisst_patches/` 记录对 `~/InfiniSST` 的改动片段（该仓库不在此处）。
