# 交接：InfiniSST 短语边界策略（给接力 session）

**状态**：主问题（音色跳变）已定位并有两个可交付方案；一个诊断刚跑到一半，
它的结论会决定重训方案能否单独交付。**先读第 4 节的未决问题再动手。**

## 1. 问题与结论

**现象**：级联输出音色频繁跳变。

**根因**：InfiniSST 把译文切得太碎。同样 5 个 ACL6060 talk，现役模型产出
**1684 个 turn**、中位 **7 字**、**30.3% 在 5 字以内**。每个 turn 是 TTS 的
一次独立生成，音色在 turn 之间不连续。

**两条解法**（都已实测，级联指标见第 3 节）：

| | 做法 | turn 数 | 中位 | ≤5字 | 结尾落标点 |
| --- | --- | ---: | ---: | ---: | ---: |
| 现役 | 不处理 | 1684 | 7 字 | 30.3% | 32.6% |
| **A. 重训策略** | InfiniSST 自己学会在短语边界写出 | 881 | 15 字 | 5.8% | **85.2%** |
| B. TTS 侧缓冲 | 现役输出送 TTS 前按规则攒到短语边界 | 628 | 21 字 | 0.3% | 79.6% |

**用户明确倾向 A**（2026-08-29）：级联系统的分段策略应该在 ST 模型里，
缓冲是"很蠢的缓冲，解释性很弱，不优雅"。B 仅作对照保留。

## 2. 方案 A 的实现

训练侧改动在 taurus/aries 的 `~/InfiniSST`（**不在本仓库**，见第 5 节路径）：

- `train/dataset.py`：`_phrase_redistribute()` —— 在**同一张 chunk 网格上**
  重新分配文本（chunk 数与每 chunk 音频 patch 数一字不动，结构上杜绝音文错位）。
  未到短语边界的 chunk 目标置空串，模型学会 hold；到边界或超 `max_hold` 才写出。
  在 collator 的 multiplier 合并之后调用。
- `train/main.py` / `model/model.py`：透传 `--phrase_boundary /
  --phrase_max_hold_s / --phrase_min_chars`。
- collator `__init__` 会打印各 multiplier 的 hold 预算与退化计数，形如
  `m=1:8 m=2:4` / `degenerate 0/2`。**`max_hold_steps==1` 等价于「永不 hold」**
  （`held>=1` 立即成立），配置退化时必须能一眼看见。

交付权重：`/mnt/gemini/data2/jiaxuanluo/stage2_phrase_v2ep1_fixed.bin`
（1 epoch、103h 子集、multiplier 1-2、lr 1e-4、LoRA r32）。

**必踩的坑（已加防护，勿删）**：Lightning 把 SpeechLLM 挂在 `self.model` 上，
存出的 checkpoint 键名多一层 `model.` 前缀；而 `agents/infinisst.py:594/667` 用
`load_state_dict(..., strict=False)` 且**丢弃返回的 missing/unexpected**，
键名不匹配时会**静默跑一个没有适配器的模型、全程零报错**。
- `phrase_pipeline/strip_lightning_prefix.py` 剥前缀并强制键集与参考逐键相同；
- `phrase_pipeline/check_lora_keys.py` 在 `run_infer.sh` 里前置校验，对不上拒绝运行。

另一个默认值陷阱：`agents/infinisst.py --model-type` 默认 `w2v2_llama31`，
用它加载 Qwen 权重会输出乱码而不报错。**必须显式给 `--model-type w2v2_qwen25`。**

## 3. 级联指标（BC 口径，1× 档）

| 配置 | BLEU | XCOMET | 漏译 | 静默 p90 | 最坏静默 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 现役 | 33.30 | 0.484 | 0.0% | 1.9 s | 9.6 s |
| **A. 重训 ep1** | **38.32** | **0.652** | 0.2% | **5.8 s** | **9.6 s** |
| B. 缓冲 hold=4 | 38.42 | 0.650 | 0.0% | 7.7 s | 9.6 s |
| B. 缓冲 hold=8 | 39.85 | 0.690 | 0.2% | 11.5 s | 21.1 s |

A 与 B(hold=4) 质量持平，但 **A 的静默更低**（5.8 s vs 7.7 s）——模型是
"一次说完整"，缓冲是"攒够了才放"，后者必然付延迟。

**两个口径都要报**：canonical（整轨 ASR）因 Qwen3-ASR 长音频少转写而
**系统性低估长 turn 配置**；BC（逐 turn 切分后独立 ASR）覆盖完整但
**对极短 turn 有惩罚**（1-2 字 turn 的字错率 44%，3-5 字 19%，21+ 字仅 7%）。
只报一个会得出相反结论。

## 4. 未决问题（**接力的第一件事**）

**方案 A 在 1.5× 语速档上级联质量掉**：BC BLEU 38.32(1×) → 31.27(1.5×)，
低于现役同档的 35.45。而缓冲档在 1.5× 不掉（39.88）。这是 A 目前唯一的短板，
也是它能否按"单版本收敛"裁定单独交付的关键。

**已排除的两个解释**（都做过实验，不要重复）：
1. ~~声学问题（1.5× 音频对 InfiniSST 是分布外）~~ —— 现役模型在 1.5× 上
   BC BLEU 反而更高（35.45 vs 33.30）。底座不掉，所以不是声学。
2. ~~内容密度（chunk 里内容更密，模型没见过）~~ —— v3 把训练 multiplier
   从 1-2 放宽到 1-4（覆盖对应密度），1.5× 结构纹丝不动（71.9% vs 73.3%）。

**诊断结论（LoRA 插值扫描，已跑完；脚本 `scripts/infinisst_phrase/interp_lora.py`）**

`W(a) = W_base + a*(W_phrase - W_base)`，a=0 是现役、a=1 是 ep1，零训练成本。

结构与文本 BLEU（SimulEval，5 个 ACL6060 talk）：

| α | 权重位移 | 1× 结尾标点 | 1× 中位 | 1× BLEU | 1.25× BLEU | 1.5× BLEU | 1×→1.5× |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0（现役） | 0% | 32.6% | 7 字 | 54.12 | 53.50 | 52.20 | **−1.92** |
| 0.5 | 7.4% | 53.0% | 8 字 | 52.34 | — | 48.70 | −3.63 |
| 0.75 | 11.0% | 76.3% | 11 字 | 50.57 | — | 49.19 | −1.38 |
| 1.0（ep1） | 14.7% | 85.2% | 15 字 | 50.67 | 49.09 | 48.78 | **−1.92** |

**决定性结论：ep1 的文本质量随语速下降的幅度与现役完全相同（都是 −1.92）。**
也就是说重训**没有**损害模型对快速语音的鲁棒性。而同一个 ep1 的**级联** BC BLEU
却从 38.32 掉到 31.27（−7.05）。**文本没坏、级联坏了 → 1.5× 的损失发生在
InfiniSST 之后**（TTS 合成或对齐/打分环节），不在翻译本身。

结构侧同样支持：ep1 @1.5× 的结尾标点 73.3%，仍远高于现役同档的 33.1%，
碎 turn 2.5% vs 13.4%——**分段策略在 1.5× 上依然有效**。

（附带：α=0.5 的 −3.63 是全表最差，说明"降低 LoRA 强度"不改善鲁棒性，
反而更差；插值不是可用的缓解手段，不必再试。）

**接力的第一件事**：验证 1.5× 档的下游。1.5× 下整场时长只有 2/3，而中文译文
长度不变，TTS 很可能被时间预算挤压。`score_generic.py` 的输出里每个 talk 有
`target_s`，与 TTS 实际生成时长对比即可证实或排除。若证实，**方案 A 可直接定版**，
1.5× 作为下游问题另案处理。

## 5. 路径与资源

**本仓库**（`luojiaxuan/S2S_omni`，main）：
- `scripts/moss_multiturn_infer.py` —— TTS 多轮推理（含 `--phrase-merge` 缓冲档，
  方案 B 用；方案 A 不需要它）
- `scripts/acl_cascade_eval/run_eval_queue.sh` —— TTS 评测队列。模式
  `slidingsoft3`（方案 A）/ `slidingsoft3phr4`（方案 B）
- `scripts/acl_cascade_eval/score_generic.py` —— ASR + 建 rundir。
  **`PREFIX` 既要作环境变量传、又要拼进 `mode` 字符串，两处必须一致且无校验**
  （踩过：目录名 `tts_wavs_{tag}_{mode}_{PREFIX}`，而 score_generic 找
  `tts_wavs_{tag}_{mode}`，所以传给它的 mode 必须已含 PREFIX）
- `docs/experiment_ledger_moss_tts_cascade_20260808.md` —— 完整台账（4.-23 起是本线）
- `docs/handoff_infinisst_phrase_20260828.md` —— 环境坑与更早的交接
- 音频浏览页：`projects/acl6060_s2s_metrics_seed/artifacts/acl6060_segale_diagnostics/v7_audio_browser.html`
  **注意仓库里有两份同名文件**，GitHub Pages 只发布 `projects/...` 那份
  （`docs/` 那份是副本，改了不会上线）。线上：
  https://luojiaxuan.github.io/S2S_omni/v7_audio_browser.html

**aries 容器 `infinisst-phrase-jaxan-1`**（GPU 2,3,6,7）：
- InfiniSST 仓库：`/mnt/taurus/home/jiaxuanluo/InfiniSST`（经容器内
  `/home/jiaxuanluo` 软链访问）
- 流水线脚本：`/mnt/gemini/data2/jiaxuanluo/phrase_pipeline/`
  （`run_infer.sh` / `instances_to_turns.py` / `cmp_turns.py` /
  `strip_lightning_prefix.py` / `check_lora_keys.py` / `interp_lora.py` /
  `finalize_v3.sh`）
- 权重：`/mnt/gemini/data2/jiaxuanluo/stage2_phrase_v2ep1_fixed.bin`（交付候选）、
  `stage2_phrase_v3_fixed.bin`、`stage2_phrase_a0.5.bin`、`stage2_phrase_a0.75.bin`
- 评测音频与 source list：`/mnt/gemini/data1/jiaxuanluo/acl6060_eval/`
  （`dev.source` / `dev.source.speed125` / `dev.source.speed150`）
- 环境：conda `infinisst`；PYTHONPATH 顺序**重要**（仓库必须在最前，否则
  fairseq 根目录的 `train.py` 会抢占 `train` 包名）：
  `$PWD:/mnt/gemini/data2/jiaxuanluo/hydra_fs:/mnt/gemini/data2/jiaxuanluo/fa_abitrue:/mnt/gemini/data2/jiaxuanluo/tf47:/mnt/aries/data6/jiaxuanluo/fairseq-0.12.2`
  （训练用 `tf446`，推理用 `tf47`——`patch_hf` 与 `patch_qwen25` 对
  transformers 版本要求冲突，4.47.1 是唯一同时满足的版本）

**hyper00 容器 `sglang-omni-jaxan-page`**（无 GPU）：音频页 demo 音频编码。
评测产物在 `/data04/jaxan/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench/`。
**hyper00 的 HF cache 是同租户共享的，gavinlaw 令牌已被他人覆盖**——
上传 HF 要用本机（Mac）的令牌。

**其他主机**：hyper01 有 XCOMET-XL 权重（13G，`/data02/jaxan/.cache/huggingface`）
与 SEGALE venv（`/data02/jaxan/venvs/acl6060-segale`）；**SEGALE 对齐会把仓库
git revision 写进结果，已有数字全是 `d0041438`，那份在 hyper00 容器内**，
换别的拷贝会让对比混入代码版本差异。moss **不要用**：hf-mirror 对大文件
只有 229 B/s，hyper00→moss 实测 22 KB/s，搬 11G 要 139 小时。

## 6. 测量纪律（踩过的坑，别重犯）

- **同 seed 的 TTS 生成不可复现**：同一张卡重跑，turn 数相同但帧数差 2-4%
  （并发度改变 cuBLAS kernel 选择）。**小于约 0.5 BLEU 的差异不能用来定版。**
- **换权重后"结果变了"不等于"换上了我以为的那个权重"**：要直接验证加载
  （键匹配数、或权重相对底座的平均位移），不要从"有变化"倒推。
  三秒的检查能挡住整轮无效实验。
- **定版判据必须覆盖全部部署档位**：只看 1× 就定版会被 1.5× 推翻。
- **断言某方案的"固有局限"之前，先扫完它的参数空间**：缓冲档的
  `max_hold` 从 8 调到 4，最坏静默 21.1 s → 9.6 s 而质量只掉噪声量级。
- **`run_eval_queue.sh` 续跑不截断已存在的 summary**：`.done` 缺失时会
  **追加**而非覆盖，产生两行 summary。修法是重跑前先 `rm -f` 掉 summary。
  （对账方法：summary 记录的总帧数 ×0.08 应等于 wav 时长。）
