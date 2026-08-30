# 交接：Codex → Claude（2026-08-30）

这份文档只记录从 Claude 上次交接之后新增、修正和完成的工作。更早的
InfiniSST phrase-boundary 训练过程仍见 `docs/handoff_codex_20260830.md`；完整实验
历史见 `docs/experiment_ledger_moss_tts_cascade_20260808.md`。

## 0. 先看这里

- **Git SoT**：<https://github.com/luojiaxuan/S2S_omni>，研究分支
  `luojiaxuan/codec-decoder-context-ab`。本交接前已推送的证据状态为
  `8eef511`；不要回到旧 `main` 或 `/Users/luojiaxuan/Documents/S2S_backlog`。
- **本机工作树**：`/Users/luojiaxuan/codec_ab`，当前无未提交修改。
- **speaker 跳变根因已确认**：同一场对话内每 turn reset codec decoder causal
  state 会造成音色跳变；跨 turn 保留 decoder context 是直接修复。一个完整用户
  session 结束时仍应 reset，不是要求跨独立 session 永久保留状态。
- **phrase 匹配 TTS 已训练并复评**：v8 checkpoint 已上传；连续 decoder 下
  1×/1.5× BLEU 为 **34.09/29.61**，1.5× 仍低 **4.48**。因此旧 v7 TTS 的
  turn-distribution mismatch 不是退化的充分解释。
- **当前 policy 决策**：保留连续 codec context。InfiniSST phrase policy 不再承担
  修音色职责，且暂不作为跨语速默认配置；它是否继续用于韵律/切分要由 latency
  与 1.5× 质量共同决定。
- **没有运行中的任务容器或 GPU 作业**。hyper00 的本轮容器已正常停止并删除，
  持久化 run root 仍在。

## 1. Claude 交接后的主要进展

### 1.1 仓库与 SoT

用户指定 Claude 交来的实际仓库/分支为新的项目 SoT。已把
`/Users/luojiaxuan/codec_ab`、GitHub `luojiaxuan/S2S_omni`、分支
`luojiaxuan/codec-decoder-context-ab` 定为主线，代码、实验合同、结果和进展均已
提交并推送。起点是 `main@ec60bfb`，本轮提交从 `d33e93a` 延续到 `8eef511`。

### 1.2 codec decoder context 破案

固定 talk110 已生成的 194 个 turn、8,581 帧 audio codes，只改变解码方式：

| 条件 | decoder 生命周期 | 输出时长 |
| --- | --- | ---: |
| A | 每 turn 新建 `AudioStreamDecoder` 并重新进入 `codec.streaming()` | 686.48s |
| B | 整个 talk 共用一个 decoder 和一个 streaming context | 686.48s |

两边 codes 完全相同，用户听感确认 B 明显更稳定。由此确认 codec decoder reset
是 speaker 音色跳变的直接原因；InfiniSST 攒更长文本只能减少边界，不是必要修复。

证据与上游工作：

- 本地报告：`projects/infinisst_moss_tts_cascade/artifacts/codec_decoder_context_ab_talk110/report.json`
- 根因文档：`docs/moss_realtime_codec_context_issue.md`
- 60 秒 A/B Release：<https://github.com/luojiaxuan/S2S_omni/releases/tag/talk110-codec-context-ab-20260829>
- sglang-omni issue：<https://github.com/sgl-project/sglang-omni/issues/1812>
- 当前实现是 PR 1410，不是旧 PR 1192/1368：
  <https://github.com/sgl-project/sglang-omni/pull/1410>
- 已在 PR 1410 留下两处 inline review：slot 应按 `session_id` 跨 turn 复用；正常
  turn final flush 不能 release/reset slot。普通 comment 已附 issue 和 60 秒 WAV。
- 2026-08-30 再核对：PR 1410 仍是 open draft，head
  `c5455d9934f0d7e44c16f0ba13ef7849c1f0e323`；issue 1812 仍 open。

服务语义不要混淆：同一 application session 内的 turn 要延续 causal state；session
close、TTL、abort 或失败时必须释放并 reset。若无法保存无限历史，可 prefill 有界的
近期 audio-code context，而不是在每个 turn 从零开始。

### 1.3 连续 decoder 四格复评与归因纠正

先用旧 v7 TTS 跑了连续 decoder 四格合同：

| InfiniSST 输出 | 1× BLEU | 1.5× BLEU | 跨速档变化 |
| --- | ---: | ---: | ---: |
| baseline + 旧 v7 TTS | 30.16 | 33.97 | +3.81 |
| phrase v2-ep1 + 旧 v7 TTS | 36.80 | 31.34 | −5.46 |

这张表最初被错误用于判断 phrase policy。用户指出：phrase 改了 text、turn length
和 boundary 分布，却仍用旧 trajectory 训练的 v7 TTS，存在分布混杂。结论已在
`270b09a` 撤回并修正：旧表只能诊断“旧 v7 TTS 接 phrase 输出”的联合行为，不能
升降级 InfiniSST policy。

轻量旧结果：
`projects/infinisst_moss_tts_cascade/artifacts/codec_context_phrase_eval_20260830/summary.json`。

### 1.4 phrase-policy 匹配 MOSS-TTS SFT

为控制上述混杂，新增 `scripts/build_moss_phrase_prepared.py`，在不改变每行拼接
文本和 codec frame 总数的前提下，按 InfiniSST 相同规则重排训练 turn。这里匹配的
是训练 trajectory 的 write/boundary 分布，不是把模型推理生成文本直接当监督。

训练集共 46,753 行：

| 组成 | 行数 |
| --- | ---: |
| v6 base | 36,529 |
| phrase full | 6,385 |
| phrase mid-start | 3,839 |

phrase 重排参数固定为 `multiplier=2`、`chunk=0.96s`、
`phrase_max_hold_s=7.68`、`phrase_min_chars=6`。6,385 条 full rows 从
157,887 个旧 turn 变为 38,621 个 phrase turn，codec frames 2,292,071 完全守恒。

训练配置：base
`OpenMOSS-Team/MOSS-TTS-Realtime@75682787d8e2fcc73faca37ba2931453ca9c4022`，
3 张 H200，global batch 15，learning rate `1e-5`，bf16，1 epoch，seed 42；完成
3,117 optimizer steps，末步 loss 3.4778。

checkpoint：

- HF：<https://huggingface.co/gavinlaw/moss-tts-realtime-infinisst-en-zh-v8-phrase/tree/521e09faa2c318801673d852e29e82a2476263b0>
- revision：`521e09faa2c318801673d852e29e82a2476263b0`
- `model.safetensors`：4,663,931,664 B
- SHA256：`074964929bce38b9069efc07789336dd231de6d5426554c2662169b610a5e4e9`

### 1.5 匹配 TTS 后的连续 decoder 复评

冻结 talk 268/367/590/110/117 的实际 InfiniSST phrase v2-ep1 输出，只替换为新
v8 TTS；每 talk 一个连续 codec decoder context。TTS seed 42、sliding window 11、
soft reset keep 3；逐 turn Qwen3-ASR，SEGALE `d0041438`，SacreBLEU
`tokenize=zh`。10/10 talk 均成功。

| 指标 | 1× | 1.5× | 变化 |
| --- | ---: | ---: | ---: |
| BLEU | **34.0855** | 29.6075 | **−4.4780** |
| null alignment | 7/467，1.50% | 34/476，7.14% | +5.64 pp |
| InfiniSST turns | 881 | 711 | −170 |
| 输出字符 | 17,989 | 18,540 | +551，+3.06% |
| TTS 生成音频 | 3593.84s | 3766.40s | +172.56s，+4.80% |

匹配训练把旧 v7 + phrase 的跨速档差值从 −5.46 缩小为 −4.48，只改善约
0.98 BLEU，没有反转。1.5× 输出文字更多，TTS 音频也更长，null alignment
显著增加。现在可以排除“旧 TTS 分布偏移足以解释退化”，但单靠级联 BLEU 仍
不能把剩余问题唯一定位为 InfiniSST、TTS 声学质量或 ASR/SEGALE 耦合。

轻量结果：
`projects/infinisst_moss_tts_cascade/artifacts/phrase_matched_tts_eval_20260830/summary.json`；
研究日志：`projects/infinisst_moss_tts_cascade/research_log.md`；实验台账 4.-35。

## 2. 最新 latency 发现：multiplier=1

用户随后质疑“等到句内标点结尾”会增加很多 latency，并要求看
`multiplier=1` 的训练例子。这里有两个容易混淆的事实：

1. InfiniSST phrase v2 的 collator 在线均匀随机采样 `m∈{1,2}`。一个 epoch 中
   预期约一半 batch 是 m1，但没有静态 m1 文件，也不保证同一 row 同时见过 m1
   和 m2 两份监督。
2. 本轮 MOSS-TTS v8 的 phrase full/mid-start 数据**固定为 m2**，没有 m1 phrase
   rows。因此如果改 InfiniSST deployment policy，v8 不能被视为新 policy 的匹配
   TTS，必须同步重做训练数据。

对 6,385 条真实 source trajectories 全量模拟当前 policy：

| policy | 非空 write | 字符平均额外延迟 | 中位 | p90 |
| --- | ---: | ---: | ---: | ---: |
| multiplier=1 | 43,943 | 1.81s | 0.96s | 4.80s |
| multiplier=2（当前 v8 数据） | 38,621 | 2.31s | 1.92s | 5.76s |

m1 能把平均字符延迟降低约 0.50s，中位和 p90 各降低 0.96s，但不会消除 phrase
gating。m1 的 29.7% write 以句内标点结尾；这些 write 的首块等待中位 1.92s、
p90 5.76s。配置名写 `max_hold=7.68s`，按当前 `held>=8` 的计数语义，m1 首块
实际最大额外等待是 6.72s。

句内标点本身是更早的释放点，比只等句号更快。真正值得改的是当前实现只检查
整个 buffer 的**最后一个字符**。例如 `政策中还有更多内容。我` 已包含句号，
但 tail 是“我”，不会 write；它会继续等到后面的 `细节，`。另一个例子是
`你们共度时光。或者`，句号同样被 tail-only 检查跳过。

用户看过的 10 个 m1 例子来自同一份训练 manifest，代表从 0s 到 6.72s 的等待：

| # | 连续 0.96s delta（`∅` 为无新增文字） | write | 首块/平均字等待 |
| --- | --- | --- | ---: |
| 1 | `打开蓝牙功能，` | 打开蓝牙功能， | 0/0s |
| 2 | `随着`｜`我们的不断发展，` | 随着我们的不断发展， | 0.96/0.21s |
| 3 | `政策中还有更多内容。我`｜`不会逐一讲解每个`｜`细节，` | 政策中还有更多内容。我不会逐一讲解每个细节， | 1.92/1.34s |
| 4 | `我们`｜`∅`｜`非常高兴能见到`｜`你们中的许多人，` | 我们非常高兴能见到你们中的许多人， | 2.88/0.78s |
| 5 | `的机会`｜`，可以让`｜`你们`｜`发挥领导`｜`作用，开展重要工作，` | 的机会，可以让你们发挥领导作用，开展重要工作， | 3.84/1.39s |
| 6 | `我`｜`∅`｜`∅`｜`以前从未获得过这样`｜`的荣誉，这`｜`对我意义重大，` | 我以前从未获得过这样的荣誉，这对我意义重大， | 4.80/1.30s |
| 7 | `这`｜`将是`｜`∅`｜`十年来我第一次`｜`无法与`｜`你们共度时光。或者`｜`，如果之前没见过面，` | 这将是十年来我第一次无法与你们共度时光。或者，如果之前没见过面， | 5.76/1.69s |
| 8 | `我们`｜`∅×6`｜`很快就会在Tdl存储库中发布本次会议的会议纪要和相关材料，` | 我们很快就会在Tdl存储库中发布本次会议的会议纪要和相关材料， | 6.72/0.45s |
| 9 | `我们将`｜`∅`｜`在今年`｜`春天着手处理此事，` | 我们将在今年春天着手处理此事， | 2.88/0.82s |
| 10 | `它`｜`∅`｜`∅`｜`利用光的红外波段`｜`∅`｜`在设备之间进行通信，` | 它利用光的红外波段在设备之间进行通信， | 4.80/1.12s |

## 3. 代码与复现状态

本轮主要提交分组：

- `d33e93a`：明确新仓库/分支 SoT。
- `9b1df72..a2eb237`：固定 codes A/B、根因文档、issue 1812、PR 1410 review
  和 60 秒附件。
- `c8812ae..270b09a`：连续 decoder 四格评测、打分兼容和分布混杂归因纠正。
- `f8fcfe1..60f6557`：phrase 匹配 TTS 数据、训练合同、可恢复训练、复评资源。
- `ee47d48`：v8 训练与复评结果、research log、HF SoT、评测环境修复。
- `8eef511`：README SoT 指向上述结果。

完成评测时发现并修了三类可复现性问题：HF partial snapshot 被误当完整资源、
训练与评测共用 site-packages 导致 NumPy ABI 混杂、SEGALE wheel 缺
`vecalign/dp_core*.so` 且 `site-packages/bin` 未进 PATH。现在
`scripts/prepare_codec_phrase_eval_runtime.py` 使用隔离 `env/eval-site`、固定
NumPy/spaCy/SacreBLEU 版本、复制并校验 `dp_core`；runner 会补 PATH。

**剩余风险**：完成本轮数字时这些依赖是手工修好后验证的；`ee47d48` 中新的
fresh-bootstrap 脚本只做了 `py_compile` 和静态检查，尚未从空 `eval-site` 做一次
完整 smoke。下次复评启动前先在新容器中执行 bootstrap 和最小 import/align smoke，
不要等 10 个 talk 合成完才发现环境问题。

## 4. Artifacts / Source of Truth

### Git

- Repo：<https://github.com/luojiaxuan/S2S_omni>
- Branch：`luojiaxuan/codec-decoder-context-ab`
- 结果配置：
  `projects/infinisst_moss_tts_cascade/configs/phrase_matched_tts_v8_20260830.json`
- 结果 summary：
  `projects/infinisst_moss_tts_cascade/artifacts/phrase_matched_tts_eval_20260830/summary.json`
- 完整台账：`docs/experiment_ledger_moss_tts_cascade_20260808.md`，新结论为 4.-35。

### Hugging Face

- v8 model：`gavinlaw/moss-tts-realtime-infinisst-en-zh-v8-phrase`
  @ `521e09faa2c318801673d852e29e82a2476263b0`。
- 训练数据与完整 10-talk WAV/ASR/SEGALE：
  `gavinlaw/infinisst-moss-tts-en-zh-multiturn`。重数据上传完成于
  `edd953263d1959f8245c12f45156c7c3cc0fde3f`；随后补最终 contract/summary，
  当前 repo head 为 `3fa7f4b1f43e5c57d9151669a17ebdcb8941f6be`。
- 训练数据路径：`prepared/phrase_v8_20260830/`；压缩 input 为 840,317,376 B。
- 评测路径：`eval/phrase-matched-tts-v8-20260830/`。

### hyper00 持久化缓存

- run root：`/data02/jaxan/tmp/runs/20260830-200824-401864000`
- checkpoint：`outputs/model/checkpoint-epoch-0/`
- 训练数据：`data/train_phrase_v8_20260830.jsonl.zst`
- 评测：`eval/output/{c2,c3}/` 与 `eval/result/{c2,c3}/`
- 原容器 `sglang-omni-jaxan-20260830-201134-498460000` 已删除，不能复用；若继续
  实验应按新任务创建新的 timestamp 容器。

### 仍欠 SoT 的旧产物

- InfiniSST phrase v2 ep1 LoRA 仍只在 Aries：
  `/mnt/gemini/data2/jiaxuanluo/stage2_phrase_v2ep1_fixed.bin`，状态
  `PENDING_HF_UPLOAD`，SHA256
  `1f2c795f8500d1d8c78e5d93e09d09bb43c6d7941cfc16cddb81a01af189a617`。
- talk110 完整 686.48s A/B bundle 仍是 local staging；GitHub Release 已有足够
  上游复核的 60 秒片段，但完整 bundle 尚未进 HF。
- 早先连续 decoder 四格旧 v7 评测的完整 1.6G 产物仍标记
  `PENDING_HF_UPLOAD`；轻量 summary 已在 Git，不影响当前 v8 结论复核。

## 5. Claude 接手后的第一件事

不要立即再训。先对 6,385 条 trajectory 做一个纯 CPU、无需模型的 policy
消融，回答 latency 问题：

1. A：当前 tail-only punctuation、m1。
2. B：当 buffer 内出现可用标点时，write 到**最后一个满足 min_chars 的标点**，
   标点后的 suffix 留在 buffer，而不是要求整个 buffer 的尾字符是标点。
3. 两边报告字符平均/p50/p90 latency、首块 p90/max、write 数、中位字数、≤5字率、
   句末/句内标点比例，并抽同一批 10 个 examples。

这个消融能判断用户看到的 2–6 秒等待中，有多少来自 m2 网格，有多少来自
tail-only 规则。看到结果后再请用户选择是否改 policy。

若决定重训：

- InfiniSST supervision 应显式保证同一 source row 同时覆盖 m1/m2，不能再依赖
  单 epoch 中随机只见一个 multiplier；具体是 paired materialization 还是确定性
  batch schedule 需要先冻结合同。
- policy 一旦改变，MOSS-TTS 也必须用相同 m1/m2 turn distribution 重训；当前
  v8 只匹配 m2，不可直接拿来裁决新 policy。
- 最终仍复用同一 5-talk、1×/1.5×、连续 codec decoder 合同，并同时检查 BLEU、
  null、输出字符、生成音频时长和人耳 speaker/prosody。

若不改 policy，而是继续追 1.5× 退化，则下一步应把 TTS/ASR fidelity 与 SEGALE
alignment 分开：对同一 frozen text 比较逐 turn ASR 字符召回、codec/audio 时长和
text-direct score，先确定 34 个 null 是音频内容没生成、ASR 没听到，还是 alignment
没对上。不要再仅凭一个级联 BLEU 给单层定罪。
