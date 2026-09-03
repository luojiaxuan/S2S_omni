# SimulS2ST-Omni(hasaki321,EMNLP 2026)在 ACL 60/60 三 talk 子集上的实跑对照(2026-09-03)

对象:[SimulS2ST-Omni](https://github.com/hasaki321/SimulS2ST-Omni)
(arXiv 2607.19810;Qwen2.5-Omni-3B thinker + 0.4B OmniTalker + VoiceBox/Vocos,
约 2k 小时配对数据;HF `HA-SA-ki/SimulS2ST-Omni`,18.9 GB)。仓库只放推理。
本次用其 README 推荐配置在 hyper01 单张 H200 上实跑我们的 3-talk 子集
(talk 268/110/117,即 A′/B′ 终局对照用的同一子集),目的:回答"他们的
S2T/S2S 表现如何、听起来自然不"。

## 口径

- **S2TT**:他们的 SimulEval agent 直接出文本,SimulEval corpus BLEU
  (整场一 instance,tokenize=zh)。我方对照取 aries
  `runs/infer_phrase/{baseline,phrv2ep1}` 的 instances.log 同法计算。
- **S2ST ASR-BLEU(主口径)**:每个 talk 的整段译文音频(SimulEval 零抖动
  FIFO 渲染出的 `wavs/<i>_pred.wav`;我方按 InfiniSST 发射时刻同法渲染)
  **一次性送 Qwen3-ASR-1.7B**(helper 内部按固定 30 s 窗切给模型,与 turn
  无关)→ SEGALE d0041438 对齐 → 句级 SacreBLEU zh。用户裁定:不按 turn
  切 ASR。`w1_ext.py --merge-window-s 1e6 --asr-window-s 30`。
  历史/无效口径留作注记:(a) A′/B′ 当时用的"逐 turn ASR + 句号拼接";
  (b) 我临时加的"相邻 piece 合并到 ≥12 s"折中;(c) helper 默认 120 s 窗的
  整段 ASR——Qwen3-ASR 在 120 s 窗下丢 25–30% 字(转写 2591 字 vs 参考
  3532),两边都失真,作废。
- **时间线**:两者都是 SimulEval 式 FIFO 放置(turn 在发射时刻到达,
  前一段没播完就排队),我方按 swrow 的 InfiniSST 发射时刻渲染
  (`render_cascade.py`)。均为仿真下界(不含计算)。

## 结果

### S2TT(文本,SimulEval corpus BLEU,3 talk)

| 系统 | BLEU | 逐 talk 268/110/117 | 备注 |
| --- | ---: | --- | --- |
| SimulS2ST-Omni m2(2 s chunk,beam 4) | **54.98** | 52.2 / 56.0 / 56.6 | 结尾漏出 `<|endoftext|>Human` |
| SimulS2ST-Omni m3(3 s chunk,beam 4) | **56.37** | 53.3 / 56.9 / 58.7 | talk 268 结尾漏出 system prompt 句 |
| ours 原版 InfiniSST(1×,LAAL 5.7 s) | 54.02 | 53.0 / 55.3 / 53.7 | 5-talk 54.12 |
| ours phrase InfiniSST v2ep1(1×,LAAL 3.9 s) | 51.78 | 48.2 / 54.0 / 52.6 | 5-talk 50.35 |

论文自报 ACL60/60-dev en→zh S2TT 52.30 BLEU @m3(beam),与本次 56.37 同量级
(子集不同)。他们的 SimulEval doc 级 LAAL 报 26–28 s,该指标在整场实例上
不可用(论文用 StreamLAAL),延迟看下面的音频时间线。

### S2ST(ASR-BLEU,同一打分链,3 talk)

主口径(整段一次 ASR,30 s 内部窗):

| 系统 | ASR-BLEU | 逐 talk 268/110/117 | 段数 / null | 转写字数 268/110/117(参考 3532/3197/3579) |
| --- | ---: | --- | --- | --- |
| SimulS2ST-Omni m2 | **44.15** | 41.9 / 44.4 / 45.9 | 262 / 0 | 3118 / 3034 / 3360 |
| ours B′ = phrase InfiniSST + v8 TTS,1× | **35.92** | 36.0 / 39.0 / 33.2 | 276 / 3 | 3766 / 3672 / 4234 |

注记口径(同一批音频,仅 ASR 切法不同):

| 系统 | 逐 turn/piece + 句号拼接 | 合并 ≥12 s | 整段 120 s 窗(作废,ASR 丢字) |
| --- | ---: | ---: | ---: |
| SimulS2ST-Omni m2 | 31.58(piece 均 1.9 s,被切碎) | 41.73 | 34.57 |
| ours B′ v8 | 33.28(原 A′/B′ 口径;时间线重打 33.17) | 35.64 | 31.86 |
| ours A′ | 19.54 | — | — |

### 音频时间线(FIFO 仿真,单位秒)

| 系统 | talk | 译文音频/源时长 | 首次出声 | 收尾偏移 | FIFO 积压 mean / p90 / max |
| --- | --- | ---: | ---: | ---: | --- |
| SimulS2ST-Omni m2 | 268 | 634/737(86%) | 4.0 | −0.7 | 1.18 / 3.44 / 6.8 |
| SimulS2ST-Omni m2 | 110 | 603/703(86%) | 2.0 | +1.9 | 0.66 / 1.90 / 3.5 |
| SimulS2ST-Omni m2 | 117 | 593/729(81%) | 2.0 | +0.7 | 0.35 / 1.12 / 3.0 |
| ours v8 级联 1× | 268 | 774/737(105%) | 9.6 | +46.0 | 29.4 / 42.8 / 46.8 |
| ours v8 级联 1× | 110 | 782/703(111%) | 3.8 | +89.0 | 41.8 / 88.6 / 93.7 |
| ours v8 级联 1× | 117 | 925/729(127%) | 7.7 | +203.6 | 108.1 / 197.4 / 203.4 |

## 判断

1. **文本质量同档**:他们 m2/m3 的 S2TT 与我们原版 InfiniSST 持平或略高
   (54.98/56.37 vs 54.02),高于 phrase 版(51.78)。
2. **语音译文质量:他们领先约 8 分**(44.15 vs 35.92,整段一次 ASR 主口径)。
   逐 turn 口径下曾出现的"我们略高"是假象——他们 1.9 s 的 piece 被 ASR 在
   短语中间切碎;任何按 turn 切 ASR 的口径都对 piece 粒度敏感,不再用于
   跨系统比较。我方转写字数超参考 7–18%,与下面的音频超长一致。
3. **实时性是他们的决定性优势**:译文音频只占源时长 81–86%,收尾偏移
   ≈0,FIFO 积压秒级;我们 v8 在 1× 下三个 talk 的音频总量都超过源时长
   (105–127%),积压几十到两百秒——**译文物理上放不完**。这不是 InfiniSST
   的延迟,是 TTS 产出的音频总量超过源,是级联要解决的核心问题
   (语速/压缩控制)。
4. 他们已知的缺陷(论文附录 J):chunk 边界拼接的咔哒/能量阶跃、偶发
   过早写出;本次实跑另见解码残留漏出(`<|endoftext|>Human`、system prompt 句)。
   自然度他们自报 MOS 4.04(m5)/3.66(m2)vs LiveInterpret 4.22(N=15,
   作者内部评分)。人耳判断见 `~/Downloads/simuls2st_samples/` 的对听文件。

## 文件

- `simuls2st_s2tt_m{2,3}.instances.log`、`simuls2st_s2st_m2.instances.log`:
  他们三条 SimulEval run 的逐 talk 记录(音频不入库;hyper01
  `/data04/jaxan/ext_s2st/out/`,hyper00 `/data02/jaxan/ext_score/`)。
- `ours_v8_timeline.instances.log`:我方 v8 按发射时刻渲染的 intervals。
- `asr_bleu_*.json`:各口径的 SEGALE/BLEU summary(`*_whole30` 为主口径)。
- `simuls2st_run_all.sh`、`w1_ext.py`、`render_cascade.py`、`make_stereo.py`。
