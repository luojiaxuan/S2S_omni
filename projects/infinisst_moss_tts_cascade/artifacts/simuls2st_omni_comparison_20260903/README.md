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
- **S2ST ASR-BLEU**:他们的 SimulEval speech 输出(`wavs/<i>_pred.wav`,
  零抖动 FIFO 渲染)按 `intervals` 切成 turn → 与 A′/B′ 完全相同的链
  (逐 turn Qwen3-ASR-1.7B → "。"拼接 → SEGALE d0041438 → SacreBLEU zh),
  在 hyper00 上打分(`w1_ext.py` 是 w1_ap 的外部输入适配)。因他们的
  piece 平均 1.9 s、常在短语中间切开,另加 `--merge-window-s 12`(相邻
  piece 合并到 ≥12 s 再 ASR)协议,两套系统各跑两种协议。
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

| 系统 | 逐 piece/turn 协议 | 12 s 窗口协议 |
| --- | ---: | ---: |
| SimulS2ST-Omni m2 | 31.58(327 段,null 11;逐 talk 31.0/30.3/33.2) | **41.73**(256 段,null 5) |
| ours B′ = phrase InfiniSST + v8 TTS,1× | 33.28(318 段,null 3;逐 talk 32.8/36.5/30.8;经时间线渲染重打 33.17) | **35.64**(283 段,null 3) |
| ours A′ = 原版 InfiniSST + baseline 配对 TTS,1× | 19.54 | — |

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
2. **语音译文质量(ASR-BLEU)取决于协议**:逐 piece 协议下他们 31.6 < 我们
   33.2,但那是他们 1.9 s 的 piece 被 ASR 在短语中间切碎的惩罚(我们 turn
   平均 4 s);合并到 ≥12 s 窗口后他们 41.7 > 我们 35.6,**领先约 6 分**。
   窗口协议更接近"听众听到的内容",应视为主结论;逐 piece 协议对 piece
   粒度敏感,不宜跨系统比。
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
- `asr_bleu_{simuls2st_m2_piece,simuls2st_omni_m2_w12,ours_v8_piece,ours_v8_w12}.json`:四格的 SEGALE/BLEU summary。
- `simuls2st_run_all.sh`、`w1_ext.py`、`render_cascade.py`、`make_stereo.py`。
