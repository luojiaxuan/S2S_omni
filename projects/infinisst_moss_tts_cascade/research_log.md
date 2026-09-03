# Research Log

## 2026-08-30 phrase-policy 匹配 TTS SFT

- **假设**：旧 v7 TTS 没有见过 InfiniSST phrase-policy 的 text/turn 分布，1.5× BLEU 退化主要来自这一分布偏移；用实际 phrase trajectory 重训 TTS 后，跨语速 BLEU 应恢复。
- **症状**：旧 v7 TTS 接 phrase 输出时，BLEU 为 36.80→31.34，1.5× 下降 5.46；speaker 跳变已由连续 codec decoder context 独立修复。
- **改动**：从相同 base model 训练 v8，训练集为 36,529 行 v6 base、6,385 行 phrase full、3,839 行 phrase mid-start；1 epoch，global batch 15，seed 42。
- **结果**：匹配训练后的 BLEU 为 34.09→29.61，1.5× 仍下降 4.48。加速档文字增加 3.06%，生成音频增加 4.80%，null alignment 从 1.50% 升至 7.14%。
- **Keep/Drop**：keep 连续 codec decoder context；drop“旧 TTS 分布偏移足以解释跨语速退化”的假设。phrase policy 暂不作为跨语速默认配置；级联指标不能继续单独定位剩余问题属于 InfiniSST 还是 TTS。
- **证据**：`artifacts/phrase_matched_tts_eval_20260830/summary.json`；合同为 `configs/phrase_matched_tts_v8_20260830.json`。

## 2026-08-30 v8 加速档掉分归因：TTS 失控生成，不是 ASR 也不是文本

- **假设**：1.5× 输入 delta 更长，TTS 输入信息更多，BLEU 不应下降；下降必有单独可定位的机制。
- **现象**：逐 turn 体检 879/878 个 turn。两档正常发音速率相同（sec/char p50 均 0.231）；失控 turn（>0.6 秒/字）从 1× 的 30/879（3.4%）升到 1.5× 的 49/878（5.6%），失控音频占总时长从 10.6% 升到 13.6%；ASR CER 长度分层无档间差；InfiniSST 文本层 BLEU 两档持平。
- **判断**（已证实）：掉分来源是 TTS 对部分 turn 的失控超额生成把邻近对齐冲乱，属于生成稳定性问题而非分布偏移或 ASR 噪声。
- **去留**：v8 数据配比保留；后续策略改动都要带"失控率"这个体检指标。
- **证据**：`artifacts/v8_speed_drop_diagnosis_20260830/diagnosis.json`（脚本 `scripts/diagnose_v8_speed_drop.py`）。

## 2026-08-30 标点门控无因果价值：配对消融 + 紧邻检验

- **假设**：训练数据过滤条件"累积串以句内标点结尾"对 TTS 质量有因果贡献（若无则可去掉以降低 latency）。
- **现象**：长度配对后带标点 write 的表观优势，在"紧跟困难片段之后"的检验里消失——困难内容紧邻的带标点 write 一样差，说明相关性来自内容难度混淆。
- **判断**（已证实，用户先行提出并被检验支持）：落点/标点不影响 TTS 质量；"一句很长的话没有落点质量不一定差"。
- **去留**：drop 标点门控；v9 起写出条件不含标点要求。
- **证据**：`artifacts/punct_gate_matched_ablation_20260830/matched.json`（脚本 `scripts/ablate_punct_gate_matched.py`）。

## 2026-08-31 v9 min_words 写出策略：设计、外审、训练与 1× 崩溃

- **假设**：去掉句边界与标点门控，multiplier∈{1,2,3,4} 随机采样象征 chunk size，仅保留 min_words=2 的实词门槛，可以在不牺牲流式性的前提下提升 delta 质量。
- **改动**：InfiniSST collator 加 `_minwords_redistribute`（aries 容器 `train/dataset.py`），LoRA 续训得 stage2_phrase_v9（权重位移 12.4%）；配套 TTS 数据 `scripts/build_moss_minwords_prepared.py`。发射前过 ChatGPT 外审（`docs/reviews/prior_judgment_phrase_policy_v9_20260830.md` 与 `chatgpt_review_phrase_policy_v9_20260830.md`）。
- **现象**：1× 文本 BLEU 崩到 36.5（对照 54.1），pred 超长 56–96%，全是幻觉凑数；1.5× 正常（47.2）。机制：min_words=2 让训练监督里"空目标 hold"几乎消失，模型丧失合法 hold，在 1× 低密度 chunk 上被迫编内容。
- **判断**（已证实）：delta 不能碎写——这是"长 delta 更稳"论点的训练侧证据；min_words=2 的失败是设计层结果而非 bug。
- **去留**：drop v9 写出策略；min_words 若再 sweep 必须保留足量 hold 监督。
- **证据**：aries `/mnt/gemini/data2/jiaxuanluo/runs/infer_phrase/v9m{1-4}{,s150}/`；数据统计 `artifacts/v9_minwords_data_stats.json`。

## 2026-08-31 纯配对 TTS 微调音频塌缩：全参与 LoRA r8 双证

- **假设**（用户定义的配对微调）：phrase InfiniSST 应配"只用 phrase 轨迹微调"的 TTS，baseline InfiniSST 配"只用 baseline 轨迹微调"的 TTS；v6 base 数据不该掺入。
- **现象**：纯配对训练后合成音频大面积塌缩——全参：phrase 侧每 talk 仅 187s（应有约 620s），baseline 侧 462s；LoRA r8 复验：baseline 缺 25–32%，phrase 缺 66–70%（BL 255s、AL 436s）。塌缩幅度与更新强度（全参 vs r8）无关。
- **判断**（已证实）：塌缩源于数据构成而非训练强度；36,529 行 v6 base 在 v8 配方里起抗遗忘正则作用，v8 的混训设计被平反。
- **改动**：改为 A′ = v6 base + baseline 轨迹混训（对称于 v8 = v6 base + phrase 轨迹混训），tilde `~/sglang-omni-tts/outputs/model_v10base/`。
- **去留**：drop 纯配对路线；配对适应必须以 base 数据打底。
- **证据**：tilde `~/sglang-omni-tts/eval/output/{BL_,AL_,Ap_}*` 合成时长统计；训练数据 `~/sglang-omni-tts/data/train_{phraseonly,baselineonly,v10base}.jsonl`。

## 2026-09-01 A′ vs B′ 终局对照：phrase 线大幅胜出，重训没有白搞

- **假设**：若 A′（原版 InfiniSST + baseline 配对混训 TTS）不优于 B′（phrase InfiniSST + phrase 配对混训 TTS，即 v8），则没有理由放弃 phrase 线退回原版 InfiniSST。
- **现象**：同一 {110,117,268} 三 talk 1× 子集、同一管道（逐 turn Qwen3-ASR → SEGALE d0041438 → SacreBLEU tokenize=zh）、同一脚本从两臂 xcomet_input.jsonl 计算：**A′ = 19.54**（401 段，SEGALE null 85/401 且全部 over-translation）vs **B′ = 33.28**（318 段，null 3）。A′ 段数多出 80+ 本身就是失控超额生成切出来的。失控率跟输入 delta 长度走：A′ 1× 失控 2/5 talk、1.5× 3/5；v8 两档均 0/5。
- **判断**（已证实）：配对条件对齐后 phrase 线 +13.7 BLEU；A′ 的死因与 v8 加速档掉分同机制（TTS 失控生成），短 delta 使失控大增。"长 text delta 质量更好"的项目前提在级联端到端口径上成立。
- **去留**：keep phrase InfiniSST（v8 组合为现役最优）；drop"退回原版 InfiniSST"的疑虑。剩余主问题回到 v8 自身的 1.5× 失控率。
- **打分链修复记录**：w3 优先读 run_config.json 绝对路径而非 `--dataset-root`（改 ap0 的 config 指向 3-talk 子集 d0_3t 解决）；w4 缺 soxr（pip --target 补装时误带入 numpy 2.5.2 覆盖 1.26.4，致 accelerate 循环导入假象，回装 numpy==1.26.4 解决）；w4 还需 `env/site/bin` 进 PATH（vecalign 可执行）。
- **证据**：`artifacts/paired_finetune_final_ab_20260901/`（README、两臂 BLEU 复算 json、A′ 逐段对齐 jsonl）；hyper00 run `20260830-200824-401864000` `result/ap0/`。

## 2026-09-01 收尾与删除记录

- hyper00 打分容器 `sglang-omni-jaxan-score` 已删、map 记录已清；打分产物已拉回本仓 `artifacts/paired_finetune_final_ab_20260901/`，原始件仍在 `/data02/jaxan/`。
- tilde `~/sglang-omni-tts/outputs` 从 ~510G 精简到 14G。已删：`model/`（v9 TTS 全部输出，正本 HF `gavinlaw/moss-tts-realtime-infinisst-en-zh-v9-minwords@b4a42b8e`，删前 `repo_info` 验证 15 files 可达）；`model_phraseonly/`、`model_baselineonly/`（纯配对全参，已判废，塌缩证据与复现配方见上一条与 `data/train_{phraseonly,baselineonly}.jsonl`+sft.py）；`model_v10base/checkpoint-step-*`（16×17G 优化器中间态，训练已完成）。保留：`model_v10base/checkpoint-epoch-0`（A′ 证物，仅本地未上 HF）、两个 `*_lora8`（塌缩双证证物）。
- aries 容器 `infinisst-phrase-jaxan-1` KEEP：phrase 线胜出，推理环境与 v9m* 推理产物待下一步方向裁决后再清。
- taurus 占位作业 48285 状态未知：taurus 直连超时且 aries 侧 no route to host，疑似主机下线，待恢复后确认。
- hyper00 上另见 `sglang-omni-jaxan-20260901-011036-265057286` Exited(1) 12h——非本 session 创建，疑属并行任务，未动，仅在此报告。

## 2026-09-03 外部对照:SimulS2ST-Omni(EMNLP 2026)在同一 3-talk 子集上实跑

- **假设**:同类开源系统(Qwen2.5-Omni-3B thinker + 0.4B talker + VoiceBox,约 2k 小时配对数据,只放推理)在我们的 ACL 60/60 子集上,文本与语音质量、实时性各处在什么位置。
- **现象**(hyper01 单卡 H200 实跑其 README 推荐配置;打分走 A′/B′ 同一链,详见 `artifacts/simuls2st_omni_comparison_20260903/`):
  - S2TT(SimulEval 整场 corpus BLEU):m2 54.98、m3 56.37;我方原版 InfiniSST 54.02、phrase v2ep1 51.78。
  - S2ST ASR-BLEU(主口径=整段译文音频一次送 ASR、SEGALE 句级,用户裁定不按 turn 切):**ElevenLabs Scribe v2 45.09 vs 我方 v8 37.66**;Qwen3-ASR(30 s 内部窗)44.15 vs 35.92,两种 ASR 结论一致。ElevenLabs 自 2026-09-04 起为主 ASR(与 Open-LiveTranslate PR #39 一致:12 分钟整段一次请求无截断、CER 更低、带逐词时间戳)。逐 turn 口径(A′/B′ 当时用法)给出 31.58 vs 33.28 的假象,原因是他们 1.9 s 的 piece 被切碎;helper 默认 120 s 窗的整段 ASR 丢 25–30% 字,作废。
  - 音频时间线(FIFO 仿真):他们译文音频占源时长 81–86%,收尾偏移 −0.7/+1.9/+0.7 s,积压 mean ≤1.2 s;我方 v8 1× 三个 talk 音频 105/111/127%,收尾 +46/+89/+204 s,积压 mean 29/42/108 s。
  - 其解码残留:`<|endoftext|>Human`、system prompt 句漏进译文(m3 talk 268);论文附录自认 chunk 边界拼接咔哒。
- **判断**(已证实):文本层同档;语音层落后约 7–8 分(两种 ASR 一致);**实时性差距是量级差**——我方 v8 在 1× 下译文总时长超过源音频,物理上放不完,这是 TTS 语速/压缩控制的问题,不是 InfiniSST 延迟,是级联当前最大的短板。
- **去留**:跨系统 ASR-BLEU 一律用整段一次 ASR 口径,ASR 换为 ElevenLabs Scribe v2(Qwen 30 s 窗为对照),逐 turn 口径退役;A′/B′ 的 19.54/33.28 是逐 turn 口径,与本条不可直接并排;下一步优先解决级联音频超长(目标译文时长 ≤ 源时长),再谈质量。
- **收尾**:hyper01 `sglang-omni-jaxan-2` 已删、map 已清(`/data04/jaxan/ext_s2st` 24 G 留至 2026-09-10,含可重建的模型/venv 与三条 run 输出);hyper00 `sglang-omni-jaxan-2` 打分完即删。对听样本在本机 `~/Downloads/simuls2st_samples/`。
