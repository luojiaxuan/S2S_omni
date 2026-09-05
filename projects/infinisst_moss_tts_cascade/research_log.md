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

## 2026-09-03 用 Open-LiveTranslate 官方打分栈复测两套系统(PR #40)

- **背景与假设**:上一条的 SimulS2ST-Omni 对照走的是本仓自建打分链(整段 ASR → SEGALE d0041438 → 句级 SacreBLEU)。同事的问题是"是用现在 openlivetranslate repo 测的吗",而实验室主仓 `LeiLiLab/Open-LiveTranslate` 有一套已固化的官方打分栈(ElevenLabs Scribe v2 ASR → SEGALE + penalize-v1 skip policy → LongYAAL/Ending-Offset/BLEU → XCOMET-XL,配不可变 `generation_config.json` 指纹)。假设:两套链会给出同向结论;差距的量级由官方口径说了算。
- **做法**:在 OLT 仓开分支 `feat/simuls2st-omni-comparison`,新增 **timeline 打分入口**——不复现外部系统,只要求它交出"贴到源时钟上的渲染音频 + 每段的播放起止 + 它自己的文本",就能进官方栈。`build_run_from_timeline.py` 写出与商用 baseline 相同的 arrival-only 时序记录(`delays == elapsed ==` 播放起点,CU 定义等于 CA),`score_timeline.sh` 生成不可变 identity 指纹后交给未改动的 `run_s2s_score.sbatch`。两个 backend:`simuls2st-omni`、`infinisst-moss-cascade`。
- **现象**(hyper00 容器 `sglang-omni-jaxan-3`,同 3 talk {110,117,268} dev 1×、en→zh、ElevenLabs Scribe v2、XCOMET-XL 开、CU=CA、两 run 的 `document_fingerprint` 相同 `8437ac49…`):

  | 系统 | chunk | BLEU | XCOMET-XL | LongYAAL | Ending Offset | 段数 | skip(欠/超译) |
  |---|---|---:|---:|---:|---:|---:|---:|
  | SimulS2ST-Omni(latency multiplier 2) | 2.0 s | 40.65 | 0.658 | 3,658 ms | 3,233 ms | 253 | 0 / 0 |
  | InfiniSST phrase v2ep1 + MOSS-TTS v8 | 1.92 s | 34.08 | 0.646 | 53,186 ms | 63,525 ms | 267 | 0 / 2 |

- **判断**(已证实):(1)质量上我们落后 6.6 BLEU,XCOMET 同向(−0.012),与自建链(45.09 vs 37.66)结论一致、绝对值不同(两条链的重分段与 skip 惩罚不同,不可并排);(2)**实时性差距被官方指标钉死为量级差**:我们的 Ending Offset 63.5 秒——最后一个词比演讲结束晚一分多钟落地;LongYAAL 53.2 秒。原因不是策略而是渲染:合成语音比源音频更长,播放队列永不排空,每段都被前面的积压罚一次。skip 惩罚只占 0.23 BLEU / 0.37 s,不构成解释。
- **去留**:级联的下一步优先级确定为"译文音频时长 ≤ 源时长",不是继续调质量;跨系统对照今后一律走 OLT 官方栈(本仓自建链退为内部快速迭代用)。
- **口径变更记录**:自建链的 45.09/37.66 与本条的 40.65/34.08 **口径不同不可并排**——前者句级 SEGALE 无 skip 惩罚,后者 penalize-v1 且重分段段数不同(253/267)。
- **证据**:PR https://github.com/LeiLiLab/Open-LiveTranslate/pull/40;run 目录 `s2st_{simuls2st-omni_dev_2000ms,infinisst-moss-cascade_dev_1920ms}_local-final`,generation 指纹 `b1fb07e5…` / `b2d3e7c3…`;文档见该仓 `eval/README.md` §7 与 §7a。
- **过程中的错误**:(1)灌进容器的 `dev.yaml` 保留了 ACL 发布包的路径前缀,官方链按裸文件名严格匹配,导致 `hyp.jsonl` 写出 0 行、连跑 9 次 dry run 才定位——已在该仓 `SETUP.md` §4 记录;(2)两次用 `pkill -f <模式>` 时模式串出现在自己的 shell 命令行里,把自己杀掉;(3)首轮正式跑时容器仓库还是 main + 未跟踪改动,`generation_repo_tree_dirty=true`,重做为干净的分支提交后再跑。

## 2026-09-03 延迟归因:63.5 秒收尾偏移全部来自 TTS 发声速度,与 thinker 无关

- **假设**:官方栈量到的 Ending Offset 63.5 s / LongYAAL 53.2 s,可能来自(a)thinker 出词太晚、(b)译文太啰嗦、(c)每个 turn 头尾静音填充、(d)turn 内停顿过多、(e)TTS 逐字发声本身太慢、(f)少数失控生成的超长 turn。逐项证伪。
- **做法**:直接读两套系统的 timeline(`instances.log` 的 arrival 与 playout intervals)与渲染音频,10 ms 帧、int16 峰值 ≤200 记为静音,统计每 piece 的前导/尾随/内部静音与发声时长;字数用各自系统自己的文本输出(非 ASR 结果)。探针脚本 `scripts/latency_probe/`。
- **现象**:

  | 口径 | 我方 v8(110/117/268) | SimulS2ST-Omni |
  |---|---|---|
  | 最后一段译文的产出时刻 − 源结束 | **−2.2 / 0.0 / −5.9 s** | −1.0 / −1.0 / −1.4 s |
  | 收尾偏移(播放结束 − 源结束) | 89.0 / 203.6 / 46.0 s | 1.9 / 0.7 / −0.7 s |
  | 平均积压(播放起点 − 产出时刻) | 41.8 / 108.1 / 29.4 s | 0.7 / 0.4 / 1.2 s |
  | 译文字数 ÷ 人工参考字数 | 1.02 / 1.00 / 0.98 | 0.96 / 0.96 / 0.92 |
  | 音频总长 ÷ 源时长 | 1.11 / 1.27 / 1.05 | 0.86 / 0.81 / 0.86 |
  | 静音占音频比 | **12.5 / 13.1 / 13.1 %** | 18.0 / 18.7 / 26.7 % |
  | 每 piece 前导静音 | **44 / 47 / 59 ms** | 92 / 96 / 136 ms |
  | **纯发声段的字/秒** | **4.77 / 4.43 / 5.10** | 6.18 / 7.12 / 6.93 |

- **判断**(已证实):(a)(b)(c)(d)全部证伪——thinker 的最后一段译文在演讲结束**之前**就产出了,字数就是人工参考的 1.0 倍,静音比对方**更少**,前导填充只有对方的一半。差距 100% 在(e):**同样的字数,我们的 TTS 要多花 30–45% 的时间说完**。(f)是加重项不是主因:每篇有 4–5 个 15 秒量级的超长 turn(v8 已知的失控生成),合计占音频 8–9%。
- **量化目标**:把全部 piece 的音频等比压缩 k 倍并重放 FIFO,使收尾偏移 ≤5 s 需要 **k = 1.19 / 1.33 / 1.12(均值 1.21)**,对应约 5.0 字/秒;人工参考译文要跟上源音频需要 4.54 / 4.90 / 4.76 字/秒,对方实测 5.07–5.79 字/秒(含静音口径)。即**我们差约 20%**。仅靠裁剪 >0.3 s 的静音段最多回收 11–12 s/篇,单独不够。
- **被证伪的顺手假设**:级联 TTS 用英文源音频做 ref_audio 克隆音色,怀疑语速也被克隆。三篇不支持——英文语速最快的 117(2.83 词/秒)对应我们**最慢**的中文输出(4.43 字/秒),方向相反。样本仅 3 篇,记录待查而非结论。
- **下一步(按成本排序,未执行)**:(1)固定文本、变换 speaker prompt 实测 v8 的字/秒是否受 prompt 控制;(2)后处理时域压缩 1.2×(WSOLA/sox tempo),走同一 OLT 栈量 BLEU/XCOMET 代价——这是保底方案,必然消除积压;(3)查 v8 训练语料目标音频的字/秒,若语料本身偏慢则考虑按 1.2× 重建或加语速增广;(4)文本侧压缩(同传本就会压缩 10–20%),我们现在是参考译文的 1.0 倍。
- **证据**:`artifacts/olt_official_scoring_20260903/`(官方栈指标);timeline 原件在 hyper00 `/data04/jaxan/olt_build/{ours,theirs}_ext`(留至 2026-09-10)。

## 2026-09-03 归因修正:不是"整体语速慢",是 4–10% 的 turn 失控空转

- **纠正对象**:同日上一条写的"同样的字数我们的 TTS 要多花 30–45% 时间,需要整体提速 1.2 倍"。**该结论错误**,已作废。错因:我用的是整篇聚合口径(总字数 ÷ 总发声秒数),这个平均值被少数极端 turn 拉低,看起来像均匀变慢。
- **如何发现**:切听力样本时看到 talk 110 最长的 turn 是 17 个字念了 14.96 秒(1.14 字/秒),不像"慢",像空转,于是改用逐 turn 口径重算。
- **修正后的现象**(逐 turn,字数取 InfiniSST 该段原文,时长取渲染 interval):

  | talk | turn 数 | 中位字/秒 | p10 | p90 | <2.5 字/秒的 turn 数 | 这些 turn 占音频 |
  |---|---:|---:|---:|---:|---:|---:|
  | 110 | 194 | 4.81 | 3.77 | 5.47 | 14 | 153 s = 20% |
  | 117 | 191 | 4.58 | 2.53 | 5.36 | 19 | 225 s = 24% |
  | 268 | 191 | 4.76 | 4.17 | 5.65 | 7 | 95 s = 12% |

  正常 turn 的发声口径是 5.19–5.49 字/秒,总口径 4.55–4.83,而追平源时长需要 4.54/4.90/4.76 —— **正常 turn 基本达标**(117 略欠)。坏 turn 是 1.58–1.63 字/秒,且 **87–88% 的帧是有声的**,不是静音:模型在念不存在的内容,不是说得慢。最坏的例子 0.48 字/秒(117 第 18 段,7 字 14.72 秒)。
- **反事实重放**(只把 <2.5 字/秒的 turn 时长换成"该段字数 ÷ 5 字/秒",其余不动,重跑 FIFO):收尾偏移 **110: 89→8 s,117: 204→52 s,268: 46→24 s**;再叠加 3% 的静音裁剪为 6/29/18 s。即**只修失控 turn 就能消掉 60–90% 的积压**,不需要整体提速。
- **失控是成串出现的**:连续段占坏 turn 的 79%/47%/29%(如 117 的第 6-8 段连着三个,合计 39.8 秒音频只念 82 字)。级联的 TTS 是 `tts_codec_context: continuous_per_row`(跨 turn 连续 codec 上下文),成串的形态与"上下文被污染后连累后续 turn"一致——**待验证的假说,不是结论**。
- **对质量的连带影响**(未量化):12–24% 的音频是与文本对不上的声音,ASR 转出来必然是垃圾,官方栈的 BLEU 34.08 里有一部分是被这个吃掉的,不全是翻译质量。
- **修正后的下一步**:(1)先定位失控的触发条件——查是否与 codec 上下文长度、上一 turn 是否失控、文本长度或标点相关;(2)最小修复候选:每 turn 重置 codec 上下文 / 加长度守卫(字数×最慢合理秒数为上限,超时截断重合成);(3)之后再看 117 的正常 turn 是否仍需小幅提速。原先排在第一位的"后处理时域压缩 1.2×"降级为兜底。
- **证据**:探针 `scripts/latency_probe/{per_turn_rate,slow_turn_shape,counterfactual}.py`;听力样本 `~/Downloads/cascade_samples/`(窗口 D=连续三个坏 turn,E=同篇正常段落,同模型同音色);逐 turn 原文 hyper00 `/data04/jaxan/S2S_omni_runs/moss_tts_infinisst_v2_20260804/acl_bench/tts_rows/talk*.phrv2e1.swrow.jsonl`(与打分 timeline 逐段对齐,194/191/191 段、delay 完全一致)。

## 2026-09-03 失控 turn 的根因:守卫预算过松 + 漏检的坏 turn 污染滑动窗口

- **假设**:上一条定位到 4–10% 的 turn 失控空转,但没说清为什么已有的 `runaway` 守卫没拦住,以及为什么失控成串出现。
- **已确认的机制**:
  1. **守卫预算过松**。`scripts/moss_multiturn_infer.py:425` 的 `budget_frames = max(floor_s, 字数 × max_seconds_per_char) × 12.5`;该次合成实跑 `--min-runaway-floor-s 15`、`--max-seconds-per-char` 取默认 `0.6`。即"每字 0.6 秒(1.67 字/秒)且任何 turn 无条件给 15 秒",而健康 turn 实测约 5 字/秒、中位 16–17 字(健康时长 3.3 秒)。**576 个 turn 里 40 个坏,守卫只触发 7 次**(110: 4/14,117: 1/19,268: 2/7),触发的都是撞到 188 帧上限的极端个案。
  2. **漏检即污染**。守卫触发时会截断并清空滑动窗口(detox),被污染的 codes 不进窗口;漏检的坏 turn 其 codes 照常进入 11 帧滑窗。条件概率:**P(下一个坏 | 上一个坏且漏检)=12/33=36%,P(下一个坏 | 上一个坏且已触发)=0/7=0%,P(下一个坏 | 上一个正常)=27/533=5%**。detox 有效,只是几乎从不运行。
- **被排除的解释**:文本特征不预测失控——坏/正常 turn 字数中位 18 vs 17,以标点结尾 72% vs 85%,篇内位置各档 2.6%–10.4% 无梯度。
- **该次合成的其他守卫状态**:`--min-frames-per-char` 默认 0(过短 turn 重生成守卫**未开**),`--loop-detect` 默认 `none`(循环检测**未开**)。当时只有过松的预算守卫在工作。
- **收紧预算的预期收益**(FIFO 重放,收尾偏移秒):截断口径 110: 89→33,117: 204→130,268: 46→24;再计入"不再被污染"为 14/106/24;若改为重合成(坏 turn 按 5 字/秒)为 8/52/24。
- **口径警告**:候选预算 `max(5 s, 字数/2.5)` 与"坏 turn"的标注判据(<2.5 字/秒)是同一统计量,"覆盖 40/40"属循环,不作为独立验证。有信息的是**误伤 0**:576 个 turn 中没有任何健康 turn 同时"超过 5 秒"且"慢于 2.5 字/秒",两分布可分。样本 3 篇 talk。
- **模型身份核对**:合成用的本地 checkpoint `.../20260830-200824-401864000/outputs/model/checkpoint-epoch-0/model.safetensors` sha256 = `074964929bce38b9…`,与 HF `gavinlaw/moss-tts-realtime-infinisst-en-zh-v8-phrase@521e09fa` 的同名文件 LFS sha256 一致。PR #40 的 identity 无误。
- **待确认(实验在跑)**:同 checkpoint、同 seed、同滑窗,只把预算从 `floor 15 / 0.6` 改成 `floor 5 / 0.4`,重跑 talk110 与 117,直接量坏 turn 数、总时长与收尾偏移。hyper00 容器 `sglang-omni-jaxan-2`(GPU 6,2 条件 × 2 分片)。该任务 GPU 利用率约 2%(12 秒采样,峰值 18%),瓶颈在 CPU 侧 codec 解码,卡数已是最低的 1 张。
- **证据**:`scripts/latency_probe/{poison_test,features,replay_fix,diag_turns}.py`;逐 turn 合成元数据 hyper00 `/data02/jaxan/tmp/runs/20260830-200824-401864000/eval/output/c2/talk*.summary.jsonl`(带 `frames` 与 `runaway_skipped`)。

## 2026-09-03 消融确认:收紧失控预算,收尾偏移 talk110 53→10 s、talk117 176→77 s

- **设计**:同 checkpoint(v8@521e09fa 的本地副本)、同 seed 42、同滑窗(11/soft-reset-keep 3/continuous codec context),**只改失控预算**:base = `floor 15 s / 0.6 s每字`(原设置),tight = `floor 5 s / 0.4 s每字`。talk110 与 117 各跑两臂,2 条件 × 2 分片并行。hyper00 容器 `sglang-omni-jaxan-2`,GPU 6。
- **结果**(坏 turn 判据仍为 <2.5 字/秒;收尾偏移按 FIFO 重放同一 InfiniSST 产出时刻):

  | 臂 | talk | 音频总长 | 坏 turn | 守卫报警 | 坏 turn 音频 | 收尾偏移 | 中位字/秒 |
  |---|---|---:|---:|---:|---:|---:|---:|
  | base | 110 | 746.4 s | 4 | 1 | 55 s | 53 s | 4.53 |
  | tight | 110 | 699.5 s | 2 | 6 | 10 s | **10 s** | 4.81 |
  | base | 117 | 897.0 s | 16 | 2 | 181 s | 176 s | 4.66 |
  | tight | 117 | 797.8 s | 4 | 15 | 18 s | **77 s** | 4.73 |

- **判断**(已证实):收紧预算使守卫报警从 1/2 次升到 6/15 次,坏 turn 音频从 55/181 s 降到 10/18 s,收尾偏移降 81%/56%。方向与机制判断一致。
- **必须记的方差发现**:**base 臂没有复现原始 run**。原始 talk110 是 14 个坏 turn / 782 s / 收尾 89 s,base 复现是 4 / 746 / 53;talk117 原始 19 / 925 / 204,base 16 / 897 / 176。同 seed 同参数,差异来自容器镜像不同(本次 `hongccc/sglang-omni:dev`,torch 2.13)。**因此失控计数不可跨环境比较,只能同环境内 base vs tight 配对比**;本次每格 n=1。
- **未测的代价**:tight 触发 6/15 次截断,被截的 turn 有文本没念完,BLEU 代价未量化。**这是下一个必须做的实验**(走 OLT 官方栈打分 tight 臂)。更优的修法是把截断改为"先重合成一次再截断"(仿照已有的 `short_regen` 分支),这样不丢文本;该改动尚未实现。
- **改动**(已提交):`scripts/moss_multiturn_infer.py` 的默认 `--max-seconds-per-char` 0.6→0.4、`--min-runaway-floor-s` 8.0→5.0;四处硬编码 15 的启动器(`configs/codec_decoder_context_ab_talk110.json`、`run_base_queue.sh`、`run_eval_queue.sh`、`run_codec_context_phrase_compare.py`)一并改为 5。语料合成侧的 `synth_moss_rows_{inprocess,batched}.py`、`generate_moss_realtime_long_targets.py`、`gen_moss_self_history.py` **不动**——那是训练数据生成的护栏,收紧会改变语料构成,属另一条决策。
- **决策日志**(替用户做的默认决定):问题=要不要在没量化 BLEU 代价前就把预算收紧为默认;默认答案=收紧;理由=守卫的存在目的就是拦这个,当前值比健康语速松 3 倍且 15 秒地板让它对中位 turn(16–17 字)形同虚设,实测收尾偏移降 81%/56%;回滚=把 5.0/0.4 改回 15/0.6,单 commit 可 revert。**外审未做**(用户在线交互中,直接汇报由其否决更快)。
- **证据**:`scripts/latency_probe/ablate.sh` 与 `ablate_compare.py`;产物 hyper00 `/data02/jaxan/tts_guard_ablation/`(容器删除后仍在宿主盘)。

## 2026-09-04 换 TTS 权重实测:owaski/moss-tts-realtime-delta-zh-125k 接我们的 phrase 输出,分数不动

- **背景**:同事的 delta-finetune checkpoint(`owaski/moss-tts-realtime-delta-zh-125k@fc2d094d`,基座同为 `OpenMOSS-Team/MOSS-TTS-Realtime`,与我们 v8 同架构、safetensors 同大小 4,663,931,664 B,可直接换权重)。其 README 报告配**它自己的** phrase thinker(`owaski/infinisst-thinker-phrase-zh`)在 ACL 60/60 dev、1920 ms chunk 下:BLEU 40.64、XCOMET-XL 0.723、收尾偏移 CU 4018 / CA 5942 ms。用户要求接我们的 InfiniSST phrase 输出实测。
- **做法**:唯一变量是 TTS 权重。同 InfiniSST phrase 行(phrv2e1 swrow)、同固定音色 ref、同滑窗(11 / soft-reset-keep 3 / continuous codec context)、同 seed 42、同失控预算(15 s / 0.6 s每字)、同 harness(`w0.py` 即 `moss_multiturn_infer.py` 快照)。渲染沿用零抖动 FIFO;**时间线构建脚本先用原始 v8 产物验证过,重建出的 intervals 与已打分的 `ours_ext` 逐字节一致**。打分走 OLT 官方栈(在 hyper00 重建:三 venv + 补丁 + SEGALE/vecalign + XCOMET-XL),ElevenLabs Scribe v2,与 PR #40 同配方。
- **结果**(同三篇 talk,`document_fingerprint` 同为 `8437ac49…`,CA 列,CU=CA):

  | TTS | BLEU | XCOMET-XL | LongYAAL | 收尾偏移 | 段数 | 超译罚 |
  |---|---:|---:|---:|---:|---:|---:|
  | 我们 v8 | 34.08 | 0.646 | 53,186 ms | 63,525 ms | 267 | 2 |
  | owaski delta-zh-125k | 34.58 | 0.652 | 53,401 ms | 61,959 ms | 269 | 2 |

  时间线层(音频/源时长、坏 turn、收尾):delta 1.11/1.26/1.08、5/17/7 个坏 turn、86/196/70 s;我们 v8 原始 1.11/1.27/1.05、14/19/7、89/204/46 s。中位语速 delta 4.41–4.79 vs v8 4.53–4.75 字/秒。
- **判断**:**直接换权重进我们现有管线,分数不动**。BLEU +0.50、XCOMET +0.006、LongYAAL 慢 0.2 s、收尾偏移快 1.6 s——而我们 v8 自己在 talk117 上三次采样的收尾偏移是 176/204/264 s,1.6 s 远在噪声内。它也**没有说得更快**,所以音频超长这个根本问题换它并不自动解决。
- **两个已知的接法不匹配(结论的限制)**:(1)**粒度**——它按 delta 训练(每 1.92 s chunk 一个 turn、2–6 字碎片),我们 phrase 每 3.84 s 吐约 16 字完整小句,是其训练分布的三倍长;(2)**harness**——其官方评测走 OLT `moss_tts_delta_server.py` 配 `--codec-context conversation`,我走的是我们的滑窗多轮脚本。因此本条只能证否"换权重即可",不能证否该模型。
- **在跑**:粒度对照——同 checkpoint 喂 `chunk192` swrow(每 1.92 s 约 9 字,三篇约 1080 段),区分"模型对我们无用"与"我喂错了"。
- **证据**:run 目录 hyper00 `/data04/jaxan/olt_build/results3/s2st_infinisst-moss-cascade_dev_1920ms_delta-final`;合成产物 `/data02/jaxan/delta_tts/`;脚本 `scripts/latency_probe/{synth_delta.sh,build_timeline.py,score_delta.sh,delta_quicklook.sh}`;容器 `sglang-omni-jaxan-2`。

## 2026-09-04 粒度对照:按它的训练粒度喂,反而更差

- **假设**:上一条"换权重分数不动"可能是因为喂错粒度——它按 delta 训练(每 1.92 s chunk 一个 turn、2–6 字),我们 phrase 每 3.84 s 约 16 字。若粒度是主因,按 `chunk192` 喂应当变好。
- **现象**(同 checkpoint、同 harness、同参数,输入换成 chunk192 swrow):

  | talk | turn 数 | 中位 turn 字数 | 音频/源 | 坏 turn | 收尾偏移 | 中位字/秒 |
  |---|---:|---:|---:|---:|---:|---:|
  | 110 | 358 | 8 | 1.11 | 11 | 79 s | 4.08 |
  | 117 | 367 | 9 | 1.30 | 17 | 220 s | 3.80 |
  | 268 | 371 | 9 | 1.28 | 29 | 214 s | 4.17 |

  对照同 checkpoint 的 phrase 粒度:1.11/1.26/1.08、86/196/70 s、4.41–4.79 字/秒。
- **判断**:**粒度不是解释,方向还相反**。turn 越短越碎,单位字数的音频越长(中位语速从 4.41–4.79 掉到 3.80–4.17),总时长与积压都变差。机制上讲得通:每个 turn 都有起音与韵律重启的固定开销,turn 数从约 190 涨到约 365 就把开销翻倍。这也与项目早先"phrase 优于 baseline"的结论同向。
- **限制**:chunk192 的文本来自原版 InfiniSST 策略而非 phrase,所以这一臂同时改了粒度与文本,只能作方向性诊断,不能作严格的粒度单因素实验;也因为文本不同,没有送打分(BLEU 与 phrase 两臂不可比,且要额外 ASR 费用)。
- **剩下的唯一未验因素**:serving 路径。它的官方评测走 OLT `moss_tts_delta_server.py` 配 `--codec-context conversation`,我们全程走自家滑窗多轮脚本。要判断该模型对我们是否真有价值,决定性实验是用它的 server + 它的 thinker 端到端复现其公布行(BLEU 40.64 / 收尾 4.0–5.9 s),再把 thinker 换成我们的。**未做。**
- **证据**:`/data02/jaxan/delta_tts/gran*`;脚本 `scripts/latency_probe/{synth_gran.sh,gran_look.sh}`。

## 2026-09-04 serving 路径复现成功:同一 TTS 权重,收尾偏移 62 s → 5 s

- **纠正一处认知错误**:我先前按仓库 2026-08-28 交接文档认定我们的 InfiniSST 是 `w2v2_qwen25`(wav2vec2 + Qwen2.5 + 380 MB stage-2 LoRA)。**该线已废弃**。实际 InfiniSST 就是 Qwen3-Omni-30B:三个 `gavinlaw/infinisst-*` HF 仓库均为 `Qwen3OmniMoeForConditionalGeneration`、70.5 GB。不带 retriever 的 baseline 是 **`gavinlaw/infinisst-no-tmsft-origin-bsz4-zh`**(Qwen3-Omni-30B-A3B-Instruct + LoRA r32/a32,数据 `manifests_rag/train_s_zh_origin.jsonl`),本仓 README 第 259 行早已写明"InfiniSST baseline(S2T 侧)"指向它,是我没查到。带 retriever 的是 `rasst-speech-llm-zh-cap16-denoise-ttag`。**PR #40 里那条级联行(BLEU 34.08 / 收尾 63.5 s)用的是废弃线的文本产物,数字本身无误但描述的是已不用的系统,待本轮出数后重写。**
- **假设**:同事的 delta TTS 直接换进我们管线分数不动(见前条),剩余未验因素是 serving 路径——他们走 OLT `moss_tts_delta_server.py`(每个 delta 一个 turn,drain 到 audio-EOS 即确认返回,`--codec-context conversation`,`--max-context-positions 600`),我们走自家 `moss_multiturn_infer.py` 滑窗多轮。
- **做法**:hyper01 容器 `sglang-omni-jaxan-3`(GPU 2/3/4/7),完整搭起 OLT cascade:agent/thinker/moss 三个 venv、MOSS-TTS 子模块按 pin commit 克隆并打补丁、vLLM 0.15.1、打过 computation-aware 补丁的 SimulEval。走 OLT 自己的 `run_s2st_eval.sbatch 1.92 1.0 dev moss-delta`,`MAX_DOCS=3` 正好命中 110/117/268,speaker prompt 用 OLT 自带 `assets/spk_prompt/zh_1.wav`。复现臂 = 他们的 thinker + 他们的 TTS。
- **结果**(复现臂,直接从 run 的 instances.log 与渲染 wav 算):

  | 文档 | 源时长 | CU 收尾偏移 | CA 收尾偏移 | 字/秒 |
  |---|---:|---:|---:|---:|
  | 268 | 737.4 s | +4.4 s | +6.8 s | 4.64 |
  | 110 | 703.0 s | −0.8 s | +1.1 s | 4.59 |
  | 117 | 729.0 s | +3.5 s | +6.9 s | 4.98 |

  均值 CU +2.4 s / CA +4.9 s,对照其 README 公布的 CU 4018 ms / CA 5942 ms(5 篇全集)——**同量级同档,复现住**。
- **判断**(已证实):**瓶颈是 serving 路径,不是 TTS 权重**。同一份权重,走它的 delta server 收尾偏移是秒级,走我们的滑窗脚本是 62 秒,差一个量级。语速几乎相同(它 4.59–4.98 vs 我方 v8 4.53–4.75 字/秒),所以差别不在"说得快",而在每个 delta 独立成 turn、立即 drain 并确认,不让上下文无限滚动、也就没有我们那条路上 4–10% 的失控空转。
- **在跑**:换我们的 thinker(`infinisst-no-tmsft-origin-bsz4-zh`)+ 同一 TTS,其余全同,指纹 `12f35ac4`(复现臂 `dc692783`)。两臂生成完统一打分(打分与生成不并行,避免污染 computation-aware 墙钟)。
- **过程中的环境坑**:(1)TTS server 分到的卡被别的租户占满 128 GB 导致 OOM,改为显式 `TTS_GPU` 指向确认空闲的卡;(2)我写的进程标题 shim 替换后又调用自己,无限递归把 thinker 启动打挂——应先捕获原函数;(3)打包时排除 `.git` 导致指纹快照失败,需补传 `.git` 并清掉 macOS 的 `._*` 元数据(否则 untracked 计数被污染);(4)`SLURM_JOB_ID` 被用于取模算端口,必须是数字。
- **共享账号提醒**:hyper01 容器挂载的共享 HF cache 里默认 token 属于同事 **jiapingW**,前几次下载不知情用了它;XCOMET 是 gated 因而静默失败(只下来 72K)才暴露。已按 side-by-side 规矩把 gavinlaw token 放在 `/data04/jaxan/.keys/hf_token_gavinlaw`(0600),此后显式 `HF_TOKEN`,不覆盖他人默认。

## 2026-09-04 phrase gating 上 Omni 线:数据齐了,但标点规则只是他们的 fallback

- **前提问题已解决(推翻我上一条的"需要向 siqiouya 要语料")**:训练轨迹与音频**都在我们自己的盘上**——`/mnt/gemini/data/jiaxuanluo/manifests_rag/train_s_zh_origin.jsonl`(12,500 行,17.8 MB,正是 `infinisst-no-tmsft-origin-bsz4-zh` 的训练集)与 `/mnt/gemini/data/jiaxuanluo/audio_clips_siqi_zh_v2/`(7.3 GB,路径逐个对得上)。taurus/aries 经 NFS 都可读写。我先前只查了 Babel 路径与 jiaxingxu 的目录,漏了本仓 `docs/remote_artifacts.md` 早已写明的"本地明文副本"。
- **数据格式**:ms-swift `messages`,user `<audio>` 与 assistant 交替,每个 assistant 就是该 chunk 的释放文本(不释放为空串);`audios` 是逐 chunk wav。轨迹改写只动 assistant 内容。
- **已实现并验证**:`scripts/phrase_gating/phrase_gate_traj.py`。规则=累积释放,直到缓冲区以短语标点结尾且实字数 ≥ `min_chars`,或攒够 `max_hold` 步。**每行做恒等校验:改写前后 assistant 内容拼接必须逐字相同,不等即拒写**;音频路径重写后抽查存在性。12,500 行 0.78 秒跑完,抽查 200 个音频 0 缺失。
- **超参扫描**(前 3000 行;基线=词对齐,14,811 次释放、中位 11 字、其中 2,079 次不足 4 字):

  | min_chars | max_hold | 释放数 | 占基线 | 中位字数 | <4 字 |
  |---:|---:|---:|---:|---:|---:|
  | 2 | 2 | 9,398 | 0.63 | 18 | 458 |
  | 3 | 3 | 7,589 | 0.51 | 22 | 174 |
  | 4 | 4 | 6,823 | 0.46 | 24 | 81 |
  | 6 | 8 | 5,965 | 0.40 | 29 | 54 |

- **判断**:**纯标点规则在这份语料上是粗代理,达不到目标粒度**。即使最松的 `max_hold=2` 也到中位 18 字,而我们 w2v2 phrase 线实测中位 16、他们 TTS 语料的 turn 中位 13。原因是这份轨迹里标点稀疏,真正起作用的是 `max_hold` 而不是短语边界——换言之我们是在按步数切,不是按短语切。
- **正解**:OLT `data/scripts/s2t/phrase_segment.py`(stage 8)用 LLM(Qwen3.8-27B-FP8 经 vLLM)标短语边界,**删掉标记必须逐字节还原原文**才接受,再把每步累积释放向下取整到最近边界;三条不变量(长度、拼接、顺序)由构造保证。**我们那条标点规则恰好是它的 fallback 路径**(`n_fallback` 计数的那条)。
- **下一步**:写 manifest 适配器(我们的 `messages` 格式 ↔ stage 8 的 trajectory 列格式),在 OLT 上跑 stage 8(只需文本,不需音频,LLM 起在 hyper01),再转回来做 LoRA SFT。按新的分工,适配器写在 OLT 仓库。

## 2026-09-04 自训 phrase-gated thinker(一):数据改写完成,规则与超参定档

- **目标**:今天量出我们的 thinker 比他们的 phrase-gated 版低 1.5 BLEU(CU 36.81 vs 38.32),差距就是 phrase gating。用户裁定**必须自己训**,不用他们的 checkpoint;架构就是现役的 Qwen3-Omni-30B(w2v2 线已废弃)。
- **前提落实**(我一度误判为"缺语料需向 siqiouya 索取",错误,收回):训练轨迹与音频都在我们自己的 gemini NFS 上——`/mnt/gemini/data/jiaxuanluo/manifests_rag/train_s_zh_origin.jsonl`(12,500 行,即原 checkpoint 训练所用)与 `audio_clips_siqi_zh_v2/`(7.3 GB),基座 `/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct` 也在。`docs/remote_artifacts.md` 早写明"本地明文副本",是我没查到。
- **规则**(用户裁定:不用 LLM 判别,就用他们的 fallback 思路,按字数为主):一步释放的条件是 **实字数达到 `release_chars`,或以短语标点结尾且已有 `punct_min` 字**。取消了原先的 `max_hold`——字数封顶本身就限制了持有时长,少一个旋钮。
- **超参 sweep**(3000 行子集;项目规矩:指定超参须报 sweep 而非试一个就定):

  | punct_min | release_chars | 释放数 | 中位 | 均值 | p90 | <4字占比 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 基线(词对齐) | — | 14,811 | 11 | 15.2 | 33 | 14% |
  | 0 | 8 | 11,871 | 16 | 19.0 | 36 | 3.2% |
  | **4** | **8** | 11,711 | **16** | 19.2 | 36 | **1.6%** |
  | 4 | 12 | 10,676 | 18 | 21.1 | 37 | 1.5% |
  | 6 | 16 | 9,698 | 21 | 23.2 | 39 | 1.4% |

  取 `punct_min=4, release_chars=8`:中位 16 字与我们 w2v2 phrase 线实测一致;`punct_min` 从 0 加到 4 使碎片率减半而中位/均值/p90 全不变,是无代价的改进。**注意**:字数上限对中位影响小(8→16 只把中位 16 推到 21),因为多数释放由标点触发,字数上限主要管长尾。
- **产物**:`/mnt/gemini/data/jiaxuanluo/phrase_gating_20260904/train_s_zh_phrase_ours.jsonl`,12,500 行。全量统计:释放 61,194 → 48,572(0.79×),中位 12 → 16 字,<4 字碎片 8,689(14%) → 754(1.6%)。**正确性**:脚本对每行断言 `"".join(改写后) == "".join(改写前)`,不等即拒写,12,500 行全过——文本一字未变,只有释放时刻后移。音频路径已重指本地,抽查 200 条全在。
- **训练配置**(逐项抄原 checkpoint 的 `args.json`,只换数据):LoRA r32/α32/dropout 0.05、`target_modules=all-linear`、`freeze_vit/aligner=True`、LLM 可训、micro 1 × global 4、max_length 2048、1 epoch、lr 1e-4 cosine→1e-5、warmup 5%、weight decay 0.01、clip 1.0、Adam β(0.9,0.95)、bf16、seed 42。
- **决策日志**:问题=原训练走 Megatron(`micro_batch_size`/`lr_warmup_fraction`/`save_interval` 等是 Megatron 参数名),复刻是否也用 Megatron?默认=改用 ms-swift 的 HF 后端 + DeepSpeed ZeRO-3;理由=Megatron 路线依赖 aries 上没有的 Apptainer 镜像,而超参可一一对齐、差异仅在并行实现,且我们比较的是数据改动、两臂同后端即可控;回滚=若显存或吞吐不达标改回 Megatron。外审未做(用户在线,直接汇报更快)。
- **aries 环境八连坑**(全部实测,已写进脚本):根分区 100% 满 → 缓存与临时目录全指 gemini;系统 python 缺 `ensurepip` → 用 conda;conda 被重定位,shebang 与 shell 集成失效 → 用其自带解释器直接驱动;conda OpenSSL legacy provider 报错 → `CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1`;`qwen_omni_utils` 隐式依赖 torchvision;modelscope 走自己的 `MODELSCOPE_CACHE` 绕过 `XDG_CACHE_HOME` 写满盘;pip 默认装 cu130 而驱动是 550.107/CUDA 12.4 → torch 2.6.0+cu124,并把 DeepSpeed 从 0.19.6 降到 0.16.4(算子注册不兼容);**NCCL 在 A6000 间的 PCIe P2P 上死锁**——四卡 all-reduce 无限挂起、GPU 100% 而显存不涨,我第一次误判为"加载慢"白等 67 分钟,用 30 秒最小 all-reduce 探针定位,加 `NCCL_P2P_DISABLE=1` 即通。
- **状态**:冒烟运行中,ZeRO-3 `zero.Init` 生效(38.28B 参数分片,每 rank 约 23 GB / 48 GB),正在加载 15 个权重分片。
- **证据**:`scripts/phrase_gating/`(改写、两次 sweep、数据冒烟、NCCL 探针、环境与训练脚本)。

## 2026-09-04 自训 phrase-gated thinker(二):aries → hyper01,吞吐差 23 倍,全量已发射

- **效率审查触发的换机**(项目规矩:预计超 1 小时的任务发射前必须估墙钟)。aries 冒烟实测 **180–208 秒/步**,全量 3,125 步 = **约 165 小时(近 7 天)**,不可接受。三条原因叠加,都指向"aries 是错的机器":(1)为绕开 NCCL 死锁而设的 `NCCL_P2P_DISABLE=1`,使 ZeRO-3 每步的参数 all-gather 退化为经主机内存拷贝;(2)A6000 间无 NVLink,只有 PCIe;(3)48 GB 显存逼迫全分片。
- **换到 hyper01(4×H200)后实测 7.89 秒/步**,即 **快 23 倍**,全量约 **7 小时**。同一份数据、同一套超参、同样 ZeRO-3;唯一去掉的是 `NCCL_P2P_DISABLE`(H200 有 NVLink,禁 P2P 反而慢)。显存每卡仅约 24 GB / 143 GB,分片压力极小。
- **搬运成本**(实测,供以后决策):taurus → hyper01 直连 **29 MB/s**(音频 7.4 GB 约 4 分钟);基座模型改从 HF 直下 hyper01,**66 GB 约 90 秒**,远快于跨机中转——**结论:hyper01 需要公开权重时一律直接从 HF 拉,不要从别的主机推**。
- **发射参数**:容器 `sglang-omni-jaxan-5`(hyper01,GPU 0-3,已登记 map);LoRA r32/α32/dropout 0.05、`all-linear`、冻结 vit+aligner、global batch 4、max_length 2048、1 epoch、lr 1e-4 cosine→1e-5、warmup 5%、wd 0.01、clip 1.0、Adam β(0.9,0.95)、bf16、seed 42、ZeRO-3;每 200 步存档、`save_total_limit 3`,可 `--resume_from_checkpoint` 续训。
- **监控判据的教训**:先以"日志 mtime 是否更新"判活,在静默的预处理阶段**误报卡死**;改为"rank CPU 时间是否累加",全量则用"步数是否推进"。**另一个真实损失**:清理旧 run 时用 `pkill -f torchru[n]`,匹配不到 `python -m torch.distributed.run`,导致上一轮 NCCL 死锁的 run 变成孤儿、四个 rank 空转近两小时并抢占同一批卡,把新 run 拖慢。**今后清理一律按工作目录路径匹配(如 `phrase_sft_20260904`),不按框架名。**
- **我在本轮的两次误判(均已当场更正)**:一是拿旧 PID 的 CPU 时间断言"训练挂了",实际当前 run 在正常推进;二是称"只剩一个 rank",实为 `ps -p` 只查了手上三个 PID。判据教训:进程存活要按**工作目录路径**列全,不要按零散 PID 抽样。
- **状态**:全量训练 06:31Z 发射,预计 7 小时。之后导出合并权重 → 走已复现过的 OLT cascade 复测,与我们词对齐 36.81 / 他们 phrase-gated 38.32(CU BLEU)三方并排。

## 2026-09-05 自训 phrase-gated thinker(三):吞吐排查,ZeRO-3→ZeRO-2 提速 1.7 倍,余下瓶颈未解

- **实测三个配置**(hyper01,4×H200,同一份数据与超参):

  | 配置 | 步速 | GPU 利用率 | 每卡显存 | 全量 ETA |
  |---|---:|---:|---:|---:|
  | ZeRO-3,dataloader 4/rank | 41.8 s/step | 17–48% | 37 GB | 34 h |
  | ZeRO-3,dataloader 16/rank | 41.8 s/step | 17–48% | 37 GB | 34 h |
  | **ZeRO-2,dataloader 16/rank** | **23.9 s/step** | 19–21% | 118–121 GB | **20.3 h** |

- **已排除的瓶颈**:(1)CPU 侧音频解码——worker 从 16 增到 64,负载从 18.5 升到 34.9,**步速一点不变**;(2)参数通信只占一半——ZeRO-2 让 76 GB 参数常驻每卡(显存 37→118 GB 证实),步速降到 24 s,但 **GPU 利用率仍只有 19–21%**,说明还有未定位的瓶颈。
- **下一个待验假设**(未做):micro-batch 太小。global batch 4 摊到 4 卡,每卡每步仅 1 个样本 × 序列 2048,对 H200 而言计算量过小,单步固定开销(优化器步、梯度同步、框架开销)可能主导。验证法:提高 micro-batch,看步速增长是否显著小于线性。**但这会改变 global batch,偏离"除数据外一切不变"的实验前提**,因此作为下一轮的提速依据而非本轮改动。
- **决策**:保持 global batch 4 跑完(约 20 小时)。理由=本实验唯一变量必须是数据;改批大小会给 BLEU 对比引入无法排除的混淆项。回滚成本=若结果异常,重训一轮。
- **本段我的三次误报(均已当场更正,记此为戒)**:
  1. 用冒烟 15 步的 7.69 s/step 外推"全量 6.7 小时"——那 15 步吃的是预处理阶段预取的批次,非稳态。**教训:冒烟步数太少测不出稳态。**
  2. 据"GPU 利用率低 + 负载 18.5/224 核"断定瓶颈在数据加载,加 worker 无效。**教训:只看资源闲置就归因,没验证因果链。**
  3. **把一条内容损坏的监控事件当结果转述**,报出"6.58 s/step、45 步、ETA 5h37m、loss 1.29"——全部为假,当时训练尚未开始第一步。**教训:只报能从日志或文件复核到的数,监控事件仅作触发信号。**
- **状态**:106/3125 步,loss 1.216 → 0.683 正常下降,ETA 20 小时;每 200 步存档可续训。打分栈已备齐(XCOMET 已补回 13 GB),训练完即可导出合并权重并走 OLT cascade 复测。

## 2026-09-05 更正:上一条的「ZeRO-2 提速」不成立,真实对比是 ZeRO-3 vs DDP

- **作废内容**:上一条把 41.8 → 23.9 s/step 归因于 ZeRO-3 改 ZeRO-2。**该归因错误**。
- **真相**:改脚本时我把说明注释插进了 `swift sft` 的**反斜杠续行之间**。bash 在续行后遇到 `#` 会**就此结束整条命令**,其后所有行被当作新命令丢弃。于是 `--deepspeed zero2`、`--seed 42`、`--logging_steps`、`--save_steps`、`--save_total_limit`、`--dataloader_num_workers`、`--output_dir` **全部没有传给 swift**。日志实测 `deepspeed=None`、`save_steps=500`(默认)、`output_dir=/workspace/output/...`(容器内默认路径)。
- **因此正确的对照是**:ZeRO-3 = 41.8 s/step,**无 deepspeed 的纯 DDP = 23.9 s/step**。机制上讲得通:H200 143 GB 能完整放下 38B 模型(实测占 118 GB),DDP 不需要任何参数通信,而 ZeRO-3 每步要 all-gather 两遍全部参数。**结论仍是"不要用 ZeRO-3",但正确的替代是 DDP,不是 ZeRO-2**(ZeRO-2 至今未实际测过)。
- **同时作废**:「dataloader worker 16 无效」这一条的证据也不成立——那轮 worker 数其实是默认 4,并非我以为的 16。worker 的真实影响**未测**。
- **另一处更正**:我曾据单次瞬时采样称「GPU 利用率只有 19–21%,说明还有未定位瓶颈」。后续采样出现 100%/100%/20%/100%,说明利用率**周期性波动**,单点采样不足以定性。该推断证据不足,撤回。
- **运维硬伤**(促成重启):输出写在容器可写层 `/workspace/output`,`docker rm` 即全部丢失;`save_steps` 退回 500;`seed` 非 42,与原配方不一致。20 小时的任务不能押在这三条上。
- **修复与新纪律**:脚本重写为**命令续行内零注释**(依据全部移到文件头),并在发全量前用 `--max_steps 2` 跑一次**参数核对**,逐项确认解析结果。核对通过后重发,确认值:`output_dir=/data/phrase_sft/out_full/...`(挂载卷)、`save_steps=200`、`seed=42`、`dataloader_num_workers=16`、`deepspeed=None`(本次有意)。
  - **纪律一**:启动脚本改动后必须验证参数真正生效,不能只看进程起来了。
  - **纪律二**(前述):清理进程按工作目录路径匹配,不按框架名。
- **今日时间损失记账**:aries 环境探索约 3 h(终致换机)、aries 冒烟 2 h、孤儿进程抢卡约 2 h、ZeRO-3 两轮约 3 h、参数被吃掉的一轮 3 h。其中**孤儿进程与参数被吃掉两项(约 5 h)属我的操作失误**,非环境问题。
