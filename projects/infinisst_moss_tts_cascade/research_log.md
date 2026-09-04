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
