# OLT cascade serving 路径对照(2026-09-04)

用 `LeiLiLab/Open-LiveTranslate` 自己的 cascade recipe(`run_s2st_eval.sbatch ... moss-delta`)
跑的两条臂,唯一变量是 thinker checkpoint。目的:验证"级联 63.5 秒收尾偏移"是 serving 路径
造成的,而非 TTS 权重或 thinker。

- 数据:ACL 60-60 dev 前 3 篇(110/117/268,`MAX_DOCS=3`),源速 1.0×,chunk 1.92 s,en→zh。
- TTS:两臂同为 `owaski/moss-tts-realtime-delta-zh-125k@fc2d094d`,经 `moss_tts_delta_server.py`
  (每个 delta 一个 turn,`--codec-context conversation`,`--max-context-positions 600`)。
- speaker prompt:OLT 自带 `assets/spk_prompt/zh_1.wav`,两臂一致。
- 打分:OLT 官方栈(ElevenLabs Scribe v2 → SEGALE + penalize-v1 → BLEU/LongYAAL → XCOMET-XL)。
- 主机:hyper01,thinker vLLM TP=2,TTS 独占第三张卡。

| thinker | regime | BLEU | XCOMET-XL | LongYAAL | 收尾偏移 | 欠译罚 |
|---|---|---:|---:|---:|---:|---:|
| `owaski/infinisst-thinker-phrase-zh`(phrase-gated) | CU | 38.32 | 0.719 | 5,039 ms | 4,730 ms | 3 |
| `gavinlaw/infinisst-no-tmsft-origin-bsz4-zh`(我们,词对齐) | CU | 36.81 | 0.703 | 4,864 ms | 4,490 ms | 0 |
| `owaski/...phrase-zh` | CA | 36.20 | 0.689 | 7,685 ms | 7,541 ms | 10 |
| `gavinlaw/...origin...` | CA | 35.43 | 0.674 | 7,537 ms | 7,323 ms | 11 |

对照他们 README 公布的(5 篇全集,L40S):BLEU 40.64、XCOMET 0.723、LongYAAL CU 4207 /
CA 6114 ms、收尾 CU 4018 / CA 5942 ms。XCOMET 几乎完全对上;BLEU 与延迟的差来自
只跑 3 篇、以及本次有欠译罚(每段 10 秒)与硬件不同(H200 vs L40S,CA 是计算感知的)。

**结论**:同一族级联在正确 serving 路径上收尾偏移 7.3 s,在我们自家
`moss_multiturn_infer.py` 上是 63.5 s。差别不在 TTS 权重(换权重只动 0.5 BLEU),
也不在 thinker(我们的在这条路上还略快),而在喂文本的方式,见
`projects/infinisst_moss_tts_cascade/research_log.md` 同日条目与
`scripts/latency_probe/`。

文件按 `<run>.<名字>` 命名。原始 run 目录在 hyper01 `/data04/jaxan/serving_ab/results/`。
