# 2×2 交互检验的四格证据

外审(`docs/reviews/20260912-phrase-gating-bug-attribution.md`)指定的关键量不是
`P_fixed > W_fixed`,而是交互项

    Δ_word   = M(W_fixed) − M(W_old)
    Δ_phrase = M(P_fixed) − M(P_old)
    I        = Δ_phrase − Δ_word

`I` 显著为正,才说明"空轮无监督"这个 bug **不成比例地**伤害了 phrase 臂。

| 格 | thinker | loss | job | 主机 | 权重(HF) |
|---|---|---|---|---|---|
| W_old | 词对齐 | default(带 bug) | 92011 | aries | `gavinlaw/infinisst-no-tmsft-origin-bsz4-zh@main` `fd0a5c8ff931` |
| P_old | phrase-gated | default(带 bug) | 92021 | aries | `gavinlaw/infinisst-thinker-phrase-gated-zh@main` `83a95f5bf730` |
| W_fixed | 词对齐 | empty_turn_end_w0.5 | 93011 | hyper01 | `gavinlaw/infinisst-no-tmsft-origin-bsz4-zh@empty-turn-end-w0.5`,权重提交 `0897538e544c` |
| P_fixed | phrase-gated | empty_turn_end_w0.5 | 93001 | hyper01 | `gavinlaw/infinisst-thinker-phrase-gated-zh@empty-turn-end-w0.5`,权重提交 `0a317c92c3f3` |

- old 两格:aries 评估驱动记录的 revision(`fd0a5c8f` / `83a95f5b`)与两个 repo 当前 `main` 的 head 一致。
- fixed 两格:评估时读的是 hyper01 本地导出,其 26 个文件与上表的 HF 权重提交逐一比对过内容(LFS 文件比 sha256,其余比 git blob sha1),完全相同。分支 head 比权重提交多一个 model card 提交;本地导出已在核验后删除。

每格存 `metrics.json`(打分)、`render_report.json`(per-talk,用于空调用占比诊断)、
`generation_config.json`(口径指纹,用于证明四格只在 thinker 上不同)。
出表:`python3 scripts/phrase_gating/ab_2x2_collect.py --w_old ... --p_old ... --w_fixed ... --p_fixed ...`

## 结果(CU,5 篇 ACL 60-60 dev,greedy thinker + greedy TTS,1.92 s chunk)

| 格 | BLEU | XCOMET-XL | 收尾偏移 | 空调用占比 |
|---|---:|---:|---:|---:|
| W_old | 40.59 | 0.7228 | 4808 ms | 3.0% |
| P_old | 39.80 | 0.7405 | 4211 ms | 7.5% |
| W_fixed | 40.01 | 0.7189 | 4481 ms | 9.5% |
| P_fixed | 42.53 | 0.7626 | 4876 ms | 41.6% |

Δ_word = −0.59 BLEU / −0.004 XCOMET,Δ_phrase = +2.74 / +0.022,**I = +3.32 / +0.026**。
数值见 `summary.json`;判读与界限(尤其是未做延迟匹配:P_fixed 的收尾偏移是四格最高)见
`projects/infinisst_moss_tts_cascade/research_log.md` 的 2026-09-13 条目。
