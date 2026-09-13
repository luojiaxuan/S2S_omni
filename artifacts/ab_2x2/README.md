# 2×2 交互检验的四格证据

外审(`docs/reviews/20260912-phrase-gating-bug-attribution.md`)指定的关键量不是
`P_fixed > W_fixed`,而是交互项

    Δ_word   = M(W_fixed) − M(W_old)
    Δ_phrase = M(P_fixed) − M(P_old)
    I        = Δ_phrase − Δ_word

`I` 显著为正,才说明"空轮无监督"这个 bug **不成比例地**伤害了 phrase 臂。

| 格 | thinker | loss | job | 主机 |
|---|---|---|---|---|
| W_old | 词对齐 | default(带 bug) | 92011 | aries |
| P_old | phrase-gated | default(带 bug) | 92021 | aries |
| W_fixed | 词对齐 | empty_turn_end_w0.5 | 93011 | hyper01 |
| P_fixed | phrase-gated | empty_turn_end_w0.5 | 93001 | hyper01 |

每格存 `metrics.json`(打分)、`render_report.json`(per-talk,用于空调用占比诊断)、
`generation_config.json`(口径指纹,用于证明四格只在 thinker 上不同)。
出表:`python3 scripts/phrase_gating/ab_2x2_collect.py --w_old ... --p_old ... --w_fixed ... --p_fixed ...`
