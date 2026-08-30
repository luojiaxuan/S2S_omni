# Research Log

## 2026-08-30 phrase-policy 匹配 TTS SFT

- **假设**：旧 v7 TTS 没有见过 InfiniSST phrase-policy 的 text/turn 分布，1.5× BLEU 退化主要来自这一分布偏移；用实际 phrase trajectory 重训 TTS 后，跨语速 BLEU 应恢复。
- **症状**：旧 v7 TTS 接 phrase 输出时，BLEU 为 36.80→31.34，1.5× 下降 5.46；speaker 跳变已由连续 codec decoder context 独立修复。
- **改动**：从相同 base model 训练 v8，训练集为 36,529 行 v6 base、6,385 行 phrase full、3,839 行 phrase mid-start；1 epoch，global batch 15，seed 42。
- **结果**：匹配训练后的 BLEU 为 34.09→29.61，1.5× 仍下降 4.48。加速档文字增加 3.06%，生成音频增加 4.80%，null alignment 从 1.50% 升至 7.14%。
- **Keep/Drop**：keep 连续 codec decoder context；drop“旧 TTS 分布偏移足以解释跨语速退化”的假设。phrase policy 暂不作为跨语速默认配置；级联指标不能继续单独定位剩余问题属于 InfiniSST 还是 TTS。
- **证据**：`artifacts/phrase_matched_tts_eval_20260830/summary.json`；合同为 `configs/phrase_matched_tts_v8_20260830.json`。
