# Research Log

## 2026-08-30：连续 codec context 下复评 phrase-policy

- **假设**：修复 codec decoder 的跨 turn context 后，InfiniSST phrase v2-ep1 相对 baseline 的级联优势会明显缩小，1.5× 退化可能随 speaker 跳变一起消失。
- **受控变量**：冻结 baseline/phrase 在 1×/1.5× 已生成的 InfiniSST rows；四格统一用 v7 TTS、seed 42、soft3 和每 talk 一个连续 codec decoder；逐 turn Qwen3-ASR 后统一做 SEGALE BLEU。
- **结果**：baseline 30.16→33.97（1×→1.5×，+3.81）；phrase 36.80→31.34（−5.46）。phrase 在 1× 高 6.64，但在 1.5× 低 2.63。phrase 1.5× 的目标音频总时长为 3562.96 秒，较其 1× 的 3265.52 秒反增 9.11%；baseline 只增 1.00%。
- **原因判断**：speaker 跳变和 1.5× 内容退化是两个问题。前者由 codec reset 直接解释；后者在连续 decoder 下仍存在。当前 TTS 仍是旧 trajectory 分布训练的 `v7@1947001d`，没有针对 phrase-policy 输出重做 SFT，因此 TTS 分布失配是最强嫌疑，但尚未用重训对照直接证明。
- **Keep/Drop**：保留跨 turn codec context 作为 speaker 修复；phrase v2-ep1 保留为 1× 质量候选，不作为跨语速默认操作点。若继续推进 phrase，下一步先用新 policy 输出重做 TTS SFT，再复跑同一冻结四格合同。
- **产物**：`projects/infinisst_moss_tts_cascade/artifacts/codec_context_phrase_eval_20260830/summary.json`；完整 1.6G run 在 hyper01 `/data02/jaxan/S2S_omni/runs/20260830-044041-145584000`，状态 `PENDING_HF_UPLOAD`。

## 2026-08-29：MOSS codec decoder 跨 turn context A/B

- **假设**：逐 turn 重建 `AudioStreamDecoder` 并重置 `codec.streaming()` 是 turn 边界 speaker 音色跳变的直接原因。
- **受控变量**：固定 talk110 方案 A 输入只生成一次 MOSS audio codes；A 每 turn 重置 decoder，B 用一个 decoder 连续解码全部 194 turns。
- **实现**：`scripts/moss_multiturn_infer.py --codes-out-dir` 冻结生成 codes；`scripts/codec_decoder_context_ab.py` 对同一 NPZ 执行 A/B，并强制校验每帧恰好对应 1,920 samples。
- **结果**：8,581 code frames，内容 SHA256 `20a157efc7a5bd7ded58d52ad958ed5f521ed41bb85568ea415892b46a029`；A/B 均为 686.48 秒、16,475,520 samples，容器 exit 0。用户听感确认 B 明显更稳定，speaker 跳变大幅减少。
- **Keep/Drop**：确认 codec decoder reset 是 talk110 音色跳变的直接原因。修复方向保留跨 turn decoder state，phrase boundary 只作为减少边界和改善韵律的独立优化。
