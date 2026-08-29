# Research Log

## 2026-08-29：MOSS codec decoder 跨 turn context A/B

- **假设**：逐 turn 重建 `AudioStreamDecoder` 并重置 `codec.streaming()` 是 turn 边界 speaker 音色跳变的直接原因。
- **受控变量**：固定 talk110 方案 A 输入只生成一次 MOSS audio codes；A 每 turn 重置 decoder，B 用一个 decoder 连续解码全部 194 turns。
- **实现**：`scripts/moss_multiturn_infer.py --codes-out-dir` 冻结生成 codes；`scripts/codec_decoder_context_ab.py` 对同一 NPZ 执行 A/B，并强制校验每帧恰好对应 1,920 samples。
- **结果**：8,581 code frames，内容 SHA256 `20a157efc7a5bd7ded58d52ad958ed5f521ed41bb85568ea415892b46a029`；A/B 均为 686.48 秒、16,475,520 samples，容器 exit 0。用户听感确认 B 明显更稳定，speaker 跳变大幅减少。
- **Keep/Drop**：确认 codec decoder reset 是 talk110 音色跳变的直接原因。修复方向保留跨 turn decoder state，phrase boundary 只作为减少边界和改善韵律的独立优化。
