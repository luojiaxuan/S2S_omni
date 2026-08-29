# MOSS Realtime 多轮 codec context 问题

## 结论

MOSS Audio Tokenizer 的 decoder 是因果模型。当前推理如果每个 turn 都重新创建 decoder，并重新进入 `codec.streaming()`，codec 看到的上下文就只剩当前 turn。

talk110 的固定 codes 对照已经确认这个问题。我们只生成一次 194 个 turn 的 audio codes，然后用两种方式解码。A 每个 turn 重置 decoder。B 全场只使用一个 decoder。两边读取完全相同的 8,581 帧 codes，输出也都是 686.48 秒。用户听感结论是 B 明显更稳定，speaker 跳变大幅减少。

这个结果说明，至少在该样本上，codec decoder reset 是音色跳变的直接原因。InfiniSST 攒更长文本可以减少边界数量，但不是修复这个问题的必要条件。

## sglang-omni 中的对应问题

PR 1192 的 `MossTTSRealtimeVocoder` 会在每个 request 的 `create_stream_state` 中进入一次 `codec.streaming()`，并在 `release_stream_resources` 中退出。

PR 1368 已经支持把历史 text 和 audio codes 放回 MOSS TTS generator prompt，但每个新 turn 仍然使用新的 request id。因此 generator 有多轮历史，vocoder 仍然没有上一轮的 decoder context。

任何把同一 speaker session 拆成多个 speech request 的 serving 路径都可能遇到这个问题。

## 建议修复

1. 用稳定的 session id 管理 vocoder state，不要只按 request id 管理。

2. 同一 session 的连续 turn 复用 codec streaming state。

3. session 关闭、超时或 abort 时释放 state，不同 session 之间不能共享。

4. 如果服务结构不方便长期保留 state，可以在新 request 开始时 prefill 最近一段 audio codes，并丢弃 prefix waveform。codec 的有效上下文有界，不需要回放整场历史。

5. 回归测试固定一份 codes，把它切成多个 turns。服务路径的拼接结果应与单次连续 decode 使用相同的 decoder context。

## 证据

实验脚本：`scripts/codec_decoder_context_ab.py`

实验报告：`projects/infinisst_moss_tts_cascade/artifacts/codec_decoder_context_ab_talk110/report.json`

完整 A/B 暂存在 `/Users/luojiaxuan/Downloads/talk110_codec_context_ab_20260829/`，状态为 `PENDING_HF_UPLOAD`。

## 上游跟踪

sglang-omni issue：<https://github.com/sgl-project/sglang-omni/issues/1812>
