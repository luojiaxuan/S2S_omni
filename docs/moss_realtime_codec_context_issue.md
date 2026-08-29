# MOSS Realtime 多轮 codec context 问题

## 结论

MOSS Audio Tokenizer 的 decoder 是因果模型。当前推理如果每个 turn 都重新创建 decoder，并重新进入 `codec.streaming()`，codec 看到的上下文就只剩当前 turn。

talk110 的固定 codes 对照已经确认这个问题。我们只生成一次 194 个 turn 的 audio codes，然后用两种方式解码。A 每个 turn 重置 decoder。B 全场只使用一个 decoder。两边读取完全相同的 8,581 帧 codes，输出也都是 686.48 秒。用户听感结论是 B 明显更稳定，speaker 跳变大幅减少。

这个结果说明，至少在该样本上，codec decoder reset 是音色跳变的直接原因。InfiniSST 攒更长文本可以减少边界数量，但不是修复这个问题的必要条件。

## sglang-omni 中的对应问题

当前对应实现是 sglang-omni PR 1410，而不是旧的 PR 1192/1368。PR 1410 是 framework-native MOSS-TTS-Realtime 路径，包含增量 text 输入、跨 turn warm session 和 persistent streaming vocoder。截至 2026-08-29，它仍是 open draft，尚未合入 main；核对版本为 `c5455d9934f0d7e44c16f0ba13ef7849c1f0e323`。

PR 1410 的 `_CodecStreamSession` 确实只进入一次 `codec.streaming()`，但 codec slot 仍按 turn/request 租用。`_RealtimeStreamState` 已记录 `session_id` 和 `turn_id`；turn 结束时，`decode_delta(is_final=True)` 先输出 pending frames，再调用 `_release_state_slot()`，最终由 `_CodecStreamSession.release()` reset 该 slot。所以下一个同 session turn 虽然复用了 generator history，codec 的因果状态仍从零开始。

任何把同一 speaker session 拆成多个 speech request 的 serving 路径都可能遇到这个问题。

## 建议修复

1. 用稳定的 session id 管理 codec slot，不要只按 request/turn 管理。

2. 同一 session 的成功连续 turn 复用同一个 slot 和 codec causal state。turn 完成只需输出 pending PCM，不应 reset slot。

3. session 关闭、TTL 到期、abort 或失败时释放并 reset slot，不同 session 之间不能共享。

4. 如果服务结构不方便长期保留 state，可以在新 request 开始时 prefill 最近一段 audio codes，并丢弃 prefix waveform。codec 的有效上下文有界，不需要回放整场历史。

5. 回归测试固定一份 codes，把它切成多个 turns。同一 session 的第二个 turn 必须延续 codec state；新 session 或已关闭 session 必须从零开始。服务路径仍应在每个 turn 及时输出 PCM。

当前 A/B 的 B 只在全场末尾调用一次 `AudioStreamDecoder.flush()`。服务侧还应补一个 C 对照：保留同一个 codec state，但在每个 turn 输出所有 pending PCM。PR 1410 的 vocoder 已把“输出 pending frames”和“reset slot”写成相邻的两个动作，最直接的验证就是只延后后者。

## 证据

实验脚本：`scripts/codec_decoder_context_ab.py`

实验报告：`projects/infinisst_moss_tts_cascade/artifacts/codec_decoder_context_ab_talk110/report.json`

用于上游复核的 60 秒 A/B 取完全相同的 sample 区间
`[8,565,120, 10,005,120)`，24 kHz 单声道 PCM，中心是 386.88 秒的
boundary 109：

- A，每个 application-level turn 重置 decoder：
  <https://github.com/luojiaxuan/S2S_omni/releases/download/talk110-codec-context-ab-20260829/talk110_boundary109_60s_reset.wav>
  （SHA256 `2b210c58f4d978fade417aca49c0404003d80d97080d617c20a9068d219aff6b`）
- B，跨 turn 保留同一个 decoder context：
  <https://github.com/luojiaxuan/S2S_omni/releases/download/talk110-codec-context-ab-20260829/talk110_boundary109_60s_continuous.wav>
  （SHA256 `ea28198d87251984448d924b04fed79ce1315df46497a45a87645ff82e37774f`）

GitHub Release：<https://github.com/luojiaxuan/S2S_omni/releases/tag/talk110-codec-context-ab-20260829>

完整 A/B 暂存在 `/Users/luojiaxuan/Downloads/talk110_codec_context_ab_20260829/`，状态为 `PENDING_HF_UPLOAD`；GitHub Release 只托管上述诊断片段。

## 上游跟踪

sglang-omni issue：<https://github.com/sgl-project/sglang-omni/issues/1812>

60 秒 A/B comment：<https://github.com/sgl-project/sglang-omni/issues/1812#issuecomment-5465326690>

当前实现：<https://github.com/sgl-project/sglang-omni/pull/1410>

PR review：<https://github.com/sgl-project/sglang-omni/pull/1410#pullrequestreview-5059392974>

- slot 应由 `session_id` 持有并跨 turn 复用：<https://github.com/sgl-project/sglang-omni/pull/1410#discussion_r3887899486>
- 正常 turn final flush 后不能 release/reset：<https://github.com/sgl-project/sglang-omni/pull/1410#discussion_r3887899490>
