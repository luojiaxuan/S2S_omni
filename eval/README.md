# eval/ — live S2S 基线音频生成

通过商用 live 翻译 API 生成基线译文音频,并记录逐 packet 到达时刻供延迟
分析。三个入口都把源音频按真实时间 pace 推流(chunk 边界 `asyncio.sleep`),
对每个返回的音频 packet 记 `received_at_s`(相对 session 起点的
`time.monotonic()`),因此测得的延迟是端到端实测值——包含服务端计算、
排队与网络,无法拆出"非计算"部分。

| 脚本 | API | 数据集 |
| --- | --- | --- |
| `run_acl6060_live_stream_eval.py` | OpenAI Realtime translations WS / Gemini live-translate WS(`--provider openai\|gemini`) | ACL 60/60(README 主表的 GPT-realtime 与 Gemini 行) |
| `run_floras_openai_realtime.py` | OpenAI Realtime translations WS | FLORAS |
| `run_floras_gemini_live.py` | Gemini live-translate WS | FLORAS |

驱动脚本:`scripts/run_acl6060_live_compare.sh`(ACL 60/60 全表)、
`scripts/run_floras_enzh_full_live_compare.sh`(FLORAS en→zh)。FLORAS 两个
入口的共享工具在 `s2s_omni/floras_live.py`。API key 经
`OPENAI_API_KEY` / `GEMINI_API_KEY` 或驱动脚本的 key 文件参数传入。
