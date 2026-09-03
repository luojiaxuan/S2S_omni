# OLT 官方打分栈复测(2026-09-03)

`LeiLiLab/Open-LiveTranslate` 的官方 S2ST 打分栈对两套系统的完整产物。经由该仓
PR #40 新增的 timeline 入口打分:外部系统只需交出"贴到源时钟的渲染音频 + 每段
播放起止 + 系统自己的文本",不改动打分栈本身。

- 数据:ACL 60-60 dev 的 3 篇 talk(2022.acl-long.110 / .117 / .268),源速 1.0×,en→zh。
- ASR:ElevenLabs Scribe v2(整段一次请求,逐词时间戳)。质量:SEGALE 重分段 +
  penalize-v1 skip policy + SacreBLEU(zh)+ XCOMET-XL。延迟:LongYAAL 与 Ending Offset,
  均由渲染音频上的 ASR 时间戳算出。
- 两 run 的 `document_fingerprint` 相同(`8437ac49…`),即同一批文档;generation 指纹
  不同,按该仓规则不可合并统计。CU 为 CA 的字节副本(外部系统一律如此)。

| 系统 | chunk | BLEU | XCOMET-XL | LongYAAL | Ending Offset | 段数 | skip(欠/超译) |
|---|---|---:|---:|---:|---:|---:|---:|
| SimulS2ST-Omni(latency multiplier 2) | 2.0 s | 40.65 | 0.658 | 3,658 ms | 3,233 ms | 253 | 0 / 0 |
| InfiniSST phrase v2ep1 + MOSS-TTS v8 | 1.92 s | 34.08 | 0.646 | 53,186 ms | 63,525 ms | 267 | 0 / 2 |

本目录文件按 `<run>.<文件名>` 命名:`metrics.json`(全部指标与 provenance)、
`generation_config.json`(不可变 identity 指纹)、`render_report.json`(两棵 wav 树的渲染记录)。
原始 run 目录在 hyper00 `/data04/jaxan/olt_build/results/`(容器已删,数据留至
2026-09-10)。口径与自建链(`artifacts/simuls2st_omni_comparison_20260903/`)不同,
两者数字不可并排,理由见 `projects/infinisst_moss_tts_cascade/research_log.md` 同日条目。
