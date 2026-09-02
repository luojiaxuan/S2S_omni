# ChatGPT 外审全文:Open-LiveTranslate gemini-live-translate 基线 PR(2026-09-02)

- 通道:chatgpt.com 临时聊天,档位"极高",思考 3m16s。
- 审前判断:`prior_judgment_gemini_baseline_pr_20260902.md`。
- 判决摘要见文末"处置"一节(哪些采纳、哪些坚持)。

## 原文(英文,略去引用角标)

The adapter is close to defensible, but I would attack two things hard before calling the measurement clean: fresh-session rotation is not yet proven necessary, and 30-second acoustic silence is not a valid success signal. There is also a subtler rotation artifact that can flatter Gemini: concurrent tail output from session n and early output from n+1 can make arrival-time CA look better than anything a listener could actually hear.

### Q1 — Rotation bias: direction is not one-sided

The nastiest issue is cross-session output overlap. At t=480 you must continue sending source into the new session on the global schedule, while the old session may still be emitting its translation tail. If both WebSockets produce audio concurrently, "timestamp every delta at arrival" gives session n+1 credit for audio that may arrive while session n's audio is still supposed to be played. GPT's single stream cannot exploit that concurrency. Unless your output semantics explicitly serialize a virtual playout queue, Gemini can get artificially low CA. Conversely, concatenating by session number hides the real wall-clock queueing delay; concatenating by callback arrival can interleave two translations and corrupt the waveform. Per-session transport logs do not by themselves solve this. I would make cross-session voiced-overlap an explicit measured diagnostic and either charge serialization delay or treat overlap as an invalid handoff.

The context reset itself has indeterminate sign. It can hurt Gemini through lost antecedents, terminology, discourse state, speaker/style continuity, and a boundary that can occur mid-phoneme or mid-word. But a reset can also help by clearing accumulated recognition mistakes, style drift, or a large context. Google's current API reference explicitly notes that shorter retained context can make responses faster, so repeated fresh sessions can plausibly flatter latency. There are also field reports on this exact preview model of within-session voice drift, making "reset always hurts" especially unsafe as an assumption.

Handshake/cold-start pressure goes the other direction. Let H_j be setup-to-setupComplete time for rotation j, and p_j how far before the boundary you preconnect: post-boundary lateness L_j = max(0, H_j − p_j). Under your repo rule, any L_j > 0.25 s kills the document. If you only start connection/setup at the boundary, a normal WebSocket handshake can turn every long document into a lottery. Preconnecting is not a pacing exemption — it sends no source early — so I would permit it and fingerprint the lead time. Google says clients should wait for setupComplete before sending subsequent messages.

You can give a useful mechanical bound but not a useful semantic bound. B = ceil(T/S) − 1 resets; T ≤ S documents have zero rotation effect. No nontrivial worst-case semantic bound exists. Once you have a live run, if only the first W seconds after each boundary are perturbed with max latency change Δ, the time-weighted contribution is bounded by |Δ_L| ≤ min(1, BW/T)·Δ. Until W and Δ are established, "rotation costs only a boundary-sized region" is unjustified.

More importantly, 480 s currently needs justification. As of September 1, 2026, Google's general Live documentation says audio-only sessions are limited to 15 minutes without compression, with connection lifetime around 10 minutes, and documents session resumption plus context-window compression as continuity mechanisms. The Live Translate-specific page itself does not document a 480-second limit. I would currently reject the wording "Gemini forces context loss every 480 s." There is even an unofficial July report that sessionResumption began working on gemini-3.5-live-translate-preview. If resumption works, deliberately opening a fresh logical session is a benchmark-induced handicap. If it does not work, rotation is a legitimate service-level limitation — but call the result "Gemini Live Translate, fresh-session rotation," because it is no longer a pure model comparison against GPT's single session.

### Q2 — I would not accept quiet-window == completion

The API now gives you a better protocol path. realtimeInput.audioStreamEnd=true explicitly says the audio stream has ended. Server content defines generationComplete as "model is done generating" and turnComplete as completion of the turn; generationComplete can precede turnComplete. Send audioStreamEnd immediately after the final scheduled chunk and treat generationComplete as the preferred successful terminal condition; turnComplete is corroboration.

Thirty seconds of silence should be a timeout/failure detector, not an affirmative success proof. "Truncated output then silence" is observationally identical to "completed output then silence" — worse, truncation improves the latency statistic because missing late material supplies no late arrivals. A PCM peak threshold is a poor proxy for server state (low-amplitude speech, intentional pauses). Record separately "last WebSocket audio-delta arrival" and "last above-threshold sample"; don't promote the latter into a clean completion event. If the first live probe shows Live Translate never emits generationComplete after audioStreamEnd, record the outcome as completion unconfirmed / quiet-timeout, not silently redefine quietness as an ACK.

### Q3 — rate abort, CU:=CA, and URL auth

The 24 kHz abort is good benchmark hygiene, provided the entire document becomes failed rather than the pre-abort prefix leaking into aggregates. Google currently specifies Live Translate output as raw 16-bit LE PCM mono 24 kHz. Parse MIME semantically (case, whitespace, parameter ordering); require raw PCM and a parseable rate of exactly 24000; a missing or contradictory rate should also be protocol failure if the contract requires a declared rate; check even-byte payloads.

CU := CA is the part I would quarantine: serializing a fake CU invites a future aggregation to present CA as conventional CU. Prefer CU=null with a reason; if the schema forces the alias, fingerprint the alias semantics and hard-assert in reporting code. (Repo already fingerprints timing_schema and provenance.cu_equals_ca.)

Key in the WebSocket URL is Google's documented raw-WebSocket auth form — not a design smell in itself. The danger is secondary propagation: connection exceptions, upgrade failures, repr() of socket/client, tracing spans, crash reports can contain the full URI. Scrub query parameters by NAME (key, access_token) before any URI reaches logging or exception paths; never persist the original connect URI; never fingerprint the key or a hash of it. A bad-key live smoke should specifically check whether the library embeds the URI in its raised exception.

### Q4 — shipping without live numbers

Defensible as "implemented / live-unvalidated / no scores"; not as a validated baseline. Stub tests establish the state machine, not the wire contract — and Google's Live Translate guide and API reference currently DISAGREE on where transcription config fields live (guide nests under generationConfig; API reference puts them on BidiGenerateContentSetup; a July user report says the runtime rejected the nested form). A perfect stub can confidently validate the wrong API. MAX_DOCS=1 is the right first transition; if that document is shorter than the rotation threshold, the most consequential branch is still live-unvalidated.

Merge blockers: (1) do not describe fresh 480-second context resets as provider-mandated until live behavior proves it; (2) do not record acoustic quietness as confirmed successful completion. Next-highest: demonstrate rotation cannot obtain parallel-output credit GPT cannot. Everything else — strict rate abort, global pacing, per-delta arrival stamps, transport records, no resampling, aggressive secret redaction — is pointed in the right direction.

## 处置(我的判断)

- **采纳(改代码)**:
  1. generationComplete/turnComplete 进事件分类;audioStreamEnd 后以最后 session 的
     generationComplete 为首选 clean exit(`generation_complete`),静默窗降级为
     fallback(保留 `quiet_after_stream_end`,语义=completion unconfirmed,README 披露);
     stub 发 generationComplete;
  2. 480s 措辞全部改为"保守的 harness 选择"(现行文档:音频 session 15min、连接
     ~10min、有 sessionResumption 机制;live-translate 页无 480s);首跑指令加
     "选长 talk 覆盖 rotation + 探测 resumption/completion 行为";结果命名带
     "fresh-session rotation";
  3. cross-session voiced-overlap 诊断:meta.sessions 记每 session 首/末 loud delta
     t_src,重叠可直接读出(不改打分口径——GPT 侧 arrival 口径同样含 backlog,
     两臂口径一致,见"坚持"第 1 条);
  4. connect/setup 异常文本过 redact 再抛;mimeType 大小写不敏感;缺 rate 计数披露。
  5. setup 字段位置分歧(guide vs API reference:transcription 配置嵌套位置)写进
     docstring 风险注记——我方沿用 2026-07 实测可用的形状(顶层),与 7 月社区报告一致。
- **坚持原判(附理由)**:
  1. 打分口径不给 delays 加 playout 串行化:GPT 侧(连续 track)的 arrival 同样先于
     可播时刻(backlog),两臂同用 arrival 口径,改单边才是引入不对称;overlap 以诊断
     披露,重叠显著时由维护者裁决是否作废该文档;
  2. CU:=CA 保持:#31 已合并的仓库合同,provenance.cu_equals_ca + timing_schema
     的隔离机制已存在,单方面改 schema 反而破坏一致性;
  3. preconnect 不加 knob:现实现即"上一段发完立即预连"(≈0.96s lead,不发早源),
     无参数可 sweep;真实首跑若握手超窗再议。
- **采纳(仅文档/PR 措辞)**:PR 定位为 implemented / live-unvalidated / no scores。
