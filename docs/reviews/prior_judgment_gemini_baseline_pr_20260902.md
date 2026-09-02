# 外审前判断:Open-LiveTranslate gemini-live-translate 基线 PR(2026-09-02)

分支 `feat/gemini-live-baseline`(commit 8dc84a4),按已合并的 GPT 基线 PR #31
同一合同。发审前我方判断与理由:

1. **Session rotation(480s,chunk 边界切段)**:Live API 每 session 有输入
   上限,长 talk 必须轮换,别无选择;480s 来自本项目 2026-07 ACL sweep 的实测
   可行值。全局绝对 schedule 不豁免 pacing guard(握手挤在一个 chunk 间隙内,
   超时即迟发 abort),段间上下文丢失如实披露并 fingerprint
   (`max_session_input_s`)。风险:轮换边界的翻译质量损失会计入 Gemini 的
   BLEU——但这是"以该产品能被驱动的方式"的真实测量,与 PR #31 的哲学一致。
2. **quiet_after_stream_end 作为唯一 clean exit**:协议无 close/ack,静默窗是
   唯一可判据;服务器截断与说完在协议上不可区分,与 GPT 侧
   `transport_closed_after_close` 的盲区同构,per-session transport 状态落
   meta.sessions 供事后审。
3. **rate 违规 abort 不重采样**:重采样会吞掉协议变化且改变样本数语义;
   CU:=CA 只报 CA 沿用 #31;key 在 URL query,连接 URL 永不入任何记录,
   redact 覆盖 AIza 与 ?key= 两种形状。
4. **未跑真实 API 即提 PR**:与 #31 先例不同(它带了真实数字),但 #31 的
   Why 一节同样记录了"先有 harness、真实首跑 MAX_DOCS=1"的规则;我方无
   Gemini key 预算授权,不应擅自花钱。PR 如实标注状态,数字留给有 key 的
   维护者按 MAX_DOCS=1 首跑。

验证状态:stub 全链路(smoke + STUB=1 整 recipe)、40 个新测试、GPT 侧回归
168 passed(仅一个 baseline 固有的环境性失败)。
