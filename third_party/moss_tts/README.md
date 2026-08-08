# MOSS-TTS 本地补丁（S2S_omni 级联所需）

上游：<https://github.com/OpenMOSS/MOSS-TTS>，基线 commit
`58b20a0d5fcc6766658d50967a90a9d890009a46`。

`moss_tts_s2s_omni.patch` 是本项目对上游的两处必要改动，之前只存在于
hyper00 的本地工作区（`/data04/jaxan/MOSS-TTS`），现固化进 Git 作为
source of truth，便于在 Tilde 等新机器上复现：

1. `moss_tts_realtime/finetuning/dataset.py` —— `context_only` turn 只进
   上下文、不进 loss。v2.1/v3 的重复污染增强和 v4 的自生成历史都依赖它。
2. `moss_tts_realtime/mossttsrealtime/modeling_mossttsrealtime_local.py`
   —— 适配 transformers 5.6 的 `create_causal_mask` 签名
   （`input_embeds` → `inputs_embeds`，去掉 `cache_position`）。不打这个
   补丁，多 turn 滑窗推理会在 local transformer 处直接抛错。

应用方式：

```bash
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
git checkout 58b20a0d5fcc6766658d50967a90a9d890009a46
git apply /path/to/moss_tts_s2s_omni.patch
```
