# MOSS-TTS 本地补丁（S2S_omni 级联所需）

上游：<https://github.com/OpenMOSS/MOSS-TTS>，基线 commit
`58b20a0d5fcc6766658d50967a90a9d890009a46`。

`moss_tts_s2s_omni.patch` 是本项目对上游的四处必要改动，之前只存在于
hyper00 的本地工作区（`/data04/jaxan/MOSS-TTS`），现固化进 Git 作为
source of truth，便于在 Tilde 等新机器上复现：

1. `moss_tts_realtime/finetuning/dataset.py` —— `context_only` turn 只进
   上下文、不进 loss；允许纯标点 turn 使用空 `audio_codes`，让模型学习直接
   预测 audio EOS。phrase policy 会真实产生这种 turn，不能把标点重新并回
   前一 turn，否则训练分布与推理分布仍不一致。
2. `moss_tts_realtime/finetuning/sft.py` —— 修正 LR scheduler 步进。
   `accelerator.prepare()` 会让 scheduler 每个 optimizer step 推进
   `num_processes` 次，而上游传给 `get_scheduler` 的步数没有乘回来，导致
   调度提前 `num_processes` 倍跑完（实测 3 进程时学习率在约 1/3 处就归零，
   后 2/3 训练等于零学习率空转）。补丁把 `num_warmup_steps` 与
   `num_training_steps` 都乘上 `accelerator.num_processes`。
   **注意**：打了这个补丁后训练动力学会变，与 2026-08-08 之前的
   v1/v2/v2.1/v3 checkpoint 不再同源，跨版本对比需重新建立基线。
3. `moss_tts_realtime/mossttsrealtime/modeling_mossttsrealtime_local.py`
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
