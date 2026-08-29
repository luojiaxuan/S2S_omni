# InfiniSST 侧改动（短语边界策略）

InfiniSST 不在本仓库。这里保存基于上游
`LeiLiLab/InfiniSST@54f3471c556473dde7fb52ac85cff62bb6fc41ee` 的可应用
patch，作为本项目 phrase-boundary SFT 代码的 Git source of truth：

- [`phrase_boundary_sft.patch`](./phrase_boundary_sft.patch)
- 涉及 `train/dataset.py`、`train/main.py`、`model/model.py`
- 记录短语重分配、参数透传、各 multiplier 的 hold 预算诊断，以及显式续训入口

应用与验证：

```bash
git checkout 54f3471c556473dde7fb52ac85cff62bb6fc41ee
git apply --check /path/to/phrase_boundary_sft.patch
git apply /path/to/phrase_boundary_sft.patch
```

patch 于 2026-08-29 从 Aries 持久工作副本
`/mnt/taurus/home/jiaxuanluo/InfiniSST` 直接导出，并通过 `git diff --check`。
该工作副本还有三个与本任务无关的 launcher/term-training 修改，均未收入 patch。

核心不变量：只在同一张 chunk 网格上重新分配文本；chunk 数量与每个 chunk
的音频 patch 数保持不变，结构上避免音文错位。运行、权重和实验结论见
[`docs/handoff_codex_20260830.md`](../../../docs/handoff_codex_20260830.md) 与
[`docs/experiment_ledger_moss_tts_cascade_20260808.md`](../../../docs/experiment_ledger_moss_tts_cascade_20260808.md)。
