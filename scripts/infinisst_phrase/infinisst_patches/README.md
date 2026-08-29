# InfiniSST 侧改动（短语边界策略）

本目录记录对 `~/InfiniSST` 的改动。**InfiniSST 不在本仓库**，
这里只留改动片段供接力者对照，不是可直接应用的 patch 文件。

改动的核心不变量：**只在同一张 chunk 网格上重新分配文本**，
chunk 数量与每 chunk 的音频 patch 数一字不动——结构上杜绝音文错位。


## `train/dataset.py` —— 模块级：短语重分配函数

```python
PHRASE_PUNCT = "。！？!?…，,、；;：:"


def _phrase_redistribute(traj, max_hold_steps, min_chars):
    """把 [ [text, loss_flag], ... ] 里的文本重新分配到短语边界上。

    chunk 数量不变（音频网格不动），只把未到边界的文本顺延到后面的 chunk：
    未到边界的 chunk 目标变为空串（模型学会 hold），到边界时一次性写出。
    hold 超过 max_hold_steps 个 chunk 则强制写出，作为延迟上限。
    """
    out = [['', True] for _ in traj]
    buf = ''
    held = 0
    for i, (text, _) in enumerate(traj):
        buf += text
        stripped = buf.strip()
        if not stripped:
            held = 0
            continue
        held += 1
        n_chars = sum(1 for c in buf if c.isalnum())
        at_boundary = stripped[-1] in PHRASE_PUNCT and n_chars >= min_chars
        if at_boundary or held >= max_hold_steps or i == len(traj) - 1:
            out[i][0] = buf
            buf = ''
            held = 0
    if buf:  # 兜底：不应发生（末位已强制写出），发生则并入最后一个 chunk
        out[-1][0] += buf
    return out
```


## `train/dataset.py` —— collator __init__：打印各 multiplier 的 hold 预算与退化计数

```python
if phrase_boundary:
            # note (luojiaxuan): 打印各 multiplier 的 hold 预算。max_hold_steps==1
            # 等价于「永不 hold」（held>=1 立即成立），若大部分 multiplier 落在 1，
            # 该配置就在教模型「立即写出」，与 phrase 策略对冲——2026-08-28 的
            # 阴性重训正是栽在这里（12 个取值里 7 个退化）。
            _tab, _bad = [], 0
            for _m in range(1, int(max_multiplier) + 1):
                _h = max(1, int(round(phrase_max_hold_s / (self.speech_segment_size * 0.08 * _m))))
                _tab.append(f"m={_m}:{_h}")
                _bad += (_h == 1)
            logger.info(f"phrase hold budget per multiplier: {' '.join(_tab)}")
            logger.info(f"phrase degenerate (max_hold_steps==1, 永不 hold): {_bad}/{len(_tab)} multipliers")
```


## `train/dataset.py` —— collator __call__：在 multiplier 合并之后调用

```python
if self.phrase_boundary:
                # note (luojiaxuan): 同一 chunk 网格上重分配文本，chunk 数不变
                step_s = self.speech_segment_size * 0.08 * multiplier
                max_hold_steps = max(1, int(round(self.phrase_max_hold_s / step_s)))
                new_traj = _phrase_redistribute(new_traj, max_hold_steps, self.phrase_min_chars)
            x.trajectory = new_traj
```


## `model/model.py` —— 透传给 collator（两处调用点之一）

```python
            phrase_boundary=self.data_args.phrase_boundary,
            phrase_max_hold_s=self.data_args.phrase_max_hold_s,
            phrase_min_chars=self.data_args.phrase_min_chars,
```


## `model/model.py` —— 透传给 collator（两处调用点之一）

```python
            phrase_boundary=self.data_args.phrase_boundary,
            phrase_max_hold_s=self.data_args.phrase_max_hold_s,
            phrase_min_chars=self.data_args.phrase_min_chars,
```
