"""What speaking rate was the TTS trained to produce? Read a sample of the training rows."""
import itertools, json, statistics as st

P = "/data02/jaxan/tmp/runs/20260830-200824-401864000/data/input.jsonl"
FPS = 12.5
rates, turn_rates, n_rows = [], [], 0
with open(P, encoding="utf-8") as fh:
    for line in itertools.islice(fh, 4000):
        row = json.loads(line)
        chars = frames = 0
        for turn in row["conversations"]:
            if turn.get("role") != "assistant":
                continue
            c = sum(1 for ch in turn.get("text", "") if not ch.isspace())
            f = len(turn.get("audio_codes") or [])
            if c and f:
                chars += c; frames += f
                turn_rates.append(c / (f / FPS))
        if chars and frames:
            rates.append(chars / (frames / FPS))
            n_rows += 1

def pct(xs, q): return sorted(xs)[int(q * len(xs))]
print(f"训练语料 {n_rows} 行 / {len(turn_rates)} 个 assistant turn（前 4000 行样本）")
print(f"  按行:  中位 {st.median(rates):.2f} 字/秒   p10 {pct(rates,0.1):.2f}   p90 {pct(rates,0.9):.2f}")
print(f"  按turn: 中位 {st.median(turn_rates):.2f} 字/秒   p10 {pct(turn_rates,0.1):.2f}   p90 {pct(turn_rates,0.9):.2f}")
print(f"  <2.5 字/秒的 turn 占比: {100*sum(1 for r in turn_rates if r < 2.5)/len(turn_rates):.1f}%")
print(f"  参考:三篇 talk 追平源时长需要 4.54 / 4.90 / 4.76 字/秒")
