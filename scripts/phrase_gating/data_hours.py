"""How much audio and text is one epoch of this manifest?"""
import json, wave, os, random

M = "/data/phrase_sft/train_s_zh_phrase_local.jsonl"
rows = [json.loads(l) for l in open(M, encoding="utf-8") if l.strip()]
print(f"rows (utterances): {len(rows)}")

chunks = sum(len(r["audios"]) for r in rows)
uniq = len({p for r in rows for p in r["audios"]})
print(f"audio chunk references: {chunks}  (unique files: {uniq})")

# duration of a sample of unique files, extrapolated
paths = list({p for r in rows for p in r["audios"]})
random.seed(0)
sample = random.sample(paths, 300)
tot = 0.0
for p in sample:
    with wave.open(p, "rb") as h:
        tot += h.getnframes() / h.getframerate()
mean = tot / len(sample)
print(f"mean chunk duration: {mean:.2f}s (300-file sample)")
print(f"=> referenced audio per epoch: {chunks*mean/3600:.1f} h")
print(f"=> distinct audio in corpus  : {uniq*mean/3600:.1f} h")

chars = sum(len(m["content"]) for r in rows for m in r["messages"] if m["role"] == "assistant")
print(f"target characters: {chars:,}  (~{chars/len(rows):.0f} per utterance)")
turns = sum(1 for r in rows for m in r["messages"] if m["role"] == "assistant")
print(f"assistant turns: {turns:,}  (~{turns/len(rows):.1f} per utterance)")
