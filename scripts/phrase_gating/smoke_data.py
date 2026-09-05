"""Encode a few phrase-gated rows with the real template: does swift read our manifest and audio?"""
import json, os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from swift.llm import get_model_tokenizer, get_template, load_dataset

BASE = "/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct"
DATA = "/mnt/gemini/data/jiaxuanluo/phrase_gating_20260904/train_s_zh_phrase_ours.jsonl"

_, processor = get_model_tokenizer(BASE, load_model=False)
template = get_template(processor.model_meta.template, processor, max_length=2048)
template.set_mode("train")

train, _ = load_dataset([DATA], split_dataset_ratio=0.0, num_proc=1)
print("dataset rows:", len(train))
row = train[0]
print("row keys:", sorted(row.keys()))
print("n messages:", len(row["messages"]), "| n audios:", len(row["audios"]))
print("first audio:", row["audios"][0], "exists:", os.path.isfile(row["audios"][0]))

for i in range(2):
    encoded = template.encode(train[i])
    ids = encoded["input_ids"]
    labels = [l for l in encoded["labels"] if l != -100]
    print(f"[row {i}] input_ids {len(ids)} | supervised tokens {len(labels)} | "
          f"keys {sorted(k for k in encoded if k not in ('input_ids','labels'))}")
print("SMOKE_DATA_OK")
