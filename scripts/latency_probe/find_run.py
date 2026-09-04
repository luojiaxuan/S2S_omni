import json, os, subprocess
paths = subprocess.run(["bash", "-lc",
    "find /data04/jaxan /data02/jaxan /data01/jaxan -name 'talk110*.summary.jsonl' 2>/dev/null"],
    capture_output=True, text=True).stdout.split()
for p in paths:
    if os.path.basename(p).startswith("._"):
        continue
    try:
        d = json.loads(open(p, encoding="utf-8").readline())
    except Exception as exc:
        print(f"  ?? {p}: {exc}"); continue
    t = d.get("turns", [])
    print(f"{len(t):5d} turns  dur={d.get('duration_s',0):8.1f}s  ctx={d.get('codec_context')}  "
          f"fail={d.get('failure')}  {p}")
