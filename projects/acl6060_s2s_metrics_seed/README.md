# ACL6060 Seed S2S Metrics Script

This project bundle tracks the Seed / ByteDance AST speech-to-speech evaluation
script provided under:

```text
/Users/luojiaxuan/Downloads/seed
```

The imported source lives in:

```text
vendor/seed/
```

## Contents

- `vendor/seed/generate.py`: streams wav audio to the AST S2S WebSocket API and
  writes target wav, timeline JSON, and transcript text.
- `vendor/seed/protos/`: protobuf definitions and generation helper.
- `vendor/seed/python_protogen/`: generated Python protobuf files required by
  `generate.py`.

Runtime cache files such as `__pycache__` and `*.pyc` are intentionally not
tracked.

## Credential Handling

The original downloaded script contained hard-coded AST credentials. Those have
been removed before committing to this public repository.

Pass credentials explicitly at runtime:

```bash
python projects/acl6060_s2s_metrics_seed/vendor/seed/generate.py \
  input.wav \
  --out-dir output/seed/en_ja \
  --src-lang en \
  --tgt-lang ja \
  --api-key '<AST_API_KEY>'
```

For legacy two-part auth:

```bash
python projects/acl6060_s2s_metrics_seed/vendor/seed/generate.py \
  input.wav \
  --out-dir output/seed/en_ja \
  --src-lang en \
  --tgt-lang ja \
  --app-key '<AST_APP_KEY>' \
  --access-key '<AST_ACCESS_KEY>'
```

## Dependencies

```bash
pip install soundfile numpy websockets "protobuf>=6.31" grpcio grpcio-tools
```

If protobuf definitions need to be regenerated:

```bash
cd projects/acl6060_s2s_metrics_seed/vendor/seed/protos
./build_python.sh
```

## Outputs

For each input wav, the script writes:

- `<stem>.wav`: translated target speech with silence gaps rendered on the live
  timeline.
- `<stem>_timeline.json`: receive/playback timing data for target chunks.
- `<stem>.txt`: subtitle transcript emitted by the backend.
