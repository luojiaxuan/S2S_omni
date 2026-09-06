#!/usr/bin/env bash
# note (luojiaxuan): Mac-side driver for everything after the Megatron phrase-gated SFT on aries:
# note (luojiaxuan): wait_train  the --rm train container is gone and train.log ends with STAGE_train_EXIT=0
# note (luojiaxuan): export      megatron_aries.sh export -> merged bf16 HF weights in $W/megatron_run/hf
# note (luojiaxuan): transfer    rsync hf/, mcore/ and the training jsonl to hyper01 (aries -> hyper01 by IP, the
# note (luojiaxuan): Mac's ssh agent signs through -A; the alias 'hyper01' exists only on the Mac)
# note (luojiaxuan): upload      hf_upload_phrase.py on hyper01: model repo + dataset repo, file-list reconciled
# note (luojiaxuan): cascade     hyper01_phrase_eval.sh up/cascade/compare: OLT run_s2st_eval, 3 dev talks
# note (luojiaxuan): score       hyper01_phrase_eval.sh score: ElevenLabs ASR + SEGALE/LongYAAL/BLEU/XCOMET
# note (luojiaxuan): down        hyper01_phrase_eval.sh down: container + map line removed
# note (luojiaxuan): Every stage appends "STAGE <name> OK|FAIL <detail>" to $STATUS and the first FAIL stops the
# note (luojiaxuan): driver; FROM=<stage> resumes from that stage.
set -uo pipefail
STATUS="${STATUS:?path of the status file this driver appends to}"
FROM="${FROM:-wait_train}"
HERE=$(cd "$(dirname "$0")" && pwd)
STATE=$(dirname "$STATUS")
W=/mnt/gemini/home/jiaxuanluo/phrase_sft_20260904
DATA=/mnt/gemini/data/jiaxuanluo/phrase_gating_20260904/train_s_zh_phrase_ours.jsonl
HY=sglang-omni@47.74.115.221
SAB=/data04/jaxan/serving_ab
REPO=gavinlaw/infinisst-thinker-phrase-gated-zh
DREPO=gavinlaw/infinisst-sft-phrase-gated-zh
O="-o RemoteCommand=none -o ConnectTimeout=30 -o ServerAliveInterval=60 -o ServerAliveCountMax=10"
SSH="ssh $O"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }
ok() { say "STAGE $1 OK $2"; }
fail() { say "STAGE $1 FAIL $2"; exit 1; }
STAGES=(wait_train export transfer upload cascade score down)
active=0
want() { [ "$1" = "$FROM" ] && active=1; [ $active = 1 ]; }

if want wait_train; then
  while :; do
    s=$($SSH aries "docker ps -q --filter name=^sglang-omni-jaxan-1\$ | wc -l; grep -ac 'after training is done' $W/train.log; grep -aoE 'STAGE_train_EXIT=[0-9]+' $W/train.log | tail -1" 2>/dev/null | tr '\n' ' ')
    alive=${s%% *}
    [ "${alive:-1}" = "0" ] && break
    sleep 120
  done
  read -r _ done_n exit_l <<<"$s"
  [ "${done_n:-0}" -ge 1 ] || fail wait_train "container gone without 'after training is done' ($s)"
  [ "${exit_l:-}" = "STAGE_train_EXIT=0" ] || fail wait_train "exit line '$exit_l'"
  iters=$($SSH aries "cat $W/megatron_run/mcore/latest_checkpointed_iteration.txt 2>/dev/null; ls -d $W/megatron_run/mcore/iter_* 2>/dev/null | xargs -n1 basename | tr '\n' ' '")
  ok wait_train "$(echo $iters)"
fi

if want export; then
  $SSH aries "cd $W && DEVICES=3,4,5,6 GPUS=4 bash megatron_aries.sh export > export.log 2>&1"
  e=$($SSH aries "grep -aoE 'STAGE_export_EXIT=[0-9]+' $W/export.log | tail -1; ls $W/megatron_run/hf/config.json $W/megatron_run/hf/model.safetensors.index.json 2>/dev/null | wc -l; du -sh $W/megatron_run/hf 2>/dev/null | cut -f1" | tr '\n' ' ')
  case "$e" in "STAGE_export_EXIT=0 2 "*) ok export "$e" ;; *) fail export "$e (aries $W/export.log)" ;; esac
fi

if want transfer; then
  R="rsync -a --partial --inplace -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new'"
  ssh -A $O aries "$R $W/megatron_run/hf/ $HY:$SAB/thinker_phrase_gated/ && $R $W/megatron_run/mcore/ $HY:$SAB/thinker_phrase_gated_mcore/ && $R $DATA $HY:$SAB/phrase_gated_data/ && $R $W/train.log $W/export.log $HY:$SAB/phrase_gated_data/" \
    || fail transfer "rsync exited $?"
  left=$(ssh -A $O aries "$R --dry-run --itemize-changes $W/megatron_run/hf/ $HY:$SAB/thinker_phrase_gated/ | grep -c '^[<>]' || true")
  sizes=$($SSH hyper01 "du -sh $SAB/thinker_phrase_gated $SAB/thinker_phrase_gated_mcore | cut -f1 | tr '\n' ' '")
  [ "${left:-1}" = "0" ] || fail transfer "$left files still differ after rsync"
  ok transfer "hf+mcore on hyper01: $sizes"
fi

if want upload; then
  facts=$($SSH hyper01 "sha256sum $SAB/phrase_gated_data/train_s_zh_phrase_ours.jsonl | cut -c1-16; wc -l < $SAB/phrase_gated_data/train_s_zh_phrase_ours.jsonl; grep -a 'elapsed time per iteration' $SAB/phrase_gated_data/train.log | head -1 | grep -aoE 'lm loss: [0-9.E+-]+' ; grep -a 'elapsed time per iteration' $SAB/phrase_gated_data/train.log | tail -1 | grep -aoE 'iteration +[0-9]+/ +[0-9]+|lm loss: [0-9.E+-]+' | tr '\n' ' '" | tr '\n' '|')
  IFS='|' read -r dsha rows loss0 lossN <<<"$facts"
  cat >"$STATE/.model_card.md" <<EOF
---
license: other
base_model: Qwen/Qwen3-Omni-30B-A3B-Instruct
language: [en, zh]
tags: [simultaneous-speech-translation, infinisst, lora, megatron, ms-swift]
---
# infinisst-thinker-phrase-gated-zh

InfiniSST en→zh thinker(Qwen3-Omni-30B-A3B-Instruct + LoRA,合并后的 bf16 权重),在**短语级 gating**
后的轨迹上训练:训练文本与词对齐版逐行恒等,只是把释放时机改成"累计 ≥8 个可读字,或遇短语标点且 ≥4 字"
(\`codec_ab/scripts/phrase_gating/phrase_gate_traj.py\`),使 thinker 以短语为单位输出 delta,供
MOSS-TTS 级联逐 delta 合成。

- 数据:\`$DREPO\` / \`train_s_zh_phrase_ours.jsonl\`(sha256 前 16 位 \`$dsha\`,$rows 行;ms-swift 按 0.01 切出 125 行验证)。
- 配方:Open-LiveTranslate \`finetune/recipes/omni_sft_recipe.sh\` 的 Megatron 三阶段(convert / train / export),
  在 aries(4×RTX A6000)上以 Docker 逐字复刻(\`codec_ab/scripts/phrase_gating/megatron_aries.sh\`):
  ms-swift 3.9.1 镜像 \`modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1\`,
  Megatron-LM \`73a28a1\`;LoRA r32/α32、\`target_modules all-linear\`、冻结 vit 与 aligner;
  \`expert_model_parallel_size 4\`、\`packing\`、\`max_length 2048\`、micro 1 × global 4、lr 1e-4(warmup 5%,min 1e-5)、
  1 epoch(597 步)。训练 lm loss:首步 ${loss0#lm loss: } → 末步 ${lossN#*lm loss: }。
- 导出:\`swift export --mcore_adapters mcore --to_hf true --torch_dtype bfloat16\`(合并 LoRA);\`mcore_lora/\` 保存
  Megatron 侧的 LoRA 迭代与 \`args.json\`。
- 评估口径与结果记录在 \`luojiaxuan/S2S_omni\` 的 \`projects/infinisst_moss_tts_cascade/research_log.md\`;
  在 OLT 级联里以 \`CKPT=<本地目录> THINKER_MODEL_REVISION=local:gavinlaw-infinisst-thinker-phrase-gated-zh:<commit8>\` 引用。
EOF
  cat >"$STATE/.data_card.md" <<EOF
---
license: other
language: [en, zh]
tags: [simultaneous-speech-translation, infinisst, sft]
---
# infinisst-sft-phrase-gated-zh

InfiniSST en→zh 的 SFT 轨迹,由词对齐轨迹经短语 gating 改写而来:按 chunk 累积译文,累计可读字 ≥8、
或以短语标点(。,、?!;:,.?!;:)结尾且 ≥4 字时释放,余量在末 chunk 释放;每行拼接后的文本与原轨迹
逐字相同(脚本 \`codec_ab/scripts/phrase_gating/phrase_gate_traj.py\`,release_chars=8,punct_min=4)。

- \`train_s_zh_phrase_ours.jsonl\`:$rows 行(sha256 前 16 位 \`$dsha\`),ms-swift messages 格式,\`audios\` 字段为
  训练主机上的本地路径;**音频不随本仓库分发**(InfiniSST 的 siqi zh v2 切片,约 7.4 GB)。
- 训练出的模型:\`$REPO\`。
EOF
  scp -q "$HERE/hf_upload_phrase.py" "$STATE/.model_card.md" "$STATE/.data_card.md" hyper01:$SAB/phrase_gated_data/ || fail upload "scp of upload script/cards"
  $SSH hyper01 "cd $SAB && HF_TOKEN=\$(cat /data04/jaxan/.keys/hf_token_gavinlaw) venvs/olt-thinker/bin/python phrase_gated_data/hf_upload_phrase.py --repo $REPO --hf thinker_phrase_gated --mcore thinker_phrase_gated_mcore --card phrase_gated_data/.model_card.md --data-repo $DREPO --data phrase_gated_data/train_s_zh_phrase_ours.jsonl --data-card phrase_gated_data/.data_card.md --out phrase_gated_data/upload.json > phrase_gated_data/upload.log 2>&1; echo UPLOAD_EXIT=\$?; cat phrase_gated_data/upload.json 2>/dev/null | tr -d '\n '" > "$STATE/.upload.out" 2>&1
  grep -q "UPLOAD_EXIT=0" "$STATE/.upload.out" || fail upload "$(tail -c 400 "$STATE/.upload.out")"
  u=$(sed -n 's/.*\({.*}\).*/\1/p' "$STATE/.upload.out" | tail -1)
  python3 -c "import json,sys; d=json.loads(sys.argv[1]); assert not d['missing'], d['missing']; assert d['data_ok']; print(d['sha'][:8])" "$u" >"$STATE/.sha8" 2>"$STATE/.upload.err" \
    || fail upload "reconciliation: $(cat "$STATE/.upload.err")"
  ok upload "$REPO@$(cat "$STATE/.sha8") $(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print('files', d['n_local'], '/', d['n_remote'], '| data', d['data_repo']+'@'+d['data_sha'][:8])" "$u")"
fi

if want cascade; then
  SHA8=$(cat "$STATE/.sha8" 2>/dev/null) || fail cascade "no .sha8 from the upload stage"
  scp -q "$HERE/hyper01_phrase_eval.sh" hyper01:$SAB/phrase_gated_data/ || fail cascade "scp of hyper01_phrase_eval.sh"
  up=$($SSH hyper01 "bash $SAB/phrase_gated_data/hyper01_phrase_eval.sh up $SHA8" 2>&1) || fail cascade "up: $up"
  say "cascade container: $(echo $up)"
  c=$($SSH hyper01 "bash $SAB/phrase_gated_data/hyper01_phrase_eval.sh cascade" 2>&1 | tr '\n' ' ')
  case "$c" in *CASCADE_phrase_EXIT=0*) ;; *) fail cascade "$c (hyper01 $SAB/phrase_gated_data/cascade_phrase.log)" ;; esac
  cmp=$($SSH hyper01 "bash $SAB/phrase_gated_data/hyper01_phrase_eval.sh compare" 2>&1)
  say "generation identity vs job 90002:"; say "$cmp"
  echo "$cmp" | grep -q "unexpected_diffs=0" || fail cascade "generation identity differs beyond the thinker"
  ok cascade "$c"
fi

if want score; then
  sc=$($SSH hyper01 "bash $SAB/phrase_gated_data/hyper01_phrase_eval.sh score" 2>&1)
  say "$sc"
  echo "$sc" | grep -q "SCORE_phrase_EXIT=0" || fail score "see hyper01 $SAB/phrase_gated_data/score_phrase.log"
  ok score "metrics.json written"
fi

if want down; then
  d=$($SSH hyper01 "bash $SAB/phrase_gated_data/hyper01_phrase_eval.sh down" 2>&1)
  ok down "$d"
fi
say "DRIVER DONE"
