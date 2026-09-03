#!/usr/bin/env bash
set -uo pipefail
R=/data/ext_s2st; cd $R/repo
until grep -q MODEL_DONE $R/setup.log; do sleep 30; done
. $R/venv/bin/activate
export PYTHONPATH="$PWD:$PWD/external/SimulEval"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
M=models/SimulS2ST-Omni
mkdir -p $R/out
common="--source $R/acl/source.txt --target $R/acl/target.txt --source-lang English --target-lang Chinese --model-name-or-path $M/offline --checkpoint-path $M/simuls2st_adapter --source-segment-size 1000 --sacrebleu-tokenizer zh --eval-latency-unit char"
# S2ST m2: the README's recommended config; outputs wavs + instances.log
( python -m simuleval.cli --agent src/agents/simuleval_omni_talker_s2st_agent.py $common \
    --output $R/out/s2st_m2 --latency-multiplier 2 \
    --history-window-turns 28 --history-overlap-turns 16 --prompt-source-chunks 2 --prompt-generated-chunks 1 \
    --thinker-max-new-tokens 256 --talker-max-new-tokens 500 --thinker-no-sample --talker-no-sample \
    --thinker-num-beams 4 --thinker-repetition-penalty 1.2 --thinker-no-repeat-ngram-size 5 \
    --talker-repetition-penalty 1.4 --talker-no-repeat-ngram-size 5 --computation-aware --no-scoring \
    > $R/out/s2st_m2.log 2>&1; echo "S2ST_M2_EXIT=$?" >> $R/out/status ) &
# S2TT m2 and m3: text BLEU with SimulEval's own scoring against the whole-talk references
( for m in 2 3; do
    python -m simuleval.cli --agent src/agents/simuleval_omni_talker_s2tt_agent.py $common \
      --output $R/out/s2tt_m$m --latency-multiplier $m --num-beams 4 --repetition-penalty 1.2 --no-repeat-ngram-size 5 \
      > $R/out/s2tt_m$m.log 2>&1; echo "S2TT_M${m}_EXIT=$?" >> $R/out/status
  done ) &
wait
echo ALL_DONE >> $R/out/status
