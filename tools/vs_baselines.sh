#!/bin/bash
# Play agent.py against each baseline bot with the bundle's judge.
#   bash tools/vs_baselines.sh [deals] [seed]
cd "$(dirname "$0")/.."
DEALS=${1:-300}; SEED=${2:-7}; PY=.venv/bin/python
for b in baseline_random baseline_calling_station baseline_always_pot baseline_always_half; do
  PORT=5108 nice -n 10 $PY agent.py >/dev/null 2>&1 & A=$!
  PORT=5109 nice -n 10 $PY baseline-bots/$b.py >/dev/null 2>&1 & B=$!
  for i in $(seq 1 50); do curl -s localhost:5108/ >/dev/null && curl -s localhost:5109/ >/dev/null && break; sleep 0.2; done
  out=$($PY judge_engine.py --agent1 5108 --agent2 5109 --deals $DEALS --seed $SEED 2>&1)
  echo "== vs $b"; echo "$out" | grep -iE "bb/deal|per deal|interval|RESULT|error|forfeit|time" | head -8
  kill $A $B 2>/dev/null; wait $A $B 2>/dev/null
done
