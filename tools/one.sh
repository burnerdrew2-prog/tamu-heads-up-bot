#!/bin/bash
# One judged match: bash tools/one.sh BASELINE DEALS SEED
cd "$(dirname "$0")/.."; PY=.venv/bin/python
PORT=5108 nice -n 10 $PY agent.py >/dev/null 2>&1 & A=$!
PORT=5109 nice -n 10 $PY baseline-bots/$1.py >/dev/null 2>&1 & B=$!
for i in $(seq 1 50); do curl -s localhost:5108/ >/dev/null && curl -s localhost:5109/ >/dev/null && break; sleep 0.2; done
$PY judge_engine.py --agent1 5108 --agent2 5109 --deals $2 --seed $3 2>&1 | grep -E "agent1: |Confidence|agent1 half|RESULT"
kill $A $B 2>/dev/null; wait $A $B 2>/dev/null; true
