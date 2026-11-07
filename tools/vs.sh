#!/bin/bash
# Judged match of agent.py vs an opponent script. bash tools/vs.sh OPP_PATH DEALS SEED
cd "$(dirname "$0")/.."; PY=.venv/bin/python
PORT=5108 nice -n 10 $PY agent.py >/dev/null 2>&1 & A=$!
PORT=5109 nice -n 10 $PY "$1" >/dev/null 2>&1 & B=$!
for i in $(seq 1 60); do curl -s localhost:5108/ >/dev/null && curl -s localhost:5109/ >/dev/null && break; sleep 0.2; done
out=$($PY judge_engine.py --agent1 5108 --agent2 5109 --deals $2 --seed $3 2>&1)
echo "$out" | grep -E "^agent1: " | sed "s|^|$(basename $1 .py): |"
echo "$out" | grep -E "Confidence|RESULT" | tr '\n' ' '; echo
echo "$out" | grep -E "agent1 half" | grep -oE "thinking [0-9.]+s|errors [0-9]+" | tr '\n' ' '; echo
kill $A $B 2>/dev/null; wait $A $B 2>/dev/null; true
