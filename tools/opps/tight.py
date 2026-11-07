"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "drew"
AGENT_NAME = "tight"


# ===== YOUR CODE HERE =====
# Fallback bot: equity against a random hand, priced against the pot.
#   Preflop: a fitted formula for equity vs a random hand (pairs, suits,
#   connectedness, high cards; R^2 0.66 against simulation), no sampling.
#   Postflop: Monte Carlo equity vs a random hand (up to 16 samples), lowered when
#   the opponent bets or raises, more so if they rarely do.
#   The sample count adapts so our own thinking stays near 3 s per half.

import random
import time

PRE_W = (0.0655, 0.0588, -0.0641, 0.0317, -0.0035, -0.0279, 0.0075, 0.0609, 0.237)
# The judge's 8 s bank per 1000-hand half also pays for request handling
# (about 1 ms a call), so our own thinking is budgeted at OWN_BUDGET seconds
# and the sample count shrinks as it is used up.
OWN_BUDGET = 3.0
HALF_HANDS = 1000


def _action_name(a) -> str:
    return getattr(a, "name", str(a)).upper()


def preflop_equity(hole) -> float:
    ranks = sorted((c >> 2 for c in hole), reverse=True)
    suits = [c & 3 for c in hole]
    cnt = {r: ranks.count(r) for r in set(ranks)}
    sc = [suits.count(s) for s in range(4)]
    distinct = sorted(set(ranks))
    best = run = 1
    for a, b in zip(distinct, distinct[1:]):
        run = run + 1 if b - a <= 2 else 1
        best = max(best, run)
    f = (
        sum(1 for r, k in cnt.items() if k == 2 and r >= 6),
        sum(1 for r, k in cnt.items() if k == 2 and r < 6),
        sum(1 for k in cnt.values() if k >= 3),
        sum(1 for k in sc if k >= 2),
        sum(1 for s in range(4) if sc[s] >= 2 and any(c >> 2 == 8 and c & 3 == s for c in hole)),
        sum(1 for k in sc if k >= 4),
        best,
        sum(ranks) / 8.0,
        1.0,
    )
    return sum(w * x for w, x in zip(PRE_W, f))


def mc_equity(hole, board, n, rng) -> float:
    known = set(hole) | set(board)
    deck = [c for c in range(36) if c not in known]
    need = 5 - len(board)
    wins = 0.0
    for _ in range(n):
        d = rng.sample(deck, 5 + need)
        full = list(board) + d[5:]
        a = omaha_score(hole, full)
        b = omaha_score(d[:5], full)
        wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / n


def _track_opponent(obs, memory):
    seen = memory.setdefault("seen", {})
    hist = obs.action_history
    start = seen.get(obs.hand_index, 0)
    for a in hist[start:]:
        if not a.is_me:
            memory["opp_acts"] = memory.get("opp_acts", 0) + 1
            if "RAISE" in _action_name(a.action):
                memory["opp_raises"] = memory.get("opp_raises", 0) + 1
    seen[obs.hand_index] = len(hist)
    if len(seen) > 4:
        for k in sorted(seen)[:-2]:
            del seen[k]


def _pick(obs, wanted):
    """First action of `wanted` that is legal; check/call, then fold, as fallbacks."""
    legal = obs.legal_actions
    for a in wanted:
        if a in legal:
            return a
    if Action.CHECK_CALL in legal:
        return Action.CHECK_CALL
    return legal[0]


def _decide(obs, memory):
    _track_opponent(obs, memory)
    acts = memory.get("opp_acts", 0)
    opp_rate = (memory.get("opp_raises", 0) + 3) / (acts + 10)   # prior 0.3
    opp_raised = any((not a.is_me) and a.street == obs.street and "RAISE" in _action_name(a.action)
                     for a in obs.action_history)
    to_call = obs.to_call
    need = to_call / max(obs.pot + to_call, 1)
    raises = obs.raises_this_street

    if obs.street == Street.PREFLOP:
        eq = preflop_equity(obs.hole_cards)
        if raises >= 2:
            if eq >= 0.62:
                return _pick(obs, [Action.RAISE_POT, Action.CHECK_CALL])
            return _pick(obs, [Action.CHECK_CALL] if eq >= 0.55 - 0.05 * (opp_rate > 0.5) else [Action.FOLD])
        if eq >= 0.60:
            return _pick(obs, [Action.RAISE_POT, Action.RAISE_HALF])
        if eq >= 0.55 and raises == 0:
            return _pick(obs, [Action.RAISE_HALF])
        if to_call == 0:
            return Action.CHECK_CALL
        floor = 0.52 if opp_rate > 0.5 else 0.55
        if eq >= max(floor, need + 0.12) or to_call <= obs.big_blind // 2:
            return Action.CHECK_CALL
        return _pick(obs, [Action.FOLD])

    spent = memory.get("spent", 0.0)
    left_hands = max(HALF_HANDS - obs.hand_index, 1)
    per_hand = (OWN_BUDGET - spent) / left_hands          # seconds we may still spend per hand
    n = 0 if per_hand <= 0 else min(16, int(per_hand / 0.00025))   # ~80 us a sample, ~3 postflop calls a hand
    if n >= 4:
        rng = random.Random(obs.hand_index * 7919 + len(obs.board) * 31 + raises)
        eq = mc_equity(obs.hole_cards, obs.board, n, rng)
    else:
        eq = preflop_equity(obs.hole_cards)
    if opp_raised:
        eq -= 0.12 if opp_rate < 0.25 else (0.07 if opp_rate < 0.5 else 0.02)
    if to_call == 0:
        if eq >= 0.75:
            return _pick(obs, [Action.RAISE_POT, Action.RAISE_HALF])
        if eq >= 0.62:
            return _pick(obs, [Action.RAISE_HALF])
        return Action.CHECK_CALL
    if eq >= 0.80 and raises < 3:
        return _pick(obs, [Action.RAISE_POT, Action.RAISE_HALF])
    if eq >= 0.70 and raises < 2:
        return _pick(obs, [Action.RAISE_HALF, Action.CHECK_CALL])
    if eq >= need + 0.12:
        return Action.CHECK_CALL
    return _pick(obs, [Action.FOLD])


def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands)."""
    memory["hands_played"] = 0
    memory["spent"] = 0.0


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Returns one of obs.legal_actions."""
    t0 = time.perf_counter()
    try:
        action = _decide(obs, memory)
        if action not in obs.legal_actions:
            action = _pick(obs, [Action.CHECK_CALL, Action.FOLD])
    except Exception:
        action = _pick(obs, [Action.CHECK_CALL, Action.FOLD])
    memory["spent"] = memory.get("spent", 0.0) + (time.perf_counter() - t0)
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand."""
    memory["hands_played"] = memory.get("hands_played", 0) + 1

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
