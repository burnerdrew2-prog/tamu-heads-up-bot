"""Play hands locally against an opponent script and print drew_v2's reasoning."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluator import cards_to_str  # noqa: E402
from game import Action, AgentConfig, Game, make_deal  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


me = load(ROOT / "agent.py", "me")
opp = load(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "tools/opps/aggro.py"), "opp")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
mem_me, mem_opp = {}, {}
me.new_half(AgentConfig(), mem_me)
opp.new_half(AgentConfig(), mem_opp)
hands = []
for h in range(N):
    g = Game(make_deal("explain", h), button_seat=h % 2, hand_index=h)
    log = []
    while not g.is_terminal:
        seat = g.to_act
        obs = g.observation(seat)
        if seat == 0:
            mem_me["_explain"] = []
            a = me.choose_action(obs, mem_me)
            log.append(("me", obs.street.name, cards_to_str(obs.board), obs.pot, obs.to_call, a.name, list(mem_me["_explain"])))
            mem_me["_explain"] = None
        else:
            a = opp.choose_action(obs, mem_opp)
            log.append(("opp", obs.street.name, cards_to_str(obs.board), obs.pot, obs.to_call, a.name, []))
        g.apply(a)
    r0, r1 = g.hand_result(0), g.hand_result(1)
    me.on_hand_end(r0, mem_me)
    opp.on_hand_end(r1, mem_opp)
    hands.append((h, cards_to_str(g.deal.hole_cards[0]), cards_to_str(g.deal.hole_cards[1]),
                  cards_to_str(g.deal.board), h % 2 == 0, r0, log))
pick = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
for h, mine, theirs, board, btn, r, log in hands:
    if pick is not None and h not in pick:
        continue
    print("\n=== hand %d  me %s (%s)  opp %s  board %s  -> me %+d chips%s" % (
        h, mine, "button" if btn else "big blind", theirs, board, r.my_net, ", showdown" if r.went_to_showdown else ""))
    for who, st, b, pot, tc, a, notes in log:
        print("  %-3s %-7s board [%s] pot %d to call %d -> %s" % (who, st, b, pot, tc, a))
        for n_ in notes:
            print("        why: " + n_)
if pick is None:
    import collections
    print("summary: net %+d over %d hands" % (sum(x[5].my_net for x in hands), N))
    big = sorted(hands, key=lambda x: -abs(x[5].my_net))[:12]
    print("biggest:", [(x[0], x[5].my_net, len(x[6])) for x in big])
