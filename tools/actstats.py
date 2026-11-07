"""Postflop action frequencies of agent.py vs v2 against the same opponent (in-process)."""
import collections, importlib.util, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from game import AgentConfig, Game, make_deal, Street  # noqa: E402

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

opp_path = sys.argv[1] if len(sys.argv) > 1 else "tools/opps/v1.py"
for label, path in (("v2", "tools/opps/v2.py"), ("v3", "agent.py")):
    me = load(ROOT / path, "me_" + label); op = load(ROOT / opp_path, "op_" + label)
    mm, mo = {}, {}; me.new_half(AgentConfig(), mm); op.new_half(AgentConfig(), mo)
    C = collections.Counter(); net = 0
    for h in range(600):
        g = Game(make_deal("acts", h), button_seat=h % 2, hand_index=h)
        while not g.is_terminal:
            s = g.to_act; obs = g.observation(s)
            if s == 0:
                a = me.choose_action(obs, mm)
                key = ("pre" if obs.street == Street.PREFLOP else "post") + ("-free" if obs.to_call == 0 else "-facing")
                C[(key, a.name)] += 1
            else:
                a = op.choose_action(obs, mo)
            g.apply(a)
        r0 = g.hand_result(0); net += r0.my_net
        me.on_hand_end(r0, mm); op.on_hand_end(g.hand_result(1), mo)
    out = []
    for key in ("pre-free", "pre-facing", "post-free", "post-facing"):
        tot = sum(v for (k, _), v in C.items() if k == key)
        out.append("%s n=%d %s" % (key, tot, " ".join("%s %.0f%%" % (a[:5], 100 * C[(key, a)] / max(tot, 1)) for a in ("FOLD", "CHECK_CALL", "RAISE_HALF", "RAISE_POT") if C[(key, a)])))
    print(label, "net %+d over 600 hands |" % net, " | ".join(out))
