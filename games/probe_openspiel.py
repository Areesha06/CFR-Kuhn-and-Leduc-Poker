"""Run once: tells you OpenSpiel's Leduc rules (is fold legal when nobody has bet?)."""
import pyspiel
g = pyspiel.load_game("leduc_poker")
s = g.new_initial_state()
s.apply_action(0); s.apply_action(2)           # deal cards
print("legal actions at start of round 1:", s.legal_actions(), "(0=fold,1=call,2=raise)")
print("info string:", s.information_state_string(0))
s.apply_action(1)
print("after check, player 1:", s.legal_actions())
k = pyspiel.load_game("kuhn_poker").new_initial_state()
k.apply_action(0); k.apply_action(1); k.apply_action(1)
print("Kuhn info strings:", k.information_state_string(0), k.information_state_string(1))
