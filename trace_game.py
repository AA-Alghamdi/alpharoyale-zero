"""Play one sim game and print a readable move-by-move trace of an agent."""
import numpy as np

from crsim.actions import action_id_to_action
from crsim.cards import CardType
from crsim.game import Action, CRGame, GameResult
from eval.baseline_agents import HeuristicAgent, ProAgent

rng = np.random.RandomState(7)
_pool = list(CardType)
deck0 = [_pool[i] for i in rng.choice(len(_pool), 8, replace=False)]
deck1 = [_pool[i] for i in rng.choice(len(_pool), 8, replace=False)]
hero, foe = ProAgent(), HeuristicAgent()
game = CRGame(deck_p0=deck0, deck_p1=deck1, seed=7)

print("HERO deck:", [c.name for c in deck0])
print("=" * 64)
interval = 8
moves = 0
while not game.done:
    if game.tick_count % interval == 0:
        aid0 = hero.select_action(game, 0)
        a0 = action_id_to_action(aid0, 0)
        aid1 = foe.select_action(game, 1)
        a1 = action_id_to_action(aid1, 1)
        if not a0.is_wait and moves < 30:
            ps = game.players[0]
            card = ps.deck[ps.hand[a0.hand_slot]]  # CardType (has .name)
            enemies_near = sum(
                1 for e in game.entities
                if e.alive and not e.is_tower and e.owner == 1 and e.y < 16
            )
            t = game.tick_count * 0.05
            print(f"t={t:5.1f}s elixir={ps.elixir:4.1f}  PLAY {card.name:16} "
                  f"@tile({int(a0.x):2d},{int(a0.y):2d})  [enemies on our side: {enemies_near}]")
            moves += 1
    else:
        a0 = Action(player=0, hand_slot=-1)
        a1 = Action(player=1, hand_slot=-1)
    game.step([a0, a1])

print("=" * 64)
res = {GameResult.P0_WIN: "HERO WINS", GameResult.P1_WIN: "HERO LOSES",
       GameResult.DRAW: "DRAW"}.get(game.result, str(game.result))
n = game.numbers if hasattr(game, "numbers") else None
ekt = game.king_towers[1]
akt = game.king_towers[0]
ept = [t.hp for t in game.princess_towers[1]]
apt = [t.hp for t in game.princess_towers[0]]
print(f"RESULT: {res}  (match {game.tick_count*0.05:.0f}s)")
print(f"  enemy towers: king={max(ekt.hp,0):.0f} princess={[max(h,0) for h in ept]}")
print(f"  our   towers: king={max(akt.hp,0):.0f} princess={[max(h,0) for h in apt]}")
print(f"  total HERO plays this game: {moves}")
