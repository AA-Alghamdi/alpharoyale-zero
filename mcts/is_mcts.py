"""Information Set MCTS for imperfect-information search.

In Clash Royale, we don't know the opponent's hand, deck order, or exact
elixir. Standard MCTS assumes perfect information (game.clone() reveals
everything). IS-MCTS handles this by:

  1. At the root, sample a "determinization" — a concrete opponent hand/deck
     consistent with our belief state (OpponentBeliefState.sample_hand())
  2. Run a standard MCTS tree search on that determinization
  3. Repeat for K determinizations and average the action statistics

This is essentially "Multiple Observer MCTS" (Cowling et al., 2012).

We integrate with Gumbel MuZero search, running a separate search tree per
determinization but sharing the neural network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from crsim.cards import CardType
from crsim.constants import ACTION_SPACE_SIZE, NUM_HAND_SLOTS
from crsim.game import CRGame
from mcts.gumbel_search import GumbelConfig, GumbelMuZeroSearch
from model.opponent_model import OpponentBeliefState

logger = logging.getLogger(__name__)


@dataclass
class ISMCTSConfig:
    """Configuration for Information Set MCTS."""
    n_determinizations: int = 8
    gumbel_config: GumbelConfig | None = None
    belief_temperature: float = 1.0


class InformationSetMCTS:
    """MCTS that handles imperfect information via determinization sampling.

    For each search:
      1. Sample K opponent hand configurations from belief state
      2. Run Gumbel MuZero search on each determinization
      3. Average the resulting action probabilities

    This converges to a Nash equilibrium strategy in the information game.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: ISMCTSConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or ISMCTSConfig()
        self.device = device or torch.device("cpu")
        self.model = model

        gumbel_cfg = self.config.gumbel_config or GumbelConfig(n_simulations=16)
        self._searcher = GumbelMuZeroSearch(
            model=model, config=gumbel_cfg, device=self.device,
        )

    def search(
        self,
        game: CRGame,
        player: int,
        belief: OpponentBeliefState | None = None,
    ) -> np.ndarray:
        """Run IS-MCTS search.

        Parameters
        ----------
        game : CRGame
            Current game state (player's perspective — may not know
            opponent's full state).
        player : int
            Which player we are (0 or 1).
        belief : OpponentBeliefState, optional
            Belief over opponent's hidden state. If None, falls back
            to uniform assumptions.

        Returns
        -------
        action_probs : ndarray, shape (ACTION_SPACE_SIZE,)
        """
        n_det = self.config.n_determinizations
        all_probs = np.zeros((n_det, ACTION_SPACE_SIZE), dtype=np.float32)

        for d in range(n_det):
            # Sample a determinization
            det_game = self._create_determinization(game, player, belief)

            # Run Gumbel search on this determinization
            probs = self._searcher.search(det_game, player)
            all_probs[d] = probs

        # Average across determinizations
        avg_probs = all_probs.mean(axis=0)

        # Renormalize
        total = avg_probs.sum()
        if total > 0:
            avg_probs /= total
        else:
            valid = game.get_valid_actions_mask(player)
            n_valid = valid.sum()
            if n_valid > 0:
                avg_probs = valid.astype(np.float32) / n_valid
            else:
                avg_probs[-1] = 1.0  # wait

        return avg_probs

    def select_action(
        self,
        game: CRGame,
        player: int,
        belief: OpponentBeliefState | None = None,
        deterministic: bool = False,
    ) -> tuple[int, np.ndarray]:
        """Select an action using IS-MCTS.

        Returns (action_id, action_probs).
        """
        probs = self.search(game, player, belief)

        if deterministic:
            action_id = int(np.argmax(probs))
        else:
            action_id = int(np.random.choice(len(probs), p=probs))

        return action_id, probs

    def _create_determinization(
        self,
        game: CRGame,
        player: int,
        belief: OpponentBeliefState | None,
    ) -> CRGame:
        """Create a concrete game state by sampling opponent's hidden info.

        We keep our player's state unchanged and randomize the opponent's
        hand/deck order based on belief state.
        """
        import copy
        det = copy.deepcopy(game)

        opp = 1 - player
        opp_state = det.players[opp]

        if belief is not None:
            # Sample likely opponent hand from belief
            sampled_hand = belief.sample_hand(
                temperature=self.config.belief_temperature,
            )
            if sampled_hand is not None:
                # Reconstruct opponent deck with sampled cards in hand
                known_cards = set()
                new_deck = list(opp_state.deck)

                for slot_idx, card_type in enumerate(sampled_hand):
                    if slot_idx < NUM_HAND_SLOTS and card_type is not None:
                        ct = CardType(card_type) if isinstance(card_type, int) else card_type
                        # Find this card in deck or replace
                        for deck_idx in range(len(new_deck)):
                            if new_deck[deck_idx] == ct and deck_idx not in known_cards:
                                opp_state.hand[slot_idx] = deck_idx
                                known_cards.add(deck_idx)
                                break
        else:
            # No belief: shuffle opponent's unplayed deck cards
            hand_indices = set(opp_state.hand)
            free_indices = [
                i for i in range(len(opp_state.deck))
                if i not in hand_indices
            ]
            np.random.shuffle(free_indices)

        return det


class BeliefUpdater:
    """Update opponent belief state from game observations during play."""

    def __init__(self) -> None:
        self.beliefs: dict[int, OpponentBeliefState] = {
            0: OpponentBeliefState(),
            1: OpponentBeliefState(),
        }

    def observe_action(
        self,
        player: int,
        action_id: int,
        game: CRGame,
        tick: int,
    ) -> None:
        """Update belief based on an observed opponent action.

        Parameters
        ----------
        player : int
            The player who performed the action (the OPPONENT from our perspective).
        action_id : int
            The action ID performed.
        game : CRGame
            Current game state.
        tick : int
            Current game tick.
        """
        from crsim.cards import CARD_DEFS
        from crsim.constants import ARENA_H, ARENA_W, WAIT_ACTION

        if action_id == WAIT_ACTION:
            return

        card_slot = action_id // (ARENA_W * ARENA_H)
        remainder = action_id % (ARENA_W * ARENA_H)
        x = remainder // ARENA_H
        y = remainder % ARENA_H

        if card_slot >= NUM_HAND_SLOTS:
            return

        ps = game.players[player]
        card_idx = ps.hand[card_slot]
        card_type = ps.deck[card_idx]
        card_def = CARD_DEFS.get(card_type)

        if card_def:
            opp_player = 1 - player
            self.beliefs[opp_player].observe_play(
                card_type=int(card_type),
                cost=card_def.cost,
                tick=tick,
                x=float(x),
                y=float(y),
            )

    def get_belief(self, for_player: int) -> OpponentBeliefState:
        """Get the belief state for a player (about their opponent)."""
        return self.beliefs[for_player]
