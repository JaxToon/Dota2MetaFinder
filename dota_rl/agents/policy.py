"""PPO + LSTM actor-critic with a masked multi-head action policy (spec 5.1).

Heads: action_type, target_entity, ability_slot, item_slot (masked categoricals)
and target_xy (diagonal Gaussian). The joint log-prob only sums the sub-heads
that the sampled action_type actually consumes (spec: hierarchical action
grammar, not one monolithic categorical). The coordinate head is scored as a
pre-squash Gaussian; the env applies the [0,1] clamp.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..env.protocol import (
    ACTION_USES_ABILITY,
    ACTION_USES_COORD,
    ACTION_USES_ENTITY,
    ACTION_USES_ITEM,
    NUM_ABILITIES,
    NUM_ACTION_TYPES,
    NUM_ENTITY_SLOTS,
    NUM_ITEMS,
    ActionType,
)
from .distributions import DiagGaussian, MaskedCategorical
from .encoder import ObsEncoder
from .recurrent import RecurrentCore


def _usage_vec(mapping) -> torch.Tensor:
    return torch.tensor([float(mapping[ActionType(i)]) for i in range(NUM_ACTION_TYPES)])


class ActorCritic(nn.Module):
    def __init__(self, encoder_dim: int = 256, lstm_hidden: int = 256):
        super().__init__()
        self.encoder = ObsEncoder(out_dim=encoder_dim)
        self.core = RecurrentCore(encoder_dim, lstm_hidden)

        self.head_type = nn.Linear(lstm_hidden, NUM_ACTION_TYPES)
        self.head_entity = nn.Linear(lstm_hidden, NUM_ENTITY_SLOTS)
        self.head_ability = nn.Linear(lstm_hidden, NUM_ABILITIES)
        self.head_item = nn.Linear(lstm_hidden, NUM_ITEMS)
        self.head_coord_mean = nn.Linear(lstm_hidden, 2)
        self.coord_log_std = nn.Parameter(torch.zeros(2) - 0.5)
        self.head_value = nn.Linear(lstm_hidden, 1)

        self.register_buffer("uses_entity", _usage_vec(ACTION_USES_ENTITY))
        self.register_buffer("uses_ability", _usage_vec(ACTION_USES_ABILITY))
        self.register_buffer("uses_item", _usage_vec(ACTION_USES_ITEM))
        self.register_buffer("uses_coord", _usage_vec(ACTION_USES_COORD))

    # ------------------------------------------------------------------ dists
    def _dists(self, feat: torch.Tensor, masks: dict):
        coord_mean = torch.sigmoid(self.head_coord_mean(feat))
        return {
            "type": MaskedCategorical(self.head_type(feat), masks["action_type"]),
            "entity": MaskedCategorical(self.head_entity(feat), masks["entity"]),
            "ability": MaskedCategorical(self.head_ability(feat), masks["ability"]),
            "item": MaskedCategorical(self.head_item(feat), masks["item"]),
            "coord": DiagGaussian(coord_mean, self.coord_log_std.expand_as(coord_mean)),
        }

    def _joint_logp(self, dists, actions, used):
        lp = dists["type"].log_prob(actions["type"])
        lp = lp + used["entity"] * dists["entity"].log_prob(actions["entity"])
        lp = lp + used["ability"] * dists["ability"].log_prob(actions["ability"])
        lp = lp + used["item"] * dists["item"].log_prob(actions["item"])
        lp = lp + used["coord"] * dists["coord"].log_prob(actions["coord"])
        return lp

    def _entropy(self, dists):
        tp = dists["type"].probs
        ent = dists["type"].entropy()
        ent = ent + (tp * self.uses_entity).sum(-1) * dists["entity"].entropy()
        ent = ent + (tp * self.uses_ability).sum(-1) * dists["ability"].entropy()
        ent = ent + (tp * self.uses_item).sum(-1) * dists["item"].entropy()
        ent = ent + (tp * self.uses_coord).sum(-1) * dists["coord"].entropy()
        return ent

    def _used_from_type(self, a_type: torch.Tensor) -> dict:
        return {
            "entity": self.uses_entity[a_type],
            "ability": self.uses_ability[a_type],
            "item": self.uses_item[a_type],
            "coord": self.uses_coord[a_type],
        }

    # ------------------------------------------------------------------- rollout
    @torch.no_grad()
    def act(self, obs: dict, state, deterministic: bool = False):
        feat = self.encoder(obs)
        h, new_state = self.core.step(feat, state)
        dists = self._dists(h, obs["masks"])
        if deterministic:
            a_type = dists["type"].probs.argmax(-1)
            actions = {
                "type": a_type,
                "entity": dists["entity"].probs.argmax(-1),
                "ability": dists["ability"].probs.argmax(-1),
                "item": dists["item"].probs.argmax(-1),
                "coord": dists["coord"].mean,
            }
        else:
            a_type = dists["type"].sample()
            actions = {
                "type": a_type,
                "entity": dists["entity"].sample(),
                "ability": dists["ability"].sample(),
                "item": dists["item"].sample(),
                "coord": dists["coord"].sample(),
            }
        used = self._used_from_type(a_type)
        logp = self._joint_logp(dists, actions, used)
        value = self.head_value(h).squeeze(-1)
        return actions, logp, value, new_state

    @torch.no_grad()
    def value_only(self, obs: dict, state):
        feat = self.encoder(obs)
        h, _ = self.core.step(feat, state)
        return self.head_value(h).squeeze(-1)

    # ---------------------------------------------------------------- ppo update
    def evaluate_sequence(self, obs_seq: dict, done_seq: torch.Tensor,
                          state0, actions_seq: dict):
        """obs_seq / actions_seq: (T, B, ...). Returns logp, entropy, value (T, B)."""
        T, B = done_seq.shape
        flat = {k: obs_seq[k].reshape(T * B, *obs_seq[k].shape[2:])
                for k in obs_seq if k != "masks"}
        feat = self.encoder(flat).reshape(T, B, -1)
        outs, _ = self.core.forward_sequence(feat, done_seq, state0)  # (T, B, H)

        masks = {k: obs_seq["masks"][k].reshape(T * B, -1) for k in obs_seq["masks"]}
        h = outs.reshape(T * B, -1)
        dists = self._dists(h, masks)
        acts = {k: actions_seq[k].reshape(T * B, *actions_seq[k].shape[2:]) for k in actions_seq}
        used = self._used_from_type(acts["type"])
        logp = self._joint_logp(dists, acts, used).reshape(T, B)
        entropy = self._entropy(dists).reshape(T, B)
        value = self.head_value(h).squeeze(-1).reshape(T, B)
        return logp, entropy, value
