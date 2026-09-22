"""
policy_network.py
==================

Actor-critic policy network used for Phase 1 (trick-maximization)
self-play training, and reused here for inference when generating
the bidding dataset.

Architecture (must match phase1_policy.pt exactly):

    obs (207,) -> Linear(207,256) -> ReLU -> Linear(256,256) -> ReLU
        -> policy_head: Linear(256,52)   (masked categorical logits)
        -> value_head:  Linear(256,1)    (state value)

Author:
    IIT Kanpur RL Card Game Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


OBS_DIM = 207
ACTION_DIM = 52
HIDDEN_DIM = 256


class PolicyNetwork(nn.Module):

    def __init__(
        self,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM
    ):
        super().__init__()

        # Names/indices below (trunk.0, trunk.2) are chosen so that
        # state_dict keys line up exactly with phase1_policy.pt.
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),   # trunk.0
            nn.ReLU(),                        # trunk.1 (no params)
            nn.Linear(hidden_dim, hidden_dim),  # trunk.2
            nn.ReLU(),                        # trunk.3 (no params)
        )

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.trunk(obs)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def masked_logits(self, logits, mask):
        """
        mask: (batch, action_dim) with 1 = legal, 0 = illegal.
        """
        neg_inf = torch.finfo(logits.dtype).min
        return torch.where(mask > 0, logits, torch.full_like(logits, neg_inf))

    @torch.no_grad()
    def act(self, obs, mask, deterministic=False):
        """
        obs:  (batch, 207) float tensor
        mask: (batch, 52) 0/1 tensor

        Returns (action, log_prob, value) each shaped (batch,)
        """
        logits, value = self.forward(obs)
        logits = self.masked_logits(logits, mask)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()

        dist = Categorical(probs)
        log_prob = dist.log_prob(action)

        return action, log_prob, value.squeeze(-1)

    def evaluate(self, obs, mask, actions):
        """
        Used by the PPO update (not needed for dataset generation,
        kept for interface completeness / future fine-tuning).
        """
        logits, value = self.forward(obs)
        logits = self.masked_logits(logits, mask)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, value.squeeze(-1)


def load_policy(checkpoint_path, device="cpu"):
    """
    Convenience loader used by generate_bid_dataset.py.
    """
    policy = PolicyNetwork()
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state_dict, strict=True)
    policy.to(device)
    policy.eval()
    return policy
