"""PPO update over recurrent rollouts (spec 10, 10.1).

Minibatches are taken over *environment* indices so each sampled sequence stays
intact in time for truncated BPTT. Advantages are normalised after GAE; the
value head is clipped; approx-KL / clip-fraction / explained-variance are logged
for the training-health dashboard (spec 14).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .gae import compute_gae


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    epochs: int = 4
    num_minibatches: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    target_kl: float | None = 0.03
    entropy_anneal: bool = True
    entropy_coef_final: float = 0.002


class PPOTrainer:
    def __init__(self, policy: nn.Module, cfg: PPOConfig, device="cpu"):
        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
        self._progress = 0.0  # 0..1 across training, for entropy anneal

    def set_progress(self, p: float):
        self._progress = float(np.clip(p, 0.0, 1.0))

    @property
    def entropy_coef(self) -> float:
        c = self.cfg
        if not c.entropy_anneal:
            return c.entropy_coef
        return c.entropy_coef + (c.entropy_coef_final - c.entropy_coef) * self._progress

    def update(self, batch: dict) -> dict:
        cfg = self.cfg
        obs, actions = batch["obs"], batch["actions"]
        done_seq = batch["dones"]
        T, B = done_seq.shape

        adv, returns = compute_gae(
            batch["rewards"], batch["values"], batch["terminated"], batch["truncated"],
            batch["last_value"], cfg.gamma, cfg.gae_lambda,
        )
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        old_logp = batch["logp"]
        old_values = batch["values"]

        h0, c0 = batch["state0"]
        mb = max(1, B // cfg.num_minibatches)
        metrics = {k: [] for k in
                   ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm"]}
        stop = False
        for _ in range(cfg.epochs):
            if stop:
                break
            perm = torch.randperm(B)
            for s in range(0, B, mb):
                idx = perm[s:s + mb]
                state0 = (h0[idx], c0[idx])
                obs_mb = {k: (obs[k][:, idx] if k != "masks"
                              else {m: obs["masks"][m][:, idx] for m in obs["masks"]})
                          for k in obs}
                act_mb = {k: actions[k][:, idx] for k in actions}

                new_logp, entropy, value = self.policy.evaluate_sequence(
                    obs_mb, done_seq[:, idx], state0, act_mb
                )
                a = adv_norm[:, idx]
                ratio = (new_logp - old_logp[:, idx]).exp()
                s1 = ratio * a
                s2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * a
                policy_loss = -torch.min(s1, s2).mean()

                v_clip = old_values[:, idx] + torch.clamp(
                    value - old_values[:, idx], -cfg.clip_range, cfg.clip_range
                )
                vl = torch.max((value - returns[:, idx]) ** 2,
                               (v_clip - returns[:, idx]) ** 2).mean()

                ent = entropy.mean()
                loss = policy_loss + cfg.value_coef * vl - self.entropy_coef * ent

                self.opt.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    logratio = new_logp - old_logp[:, idx]
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clip_frac = ((ratio - 1).abs() > cfg.clip_range).float().mean().item()
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(vl.item())
                metrics["entropy"].append(ent.item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_frac"].append(clip_frac)
                metrics["grad_norm"].append(float(gn))

            if cfg.target_kl is not None and np.mean(metrics["approx_kl"][-cfg.num_minibatches:]) > 1.5 * cfg.target_kl:
                stop = True

        with torch.no_grad():
            y = returns.reshape(-1)
            yhat = batch["values"].reshape(-1)
            var_y = y.var()
            explained_var = float("nan") if var_y == 0 else float(1 - (y - yhat).var() / var_y)

        out = {k: float(np.mean(v)) for k, v in metrics.items() if v}
        out["explained_variance"] = explained_var
        out["entropy_coef"] = self.entropy_coef
        out["advantage_mean"] = float(adv.mean())
        out["return_mean"] = float(returns.mean())
        return out
