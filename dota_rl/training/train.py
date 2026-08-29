"""Training loop -- exact engineering sequence from spec section 10.

  load config -> seed -> env pool -> policy -> [collect -> GAE -> PPO ->
  quick eval -> log -> checkpoint -> regression vs champion] -> promotion gate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..agents.policy import ActorCritic
from ..env.mock_env import MockDotaEnv
from ..env.reward import RewardConfig
from ..evaluation.harness import EvalConfig, evaluate_policy
from ..evaluation.regression import check_regression
from ..utils.config import pick
from ..utils.seeding import seed_all
from .checkpointing import CheckpointRegistry
from .league import League, LeagueConfig
from .ppo import PPOConfig, PPOTrainer
from .promotion import check_f0, check_f1
from .rollout import RolloutCollector


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w")
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(self.path.parent / "tb"))
        except Exception:
            self.tb = None

    def log(self, step: int, metrics: dict):
        rec = {"step": step, **{k: _num(v) for k, v in metrics.items()}}
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        if self.tb:
            for k, v in rec.items():
                if isinstance(v, (int, float)) and k != "step":
                    self.tb.add_scalar(k, v, step)

    def close(self):
        self.f.close()
        if self.tb:
            self.tb.close()


def _num(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def make_reward_cfg(cfg: dict) -> RewardConfig:
    r = cfg.get("reward", {})
    base = RewardConfig()
    w = r.get("weights", {})
    return RewardConfig(
        w_eco=w.get("eco", base.w_eco), w_xp=w.get("xp", base.w_xp),
        w_com=w.get("com", base.w_com), w_obj=w.get("obj", base.w_obj),
        w_str=w.get("str", base.w_str), w_term=w.get("term", base.w_term),
        clip=r.get("clip", 5.0),
        team_blend_tau=r.get("team_blend_tau", 0.0),
        terminal_win=r.get("terminal_win", 10.0),
        normalize=r.get("normalize", True),
        enabled=r.get("enabled", {"eco": True, "xp": True, "com": True,
                                  "obj": False, "str": True, "term": True}),
    )


def train(cfg: dict, run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    seed_all(seed)
    # device: "cpu" (default) | "cuda" | "auto". At MVP scale the Python env sim
    # is the bottleneck, not the ~2M-param net -- CUDA is ~a wash and pays
    # per-step transfer overhead in the rollout (see README benchmark). Worth
    # flipping to cuda only with a much larger model / batch (F4-F5).
    device = str(cfg.get("train", {}).get("device", "cpu")).lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("train.device=cuda but no CUDA device found -> falling back to cpu")
        device = "cpu"
    print(f"device: {device}")

    phase = cfg.get("experiment", {}).get("curriculum_phase", "F1")
    patch_id = cfg.get("experiment", {}).get("patch_id", "sandbox_mock_v1")
    reward_cfg = make_reward_cfg(cfg)
    env_cfg = cfg.get("environment", {})
    num_envs = int(env_cfg.get("num_envs", 16))
    horizon = int(cfg.get("train", {}).get("horizon", 128))
    iterations = int(cfg.get("train", {}).get("iterations", 40))
    eval_every = int(cfg.get("train", {}).get("eval_every", 10))

    def make_env(i: int) -> MockDotaEnv:
        return MockDotaEnv(phase=phase, reward_cfg=reward_cfg, seed=seed * 7919 + i,
                           patch_id=patch_id,
                           max_steps=int(env_cfg.get("max_steps", 400)))

    model_cfg = cfg.get("model", {})
    policy = ActorCritic(
        encoder_dim=int(model_cfg.get("encoder_dim", 256)),
        lstm_hidden=int(model_cfg.get("lstm_hidden", 256)),
    ).to(device)

    # transfer / fine-tune from an earlier curriculum phase (spec 10 step 4, 4.3)
    init_from = cfg.get("train", {}).get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location=device, weights_only=False)
        missing, unexpected = policy.load_state_dict(blob["policy"], strict=False)
        print(f"initialised from {init_from} (missing={len(missing)}, unexpected={len(unexpected)})")

    ppo_cfg = pick(cfg, "ppo", PPOConfig)
    trainer = PPOTrainer(policy, ppo_cfg, device)

    # self-play / league (spec section 7). Off unless league.enabled.
    league = None
    lg_cfg = pick(cfg, "league", LeagueConfig)
    if lg_cfg.enabled:
        league = League(lg_cfg, model_cfg, num_envs, device, seed=seed)
        league.refresh_self(policy)
        league.assign_all()
        print(f"self-play league ON  (self {lg_cfg.ratio_self} / hist {lg_cfg.ratio_history} "
              f"/ scripted {lg_cfg.ratio_scripted}, pool<= {lg_cfg.pool_capacity})")

    collector = RolloutCollector(make_env, num_envs, horizon, device, league=league)
    registry = CheckpointRegistry(run_dir / "checkpoints")
    logger = JsonlLogger(run_dir / "metrics.jsonl")

    eval_cfg = EvalConfig(phase=phase,
                          games_per_opponent=int(cfg.get("evaluation", {}).get("games_per_opponent", 20)))

    champion_eval = None
    champion_ckpt = None
    champion_wr = -1.0
    invalid_actions = 0
    total_actions = 0
    t0 = time.time()

    for it in range(1, iterations + 1):
        trainer.set_progress(it / iterations)
        batch = collector.collect(policy)

        # environment-health accounting (spec 9.1 tier 1 / AC-03)
        acts = batch["actions"]
        masks = batch["obs"]["masks"]
        legal = masks["action_type"].gather(-1, acts["type"].unsqueeze(-1).long()).squeeze(-1)
        invalid_actions += int((~legal.bool()).sum().item())
        total_actions += legal.numel()

        stats = trainer.update(batch)
        ep = collector.drain_episode_stats()

        league_row = {}
        if league is not None:
            league.refresh_self(policy)              # opponent tracks the learner
            if league.maybe_snapshot(policy, it):
                print(f"      + league snapshot (pool size {len(league.pool)})")
            league_row = league.summary()

        sps = (horizon * num_envs * it) / max(1e-9, time.time() - t0)
        row = {**stats, **{f"rollout/{k}": v for k, v in ep.items()}, **league_row,
               "env/invalid_action_rate": invalid_actions / max(1, total_actions),
               "perf/env_steps_per_sec": sps}
        logger.log(it, row)
        print(f"it {it:4d} | R {ep['return_mean']:+.2f} | wr {ep['winrate']:.2f} "
              f"| KL {stats.get('approx_kl', 0):.4f} | ent {stats.get('entropy', 0):.3f} "
              f"| EV {stats.get('explained_variance', float('nan')):.2f} | sps {sps:.0f}")

        if it % eval_every == 0 or it == iterations:
            ev = evaluate_policy(policy, eval_cfg, device=device, reward_cfg=reward_cfg)
            s = ev["vs_scripted_bot"]
            logger.log(it, {"eval/winrate": s["winrate"],
                            "eval/winrate_ci_low": s["winrate_ci_low"],
                            "eval/kills": s["kills_mean"], "eval/deaths": s["deaths_mean"],
                            "eval/random_winrate": ev["random_vs_scripted_bot"]["winrate"]})
            entry = registry.save(policy, trainer.opt, iteration=it, config=cfg,
                                  patch_id=patch_id, curriculum_phase=phase, metrics=s)
            if champion_eval is not None:
                reg = check_regression(champion_eval, ev,
                                       threshold=cfg.get("evaluation", {}).get("regression_threshold", 0.05))
                print(f"      regression check: {'PASS' if reg.passed else 'FAIL'} -- {reg.reason}")
                if reg.passed:
                    champion_eval = ev
                else:
                    print("      -> challenger quarantined; champion unchanged")
            else:
                champion_eval = ev
            # keep the best checkpoint by winrate for the promotion gate / rollback
            if s["winrate"] > champion_wr:
                champion_wr = s["winrate"]
                champion_ckpt = entry["path"]

    # roll back to the best checkpoint before gating (spec 10 step 12, 14)
    if champion_ckpt and champion_ckpt != registry.latest()["path"]:
        registry.load(champion_ckpt, policy)
        print(f"rolled back to champion checkpoint {champion_ckpt} (winrate {champion_wr:.3f})")

    # ---- promotion gate --------------------------------------------------
    final_eval = evaluate_policy(policy, eval_cfg, device=device, reward_cfg=reward_cfg)
    rot_eval = evaluate_policy(policy, eval_cfg, device=device, reward_cfg=reward_cfg, rotating=True)
    inv_rate = invalid_actions / max(1, total_actions)
    if phase == "F0":
        gate = check_f0({
            "valid_transition_rate": 1.0 - inv_rate,
            "protocol_deadlocks": 0,
            "reward_observed": abs(final_eval["vs_scripted_bot"]["net_worth_mean"]) > 0,
            "invalid_action_rate": inv_rate,
        })
    elif phase == "F2":
        from ..training.promotion import check_f2
        f1_eval = evaluate_policy(policy, EvalConfig(phase="F1", games_per_opponent=eval_cfg.games_per_opponent),
                                  device=device, reward_cfg=reward_cfg)
        f1_prev = cfg.get("train", {}).get("f1_reference_winrate")
        gate = check_f2(final_eval, f1_eval,
                        rot_eval["random_vs_scripted_bot"]["winrate"], f1_prev)
    else:
        seeds_stable = abs(final_eval["vs_scripted_bot"]["winrate"]
                           - rot_eval["vs_scripted_bot"]["winrate"]) < 0.25
        gate = check_f1(final_eval, rot_eval["random_vs_scripted_bot"]["winrate"], seeds_stable)
    print(gate)
    (run_dir / "promotion_gate.txt").write_text(str(gate))
    logger.log(iterations, {"gate/passed": int(gate.passed)})
    logger.close()

    return {
        "final_eval": final_eval,
        "rotating_eval": rot_eval,
        "invalid_action_rate": inv_rate,
        "promotion_passed": gate.passed,
        "checkpoint": registry.latest(),
    }
