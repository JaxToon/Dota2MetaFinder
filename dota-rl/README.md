# dota-rl — Dota 2 RL & Meta-Discovery sandbox (MVP)

Implementation of the *Technical Development Specification — Dota 2 Reinforcement
Learning & Meta-Discovery System* (which itself revises "Blueprint Sistem RL &
Analisis Meta Dota 2").

> **Scope: sandbox / offline research only.** Automated public/ranked play is out
> of scope and unsafe (blueprint §0). This repo never touches a live match.

The spec's core principle is *environment correctness first, model complexity
later, measurement-driven scaling*. So the MVP stops at curriculum **F0–F2**,
runs on a **custom simulator** (`dota_rl/env/mock_env.py`) rather than the Dota
client, and treats the real-client path as a gated follow-up (see *Milestone M0*
below).

## What is built (spec § → module)

| Spec area | Module | Status |
|---|---|---|
| §4 Environment contract (obs/action/mask schema, versioning) | `dota_rl/env/protocol.py`, `observation.py`, `action.py`, `masks.py` | ✅ |
| §6 Reward system (z-score + clip + terminal + team-blend τ) | `dota_rl/env/reward.py` | ✅ |
| §0/§5 Custom simulator, curriculum F0–F2, scripted opponent | `dota_rl/env/mock_env.py` | ✅ |
| §5 PPO + LSTM, masked multi-head policy, attention set encoder | `dota_rl/agents/` | ✅ |
| §5.3 Attribute-style hero/item embeddings | `dota_rl/agents/embeddings.py` | ✅ (architectural compat; not zero-shot optimality) |
| §10 Rollout → GAE (terminal/bootstrap masks) → PPO (TBPTT) | `dota_rl/training/` | ✅ |
| §8 Multidimensional promotion gates | `dota_rl/training/promotion.py` | ✅ |
| §9 Eval harness: winrate + Wilson CI + Elo + regression gate | `dota_rl/evaluation/` | ✅ |
| §12 Checkpoint registry (model/config/git/patch hashes) | `dota_rl/training/checkpointing.py` | ✅ |
| §14 Observability (JSONL + optional TensorBoard) | `dota_rl/training/train.py` | ✅ |
| §15 Event-aligned value-head meta analytics | `dota_rl/analytics/value_events.py` | ✅ |
| §4.3 Curriculum transfer / fine-tune (`train.init_from`) | `dota_rl/training/train.py` | ✅ |
| §7 League self-play / RND | — | ⏳ after M6 (spec sequencing) |
| §3.1 Real Dota adapter + ZMQ bridge + Lua harvester | `dota_rl/env/dota_adapter.py`, `dota_rl/transport/`, `scripts/api_probe.lua`, `dota2/…/bots/*.lua` | ⏳ stub — needs M0 API probe first |

## Current results (mock env, CPU)

| Phase | Config | Promotion gate | Notes |
|---|---|---|---|
| **F0** pipeline validation | `configs/ppo_f0.yaml` | ✅ **PASS** | valid transitions 1.000, invalid-action rate 0.000, reward observable |
| **F1** 1v1 basic skill | `configs/ppo_f1.yaml` | ✅ **PASS** | winrate vs scripted **0.75** (CI 0.55–0.88), beats random baseline (0.00), K/D 0/0.5, stable across frozen+rotating seeds |
| **F2** 1v1 full + aggressive bot | `configs/ppo_f2.yaml` | ⚠️ **not yet** | best checkpoint **0.46** winrate (CI low 0.28 < 0.50 bar); **no F1 regression** (F1 held at 0.72–0.92 through transfer); build progression ✅. Needs the §7 exploration tools (RND / league) or more compute — the gate correctly reports "not ready" rather than rubber-stamping. |

Training is unstable for F2 (policy oscillates between safe-farm and fight); the
run auto-rolls-back to the best checkpoint before gating (spec §10 step 12 / §14).

## Install

```bash
cd dota-rl
python -m venv .venv && . .venv/Scripts/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# F0 — pipeline validation (reset/step correctness, reward observability, no invalid actions)
python scripts/train.py configs/ppo_f0.yaml

# F1 — 1v1 vs scripted laner (last-hit / survival / combat / winrate)
python scripts/train.py configs/ppo_f1.yaml
python scripts/train.py configs/ppo_f1.yaml --set ppo.lr=1e-4 train.iterations=120

# F2 — 1v1 full, aggressive bot; fine-tunes from the F1 checkpoint (transfer)
python scripts/train.py configs/ppo_f2.yaml

# evaluate a checkpoint on the frozen + rotating benchmark
python scripts/evaluate.py runs/dota_rl_f1_baseline/checkpoints/ckpt_000140.pt

# event-aligned meta / power-spike report
python scripts/meta_report.py runs/dota_rl_f1_baseline/checkpoints/ckpt_000140.pt

# tests (spec §13 testing strategy)
python -m pytest
```

Each run writes to `runs/<name>/`: `metrics.jsonl`, `tb/` (TensorBoard),
`checkpoints/` (with `index.json`), `promotion_gate.txt`.

## Acceptance criteria (spec §20) — current MVP coverage

| ID | Test | Where |
|---|---|---|
| AC-01 connectivity / no deadlock | soak in `test_env.py`, `env/invalid_action_rate` logged | ✅ |
| AC-02 observation correctness | `test_observation.py` (range + missing-data) | ✅ |
| AC-03 action validity > 99% | `env/invalid_action_rate` gate + `test_policy.py` | ✅ (0.000 observed) |
| AC-04 PPO numerical stability | `test_policy.py`, grad-norm logged | ✅ |
| AC-05 learning across ≥3 seeds | rotating-seed eval + F1 gate | ✅ (F1 frozen 0.75 / rotating comparable) |
| AC-06 regression | `evaluation/regression.py` + `test_evaluation.py`; champion/challenger check each eval, auto-rollback | ✅ |
| AC-07 reproducibility | `utils/seeding.py`, `test_env.py` determinism | ✅ |
| AC-08 meta analytics | `analytics/value_events.py`, `meta_report.py` | ✅ |
| AC-09 patch safety | schema versions in `protocol.py`; structural-diff gate | ⏳ pipeline stub |

## Milestone M0 — before any real-client training

`scripts/api_probe.lua` enumerates which observation fields the **public** Bot
Scripting API actually exposes (fogged enemy HP, enemy ability cooldowns, …). If
too many are missing, stay on `mock_env.py`. Only after the probe passes:
implement `transport/zmq_bridge.py` and `env/dota_adapter.py` against the
existing `reset()/step()` contract — the learner does not change.

## Design deviations from the source blueprint (spec §22)

- Compact structured observation (~few hundred floats), **not** the ~20k OpenAI
  Five figure.
- C++/Rust ingestion **deferred** — Python until profiling proves IPC is the
  bottleneck.
- RND / league **not** enabled from the start — entropy-only baseline first.
- "winrate > 70%" replaced by a multi-metric + confidence-bound gate.
- Value-head meta extraction is **event-aligned & matched**, not raw ΔV.
