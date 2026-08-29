# Dota 2 RL & Meta-Discovery System

Reinforcement-learning research sandbox for Dota 2: train an agent that not only
learns to win but discovers **new meta, power spikes, efficient item/skill
orders, and optimal fight tactics** — in an offline research environment.

> ⚠️ **Scope: sandbox / offline research only.** Automated play in public/ranked
> matches violates Valve Anti-Cheat / ToS and risks a permanent ban. This project
> never touches a live match — local custom lobby / simulator only.
> ([blueprint §0](docs/Blueprint_RL_Dota2_Revisi.md))

Core principle from the spec: *environment correctness first, model complexity
later, measurement-driven scaling.* The MVP therefore stops at curriculum
**F0–F2**, runs on a **custom simulator** ([`dota_rl/env/mock_env.py`](dota_rl/env/mock_env.py))
rather than the Dota client, and treats the real-client path as a gated
follow-up (Milestone M0).

---

## Repository layout

```
.
├── docs/
│   ├── Blueprint_RL_Dota2_Revisi.md                  source blueprint (ID)
│   └── Dota2_RL_Technical_Development_Specification.pdf  engineering spec (24 §)
├── dota_rl/                     the package
│   ├── env/          observation / action / mask contract, reward, mock 1v1 sim
│   ├── agents/       PPO+LSTM masked multi-head policy, attention encoder, embeddings
│   ├── training/     rollout · GAE · PPO (TBPTT) · checkpoint registry · promotion gates
│   ├── evaluation/   Wilson-CI winrate · Elo · regression gate · baselines
│   ├── analytics/    event-aligned value-head meta analysis
│   ├── transport/    Lua↔Python bridge (stub) + protobuf schema
│   └── utils/        config loading, seeding
├── configs/          ppo_f0.yaml · ppo_f1.yaml · ppo_f2.yaml · env.yaml · eval.yaml
├── scripts/          train.py · evaluate.py · meta_report.py · api_probe.lua
├── dota2/scripts/vscripts/bots/   Lua harvester + action handler (stub; mirrors the
│                                  real install path)
├── tests/            25 tests (spec §13)
└── runs/             training output (git-ignored)
```

---

## Current results (mock env, CPU)

| Phase | Config | Promotion gate | Result |
|---|---|---|---|
| **F0** pipeline validation | `configs/ppo_f0.yaml` | ✅ **PASS** | valid transitions 1.000, invalid-action rate 0.000, reward observable |
| **F1** 1v1 basic skill | `configs/ppo_f1.yaml` | ✅ **PASS** | winrate vs scripted **0.75** (CI 0.55–0.88), beats random baseline (0.00), K/D 0/0.5, stable across frozen + rotating seeds |
| **F2** 1v1 full + aggressive bot | `configs/ppo_f2.yaml` | ⚠️ **not yet** | best checkpoint **0.46** winrate (CI low 0.28 < 0.50 bar); **no F1 regression** (F1 held 0.72–0.92 through transfer); build progression ✅ |

F2 needs the §7 exploration tools (RND / league self-play), not yet implemented —
per spec sequencing those come after M6 (eval harness). The gate correctly
reports "not ready" rather than rubber-stamping. F2 training oscillates
(safe-farm vs. fight); the run auto-rolls-back to the best checkpoint before
gating (spec §10 step 12 / §14).

---

## Spec § → module

| Spec area | Module | Status |
|---|---|---|
| §4 Environment contract (obs/action/mask schema, versioning) | `dota_rl/env/protocol.py`, `observation.py`, `action.py`, `masks.py` | ✅ |
| §6 Reward system (z-score + clip + terminal + potential shaping + team-blend τ) | `dota_rl/env/reward.py` | ✅ |
| §0/§5 Custom simulator, curriculum F0–F2, scripted opponent | `dota_rl/env/mock_env.py` | ✅ |
| §5 PPO + LSTM, masked multi-head policy, attention set encoder | `dota_rl/agents/` | ✅ |
| §5.3 Attribute-style hero/item embeddings | `dota_rl/agents/embeddings.py` | ✅ (architectural compat; not zero-shot optimality) |
| §10 Rollout → GAE (terminal/bootstrap masks) → PPO (TBPTT) | `dota_rl/training/` | ✅ |
| §8 Multidimensional promotion gates | `dota_rl/training/promotion.py` | ✅ |
| §9 Eval harness: winrate + Wilson CI + Elo + regression gate + auto-rollback | `dota_rl/evaluation/` | ✅ |
| §12 Checkpoint registry (model/config/git/patch hashes) | `dota_rl/training/checkpointing.py` | ✅ |
| §14 Observability (JSONL + optional TensorBoard) | `dota_rl/training/train.py` | ✅ |
| §15 Event-aligned value-head meta analytics | `dota_rl/analytics/value_events.py` | ✅ |
| §4.3 Curriculum transfer / fine-tune (`train.init_from`) | `dota_rl/training/train.py` | ✅ |
| §7 League self-play / RND | — | ⏳ after M6 (spec sequencing) |
| §3.1 Real Dota adapter + ZMQ bridge + Lua harvester | `dota_rl/env/dota_adapter.py`, `dota_rl/transport/`, `scripts/api_probe.lua`, `dota2/…/bots/*.lua` | ⏳ stub — needs M0 API probe first |

---

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate       # Windows
pip install -r requirements.txt

python scripts/train.py configs/ppo_f0.yaml          # F0: pipeline validation
python scripts/train.py configs/ppo_f1.yaml          # F1: 1v1 basic skill
python scripts/train.py configs/ppo_f2.yaml          # F2: 1v1 full (transfers from F1)

# overrides
python scripts/train.py configs/ppo_f1.yaml --set ppo.lr=1e-4 train.iterations=120

# evaluate a checkpoint on the frozen + rotating benchmark
python scripts/evaluate.py    runs/dota_rl_f1_baseline/checkpoints/ckpt_000140.pt

# event-aligned meta / power-spike report
python scripts/meta_report.py runs/dota_rl_f1_baseline/checkpoints/ckpt_000140.pt

python -m pytest                                     # spec §13 test suite
```

Each run writes `runs/<name>/`: `metrics.jsonl`, `tb/` (TensorBoard),
`checkpoints/` (+ `index.json`), `promotion_gate.txt`.

---

## Acceptance criteria (spec §20)

| ID | Test | Status |
|---|---|---|
| AC-01 connectivity / no deadlock | soak in `test_env.py`, `env/invalid_action_rate` logged | ✅ |
| AC-02 observation correctness | `test_observation.py` (range + missing-data) | ✅ |
| AC-03 action validity > 99% | `env/invalid_action_rate` gate + `test_policy.py` | ✅ (0.000 observed) |
| AC-04 PPO numerical stability | `test_policy.py`, grad-norm logged | ✅ |
| AC-05 learning across ≥3 seeds | rotating-seed eval + F1 gate | ✅ (frozen 0.75 / rotating comparable) |
| AC-06 regression | `evaluation/regression.py` + `test_evaluation.py`; champion/challenger check + auto-rollback | ✅ |
| AC-07 reproducibility | `utils/seeding.py`, `test_env.py` determinism | ✅ |
| AC-08 meta analytics | `analytics/value_events.py`, `meta_report.py` | ✅ |
| AC-09 patch safety | schema versions in `protocol.py`; structural-diff gate | ⏳ pipeline stub |

---

## Roadmap (spec §18)

| Milestone | Deliverable | Status |
|---|---|---|
| M0 | API probe + observation-field access matrix (`scripts/api_probe.lua`) | ⏳ ready to run |
| M1 | F0 environment adapter — 1000+ clean transitions, no deadlock | ✅ (mock) |
| M2 | PPO single-worker — finite losses, measurable learning | ✅ |
| M3 | PPO multi-worker — throughput scaling | ✅ (parallel rollout) |
| M4 | F1 curriculum — basic 1v1 competence beats baseline | ✅ |
| M5 | F2 build learning — stable power-spike experiments | ⚠️ partial |
| M6 | Evaluation harness — automated regression + Elo | ✅ |
| M7 | League self-play — historical pool + champion/challenger | ⏳ |
| M8 | Meta analyzer — event-aligned value analytics | ✅ |
| M9 | Patch pipeline — diff → retrain → regression gate | ⏳ stub |

### Milestone M0 — before any real-client training

`scripts/api_probe.lua` enumerates which observation fields the **public** Bot
Scripting API actually exposes (fogged enemy HP, enemy ability cooldowns, …). If
too many are missing, stay on `mock_env.py`. Only after the probe passes:
implement `dota_rl/transport/zmq_bridge.py` and `dota_rl/env/dota_adapter.py`
against the existing `reset()/step()` contract — the learner does not change.

---

## Design deviations from the source blueprint (spec §22)

- Compact structured observation (~few hundred floats), **not** the ~20k OpenAI Five figure.
- C++/Rust ingestion **deferred** — Python until profiling proves IPC is the bottleneck.
- RND / league **not** enabled from the start — entropy-only baseline first.
- "winrate > 70%" replaced by a multi-metric + confidence-lower-bound gate.
- Value-head meta extraction is **event-aligned & matched**, not raw ΔV.
