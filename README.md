# Dota 2 RL & Meta-Discovery System

Riset reinforcement learning untuk Dota 2: melatih agen yang tidak hanya belajar
menang, tapi menemukan **meta baru, power spike, urutan item/skill efisien, dan
taktik bertarung optimal** — di lingkungan sandbox/offline.

> ⚠️ **Scope: riset sandbox / offline saja.** Bermain otomatis di match
> publik/ranked melanggar Valve Anti-Cheat/ToS dan berisiko permanent ban.
> Proyek ini **tidak pernah** menyentuh match live — hanya custom lobby lokal /
> simulator. (Blueprint §0)

---

## Isi repo

| Path | Isi |
|---|---|
| [`Blueprint_RL_Dota2_Revisi.md`](Blueprint_RL_Dota2_Revisi.md) | Blueprint awal (Bahasa Indonesia): konsep sistem, tech stack, curriculum, reward shaping, embedding ketahanan patch, evaluasi. |
| [`Dota2_RL_Technical_Development_Specification.pdf`](Dota2_RL_Technical_Development_Specification.pdf) | Spesifikasi teknis engineering-ready (24 bagian): kontrak environment, arsitektur model, protokol training, interface, dan acceptance criteria — turunan dari blueprint. |
| [`dota-rl/`](dota-rl/) | **Implementasi MVP** dari spesifikasi tersebut (Python/PyTorch). Lihat [`dota-rl/README.md`](dota-rl/README.md) untuk detail lengkap. |

---

## Status implementasi (`dota-rl/`)

Prinsip spec: *environment correctness dulu, kompleksitas model belakangan,
scaling berbasis pengukuran*. MVP berhenti di curriculum **F0–F2**, jalan di
**simulator kustom** (`dota_rl/env/mock_env.py`), bukan client Dota 2 — jalur
client asli adalah follow-up bergating (Milestone M0).

| Fase | Promotion gate | Hasil (mock env, CPU) |
|---|---|---|
| **F0** — validasi pipeline | ✅ **PASS** | valid transitions 1.000, invalid-action rate 0.000, reward observable |
| **F1** — 1v1 basic skill | ✅ **PASS** | winrate vs bot scripted **0.75** (CI 0.55–0.88), kalahkan baseline random (0.00), K/D 0/0.5, stabil di frozen + rotating seeds |
| **F2** — 1v1 penuh + bot agresif | ⚠️ **belum lolos** | checkpoint terbaik **0.46** winrate (CI low 0.28 < bar 0.50); **F1 tidak regresi** (tetap 0.72–0.92 lewat transfer); build progression ✅ |

25 unit test lulus (`pytest`). Detail acceptance criteria AC-01…AC-09 ada di
[`dota-rl/README.md`](dota-rl/README.md).

**F2 belum lolos** karena butuh exploration tools §7 (RND / league self-play)
yang belum diimplementasi — per urutan spec, itu datang setelah M6 (eval
harness). Gate-nya benar melaporkan "belum siap", bukan asal lolos.

---

## Komponen yang sudah dibangun (spec § → modul)

- **§4 Environment contract** — schema observation/action/mask + versioning (`dota_rl/env/`)
- **§6 Reward system** — z-score + clip + terminal + potential shaping + team-blend τ (`dota_rl/env/reward.py`)
- **§0/§5 Simulator kustom + curriculum F0–F2** — 1v1 lane sim, opponent scripted (`dota_rl/env/mock_env.py`)
- **§5 PPO + LSTM** — masked multi-head policy, attention set encoder (`dota_rl/agents/`)
- **§5.3 Embedding atribut** hero/item (kompatibilitas arsitektur, bukan jaminan zero-shot)
- **§10 Rollout → GAE → PPO (TBPTT)** dengan terminal/bootstrap mask (`dota_rl/training/`)
- **§8 Promotion gates multidimensi** (`dota_rl/training/promotion.py`)
- **§9 Eval harness** — winrate + Wilson CI + Elo + regression gate + auto-rollback (`dota_rl/evaluation/`)
- **§12 Checkpoint registry** — hash model/config/git/patch (`dota_rl/training/checkpointing.py`)
- **§14 Observability** — JSONL + TensorBoard opsional
- **§15 Meta analytics** — analisis Value-Head event-aligned & matched (`dota_rl/analytics/`)
- **§4.3 Curriculum transfer** — fine-tune antar fase (`train.init_from`)

**Belum:** §7 league/RND · §3.1 adapter Dota asli + ZMQ bridge + Lua harvester
(stub — butuh probe API M0 dulu).

---

## Quick start

```bash
cd dota-rl
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

python scripts/train.py configs/ppo_f0.yaml         # F0: validasi pipeline
python scripts/train.py configs/ppo_f1.yaml         # F1: 1v1 basic skill
python scripts/train.py configs/ppo_f2.yaml         # F2: 1v1 penuh (transfer dari F1)

python scripts/evaluate.py    runs/dota_rl_f1_baseline/checkpoints/ckpt_000140.pt
python scripts/meta_report.py runs/dota_rl_f1_baseline/checkpoints/ckpt_000140.pt
python -m pytest
```

Tiap run menulis ke `runs/<name>/`: `metrics.jsonl`, `tb/`, `checkpoints/`
(+ `index.json`), `promotion_gate.txt`.

---

## Roadmap (spec §18)

| Milestone | Deliverable | Status |
|---|---|---|
| M0 | API probe + matriks akses field observation (`scripts/api_probe.lua`) | ⏳ stub siap dijalankan |
| M1 | F0 environment adapter — 1000+ transisi bersih, no deadlock | ✅ (mock) |
| M2 | PPO single-worker — loss finite, learning terukur | ✅ |
| M3 | PPO multi-worker — throughput scaling | ✅ (rollout paralel) |
| M4 | F1 curriculum — kompetensi 1v1 dasar mengalahkan baseline | ✅ |
| M5 | F2 build learning — eksperimen power-spike stabil | ⚠️ parsial |
| M6 | Evaluation harness — regression + Elo otomatis | ✅ |
| M7 | League self-play — historical pool + champion/challenger | ⏳ |
| M8 | Meta analyzer — event-aligned value analytics | ✅ |
| M9 | Patch pipeline — diff → retrain → regression gate | ⏳ stub |

---

## Keputusan desain (menyimpang dari blueprint awal — spec §22)

- Observation kompak (~ratusan float), **bukan** ~20.000 (angka referensi OpenAI Five untuk 5v5).
- Ingestion C++/Rust **ditunda** sampai profiling membuktikan IPC jadi bottleneck.
- RND / league **tidak** diaktifkan dari awal — baseline entropy-only dulu.
- "winrate > 70%" diganti gate multi-metrik + confidence-lower-bound.
- Ekstraksi meta dari value-head pakai **analisis event-aligned & matched**, bukan raw ΔV.
