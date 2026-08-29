# Dota 2 Meta Finder

Empirical Dota 2 meta analysis from **public match data** (OpenDota). Answers the
draft-and-macro questions:

- **Tier list** — hero win / pick rates by rank bracket, this patch
- **Counters** — hero-vs-hero head-to-head advantage (who beats who)
- **Synergy** — hero-pair win rate *above* what the two heroes do solo
- **Comp scoring** — rate a full 5-stack: meta strength, synergy, role coverage
- **Next-pick recommendation** — given picks / bans so far and your role

> Read-only, official [OpenDota API](https://docs.opendota.com/). No automated
> play, no scraping. Standard-library Python only (`urllib`, `sqlite3`).

The reinforcement-learning sandbox from the original project spec is archived on
the **`archive/rl-sandbox`** branch (tag `rl-sandbox-v1`) — see
[why](#why-not-reinforcement-learning) below.

---

## Quick start

```bash
python scripts/fetch.py                      # warm the cache (~3 min first run)

python scripts/tierlist.py --bracket divine
python scripts/tierlist.py --bracket archon --role Support --top 20

# "we have CM, they have AM + Sniper, I'm mid — what do I pick?"
python scripts/draft.py --allies "Crystal Maiden" --enemies "Anti-Mage,Sniper" --role mid

# rate a full comp
python scripts/draft.py --comp "Mars,Zeus,Juggernaut,Crystal Maiden,Hoodwink"

python scripts/meta.py --bracket divine --json meta.json   # full report

python -m pytest                             # 7 tests, no network (synthetic data)
```

`fetch.py` caches everything under `.cache/` (git-ignored): API responses as
JSON, public matches in `matches.sqlite`. Re-run it to refresh; `heroStats` and
matchups expire after a few hours / days, stored matches never.
A free `OPENDOTA_API_KEY` env var raises the rate limit.

---

## How each number is computed

| Question | Source | Method |
|---|---|---|
| Tier list | `/heroStats` (rolling window, brackets 1–7 = Herald–Divine) | shrinkage-adjusted win rate `(wins + k/2)/(games + k)`, `k≈250`, so a 55 %-on-40-games hero doesn't outrank a 52 %-on-40 000 one; pick share reported alongside |
| Counters | `/heroes/{id}/matchups` | `advantage(a,b) = winRateAdj(a facing b) − 0.5`, shrunk toward 0 (`k≈75`). Positive = a beats b |
| Synergy | `/publicMatches` → local SQLite | for every same-team pair: `pairWinRate − expectedWinRate`, where `expected` combines the two solo win rates on the **log-odds scale** (independent-contribution baseline). Positive = the pair wins more than the sum of its parts |
| Comp score | above | `Σ (meta − 0.5)` + `1.6 · Σ pairwise synergy`, plus attribute / damage-type / role-coverage checks |
| Next pick | above | `w_meta·(metaWin−0.5) + w_syn·synergyWithAllies + w_ctr·advantageVsEnemies − w_vuln·enemiesAdvantageVsYou + w_role·roleFit` |

Everything uses **matched comparisons and shrinkage**, not raw rates — a low-sample
55 % is not treated as better than a high-sample 52 %.

---

## Layout

```
metafinder/
  opendota.py   disk-cached, rate-limited OpenDota client (stdlib)
  heroes.py     hero metadata + tier_list(bracket)
  store.py      SQLite match store + iterators
  counters.py   CounterMatrix — from /matchups or from the store
  synergy.py    SynergyMatrix — solo-strength-adjusted pair win rates
  draft.py      recommend(...) + score_comp(...)
  report.py     assemble + render a full meta report
scripts/        fetch · tierlist · draft · meta
tests/          synthetic-data tests, no network
docs/           original project spec + blueprint (RL approach)
```

---

## Why not reinforcement learning

The original spec proposed *training an RL agent and reverse-engineering the meta
from its behaviour* (the OpenAI Five approach). That was built and works for
**1v1 lane mechanics** (last-hitting, ability timing, positioning) — archived on
`archive/rl-sandbox`. But draft-level meta (5-hero comps, counters, rotations)
needs a **full 5v5 environment with every hero modelled + league self-play at
scale** (~128k CPU / 256 GPU / months). Not feasible on a laptop, and no public
simulator models all ~124 heroes.

Aggregating real match data is how every meta tool actually works, and it runs on
a laptop in minutes. Hence this repo.
