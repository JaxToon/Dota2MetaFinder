# Dota 2 Meta Finder

Empirical Dota 2 meta analysis from **public match data** (OpenDota). Answers the
draft-and-macro questions:

- **Tier list** — hero win / pick rates by rank bracket, this patch
- **Counters** — hero-vs-hero head-to-head advantage (who beats who)
- **Synergy** — hero-pair win rate *above* what the two heroes do solo
- **Comp scoring** — rate a full 5-stack: meta strength, synergy, role coverage
- **Next-pick recommendation** — given picks / bans so far and your role
- **Item build** — the hero's popular progression + situational items that fire
  on the actual enemy/ally heroes (heavy magic → Pipe, hard lockdown → BKB, …)
- **Lane advice** — usual lane + the direct 1v1 reads vs the enemy lineup

> Read-only, official [OpenDota API](https://docs.opendota.com/). No automated
> play, no scraping. Standard-library Python only (`urllib`, `sqlite3`).

The reinforcement-learning sandbox from the original project spec is archived on
the **`archive/rl-sandbox`** branch (tag `rl-sandbox-v1`) — see
[why](#why-not-reinforcement-learning) below.

---

## Interactive draft coach

**[Live tool](https://jaxtoon.github.io/Dota2MetaFinder/)** — enter both
lineups, get hero picks + an item build + lane advice, all reacting to the
specific enemy/ally heroes. Rebuild it with your own fresh data:

```bash
python scripts/fetch.py          # heroStats + matchups + ~20k public matches (~6 min once)
python scripts/build_web.py      # + itemPopularity/laneRoles per hero -> web/index.html
# open web/index.html; the live copy is served from the gh-pages branch
```

## Command line

```bash
python scripts/tierlist.py --bracket divine
python scripts/tierlist.py --bracket archon --role Support --top 20

# full coach: picks + build + lane
python scripts/coach.py --allies "Crystal Maiden,Tidehunter" \
    --enemies "Lion,Lina,Zeus,Sniper,Bane" --hero Juggernaut --role carry

# picks only
python scripts/coach.py --enemies "Sniper,Drow Ranger"

python scripts/draft.py --comp "Mars,Zeus,Juggernaut,Crystal Maiden,Hoodwink"   # rate a comp
python scripts/meta.py --bracket divine --json meta.json                        # tier/counter/synergy report

python -m pytest                  # 13 tests, no network (synthetic data)
```

`fetch.py` caches everything under `.cache/` (git-ignored): API responses as
JSON, public matches in `matches.sqlite`. Re-run it to refresh; `heroStats` and
matchups expire after a few hours / days, stored matches never.
A free `OPENDOTA_API_KEY` env var raises the rate limit.

**Sample output** (Divine bracket, ~26k matches):
[Meta Snapshot](https://jaxtoon.github.io/Dota2MetaFinder/meta.html)
— tier list, counters, synergy pairs.

- Top of the tier list: Bounty Hunter (56%), Phantom Lancer, OD, Spectre, Lifestealer
- Hard counters: Treant > Invoker (+0.12), Earthshaker > Sven (+0.11), Nyx/Phoenix/Ember > Void Spirit
- Synergy pairs: Invoker + Skywrath, Tidehunter + Luna, CM + Earth Spirit

> Pair-level stats (counters, synergy) are noisy at 26k matches — run
> `scripts/fetch.py --matches 200000` for firm numbers.

---

## How each number is computed

| Question | Source | Method |
|---|---|---|
| Tier list | `/heroStats` (rolling window, brackets 1–7 = Herald–Divine) | shrinkage-adjusted win rate `(wins + k/2)/(games + k)`, `k≈250`, so a 55 %-on-40-games hero doesn't outrank a 52 %-on-40 000 one; pick share reported alongside |
| Counters | `/heroes/{id}/matchups` | `advantage(a,b) = winRateAdj(a facing b) − 0.5`, shrunk toward 0 (`k≈75`). Positive = a beats b |
| Synergy | `/publicMatches` → local SQLite | for every same-team pair: `pairWinRate − expectedWinRate`, where `expected` combines the two solo win rates on the **log-odds scale** (independent-contribution baseline). Positive = the pair wins more than the sum of its parts |
| Comp score | above | `Σ (meta − 0.5)` + `1.6 · Σ pairwise synergy`, plus attribute / damage-type / role-coverage checks |
| Next pick | above | `w_meta·(metaWin−0.5) + w_syn·synergyWithAllies + w_ctr·advantageVsEnemies − w_vuln·enemiesAdvantageVsYou + w_role·roleFit` |
| Item build | `/heroes/{id}/itemPopularity` + a rules layer | **base**: the hero's most-bought items per game stage. **situational**: `metafinder/heroprops.py` tags each hero (magic / lockdown / invis / evasion / heal / …); rules count those tags on the enemy & ally side you entered and fire "consider Pipe / BKB / sentries / MKB / Blink". Not per-matchup mined data — the *trigger* is the specific lineup, the *item* is a Dota fundamental |
| Lane | `/scenarios/laneRoles` + counter matrix | base: the hero's most-played lane and its win rate. Lineup-aware: the direct 1v1 advantage vs each enemy, plus trilane / roam warnings from enemy disable & mobility tags |

Everything data-derived uses **matched comparisons and shrinkage**, not raw rates
— a low-sample 55 % is not treated as better than a high-sample 52 %. The build /
lane *rules* are hand-written Dota knowledge; they react to your lineup but they
are not themselves fit from data.

---

## Layout

```
metafinder/
  opendota.py    disk-cached, rate-limited OpenDota client (stdlib)
  heroes.py      hero metadata + tier_list(bracket)
  store.py       SQLite match store + iterators
  counters.py    CounterMatrix -- from /matchups or from the store
  synergy.py     SynergyMatrix -- solo-strength-adjusted pair win rates
  heroprops.py   curated threat/need tags -- the knowledge layer
  builds.py      core_build(hero) + situational_items(hero, allies, enemies)
  lanes.py       lane_roles(hero) + lane_advice(...)
  draft.py       recommend(...) + score_comp(...)
  report.py      assemble + render the meta report
scripts/         fetch · build_web · coach · tierlist · draft · meta
web/             template.html (the app) ; index.html + data.json are generated
tests/           13 synthetic-data tests, no network
docs/            original project spec + blueprint (RL approach)
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
