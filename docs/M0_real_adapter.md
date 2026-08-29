# Milestone M0 — connecting to the real Dota 2 sandbox

Everything below the learner is already built and tested against an in-process
fake game loop (`tests/test_adapter.py`). Three manual steps remain, in order.
If step 1 shows the public Bot API can't expose what we need, **stop and stay on
`MockDotaEnv`** — that is the spec's designated fallback (blueprint §0).

> Sandbox / offline only. Local custom lobby, bot-vs-bot. Never a ranked/public
> match.

---

## Step 1 — API probe (decides everything)

1. Copy the probe into the game's bot-script folder:
   ```
   scripts/api_probe.lua
     -> <SteamLibrary>/steamapps/common/dota 2 beta/game/dota/scripts/vscripts/bots/
   ```
2. Steam → Dota 2 → Properties → Launch Options:
   ```
   -dev -vconsole +dota_force_gamemode 1
   ```
3. Create a **LOCAL lobby**, add bots, start. Open the console (`` ` ``).
4. The probe prints an access matrix and writes `api_probe_result.json` (next to
   the Dota executable's working dir). It checks the fields that matter:
   - own HP / mana / gold / level / position / facing — expected OK
   - **ability cooldowns / mana cost** — needed for the ability head
   - **enemy HP while fogged**, **enemy ability cooldowns**, **enemy position
     when not visible** — these are the risky ones (OpenAI Five had a private
     API for full state)
   - `GetUnitList(ENEMY_CREEPS)`, `GameTime()` — needed for last-hit + timing

**Decision:**
| probe result | action |
|---|---|
| own state + creep list + ability cd all OK, most enemy/fog fields OK | proceed to step 2 |
| enemy/fog fields MISSING | keep training on `MockDotaEnv`; treat fog as `visible=false` / masked (the observation contract already represents this honestly) |

---

## Step 2 — finish the Lua bot

`dota2/scripts/vscripts/bots/bot_generic.lua` (harvester) and
`rl_action_handler.lua` (executor) are skeletons. Fill in:

- **`build_observation(bot)`** → emit exactly the JSON `schemas.proto`
  `Observation` describes, with the fixed 8-slot entity layout
  (`0` = enemy hero, `1..4` = lane creeps, `5` = enemy tower, `6` = own tower).
  The Python parser (`dota_rl/env/dota_adapter.py :: parse_wire_observation`)
  already maps those field names — only change it if the probe shows the API
  uses different names.
- **transport**: `require` a Lua ZMQ (or a raw TCP socket) client. Each
  `Think()`: send `{"type":"obs", "step_id", "schema_version":1, "observation":…,
  "reward", "terminated", "truncated"}`, block for the reply, hand the `action`
  to `rl_action_handler.apply(bot, action)`.
- **reward**: compute the spec-§6 components engine-side (Δ net worth, kill/death
  events, tower damage, terminal) OR send raw quantities and let
  `dota_rl/env/reward.py` shape them (preferred — keeps weights in config).

The Python REP server is `dota_rl/transport/zmq_bridge.py :: ZmqBridge`
(`pip install pyzmq`). It already does schema-version + step-id order checks.

---

## Step 3 — point the learner at the adapter

```python
from dota_rl.transport.zmq_bridge import ZmqBridge, BridgeConfig
from dota_rl.env.dota_adapter import DotaSandboxEnv

bridge = ZmqBridge(BridgeConfig(endpoint="tcp://127.0.0.1:5555"))
env = DotaSandboxEnv(bridge=bridge)          # same reset()/step() as MockDotaEnv
```

Wire `configs/env.yaml : environment.backend: dota_sandbox` and have
`scripts/train.py` build `DotaSandboxEnv` instead of `MockDotaEnv` when
`backend == "dota_sandbox"`. **Nothing in `dota_rl/agents/` or
`dota_rl/training/` changes** — that is the whole point of the contract.

Start with **F0 on the real client** (1 hero, dummy target) and re-run the same
promotion gate. Only advance when it passes, exactly as with the mock.

---

## What already works without Dota

- protocol framing + schema/order checks — `ZmqBridge` / `InProcessBridge`
- wire-observation → Observation contract parsing + validation
- Action → wire round-trip
- full `reset()/step()` loop over a fake game loop — `tests/test_adapter.py`
