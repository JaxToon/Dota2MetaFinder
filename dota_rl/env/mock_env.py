"""F0-F2 deterministic mock Dota environment (spec ticket #5, milestone M1).

This is NOT the Dota 2 client. It is a compact, seedable 1v1-lane simulator that
mirrors the *core* mechanics the curriculum needs (last-hit economy, abilities,
a structure objective) so the learner, rollout, PPO, mask and evaluation code can
be built and verified independently of the game runtime (blueprint section 0:
"environment simulasi kustom" fallback).

The opponent hero can be driven by:
  * the built-in scripted laner (default, ``opponent=None``), or
  * any external policy ``opponent(enemy_obs) -> Action`` -- used for self-play /
    league training (spec section 7). The enemy sees a point-mirrored egocentric
    observation (the map is symmetric about its centre), so a policy trained on
    the agent side can play the enemy side unchanged; its action is un-mirrored
    before being applied.

The real-game adapter (`dota_adapter.py`) implements the same reset()/step()
contract so the learner does not change.
"""
from __future__ import annotations

import math
import uuid

import numpy as np

from .action import Action
from .masks import build_masks
from .observation import empty_observation, encode_agent_scalars
from .protocol import (
    NUM_ABILITIES,
    NUM_ENTITY_SLOTS,
    NUM_ITEMS,
    ActionType,
    EpisodeMeta,
)
from .reward import RewardConfig, RewardInputs, RewardShaper

# entity slot layout (fixed ordering -- spec 4.2). Slots are egocentric:
# 0 = the foe hero, 1..4 = lane creeps, 5 = the foe's tower, 6 = own tower.
SLOT_FOE_HERO = 0
SLOT_ENEMY_HERO = SLOT_FOE_HERO   # backwards-compatible alias
SLOT_CREEP_0 = 1
NUM_CREEPS = 4
SLOT_ENEMY_TOWER = 5              # the foe's tower (attackable objective)
SLOT_ALLY_TOWER = 6              # own tower (defend)
SLOT_RESERVED = 7

HERO_ID_EMPTY = 0
HERO_ID_ENEMY = 1
HERO_ID_CREEP = 2
HERO_ID_TOWER = 3

MAX_HERO_HP = 600.0
MAX_HERO_MANA = 300.0
MAX_LEVEL = 25
CREEP_HP = 55.0
CREEP_BOUNTY = 40.0
CREEP_XP = 40.0
KILL_BOUNTY = 200.0
DEATH_GOLD_LOSS = 100.0
TOWER_HP = 1400.0
ATTACK_RANGE = 0.14
VISION_RANGE = 0.45
MOVE_SPEED = 0.045
HERO_ATTACK_DMG = 22.0
TOWER_ATTACK_DMG = 40.0
RESPAWN_STEPS = 12
MAX_STEPS = 400
WIN_SCORE_MARGIN = 4

# Per-phase opponent difficulty (spec 8: F1 = "basic skill", F2 = mid-game full).
# Only used by the built-in scripted opponent; ignored when an external opponent
# policy is supplied.
PHASE_DIFFICULTY = {
    "F0": {"enemy": 0.0, "tower_zap": 0.0, "retreat_hp": 1.0, "dmg_scale": 0.0,
           "tower_hp": 700.0, "win_margin": 1.0},
    "F1": {"enemy": 0.5, "tower_zap": 0.0, "retreat_hp": 0.5, "dmg_scale": 0.6,
           "tower_hp": 500.0, "win_margin": 1.0},
    # F2 = mid-game decisions (build order, power spike) vs a moderately
    # aggressive laner -- NOT a brutal duelist (that is F3+). Calibrated so a
    # competent policy can win ~65-70%, leaving head-room for the CI-low gate.
    "F2": {"enemy": 0.6, "tower_zap": 0.4, "retreat_hp": 0.45, "dmg_scale": 0.7,
           "tower_hp": 850.0, "win_margin": 0.75},
}

# ability_slot -> (mana_cost, cooldown_steps, kind)
ABILITIES = [
    {"mana": 90.0, "cd": 8, "kind": "nuke", "dmg": 110.0, "range": 0.30},
    {"mana": 60.0, "cd": 14, "kind": "heal", "heal": 140.0, "range": 0.0},
    {"mana": 75.0, "cd": 10, "kind": "nuke", "dmg": 70.0, "range": 0.22},
]
# item_slot -> effect
ITEMS = [
    {"cd": 20, "kind": "heal", "heal": 100.0},
    {"cd": 30, "kind": "nuke", "dmg": 60.0, "range": 0.25},
]


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mirror(p) -> np.ndarray:
    """Point reflection through the map centre (0.5, 0.5)."""
    return 1.0 - np.asarray(p, dtype=np.float64)


class _Hero:
    def __init__(self, pos, tower_pos, hp0: float = MAX_HERO_HP, mana0: float = MAX_HERO_MANA):
        self.pos = np.array(pos, dtype=np.float64)
        self.tower_pos = np.array(tower_pos, dtype=np.float64)
        self.hp0, self.mana0 = hp0, mana0
        self.hp = hp0
        self.mana = mana0
        self.level = 1
        self.xp = 0.0
        self.gold = 0.0
        self.facing = 0.0
        self.ability_cd = [0] * NUM_ABILITIES
        self.item_cd = [0] * NUM_ITEMS
        self.respawn_timer = 0
        self.kills = 0
        self.deaths = 0

    @property
    def alive(self) -> bool:
        return self.respawn_timer == 0

    def maybe_level(self):
        want = min(MAX_LEVEL, 1 + int(self.xp // 220))
        self.level = max(self.level, want)


class _Creep:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=np.float64)
        self.hp = CREEP_HP
        self.alive = True
        self.respawn = 0


class MockDotaEnv:
    """Gym-like single-agent environment; the opponent is scripted or a policy."""

    def __init__(self, phase: str = "F1", reward_cfg: RewardConfig | None = None,
                 seed: int = 0, patch_id: str = "sandbox_mock_v1",
                 max_steps: int = MAX_STEPS, opponent=None, patch=None):
        self.phase = phase
        self.max_steps = max_steps
        self.reward_cfg = reward_cfg or RewardConfig()
        self.diff = PHASE_DIFFICULTY.get(phase, PHASE_DIFFICULTY["F2"])
        # opponent(enemy_obs) -> Action ; None = built-in scripted laner
        self.opponent = opponent

        # patch manifest -> the game data the env depends on (spec 12, 16)
        if patch is None:
            from .patch import default_manifest
            patch = default_manifest()
        self.patch = patch
        self.patch_id = patch.patch_id or patch_id
        self._hp_max = patch.hero.max_hp
        self._mana_max = patch.hero.max_mana
        self._hero_dmg = patch.hero.attack_damage
        self._move_speed = patch.hero.move_speed
        self._atk_range = patch.hero.attack_range
        self._abilities = [dict(a) for a in patch.abilities] or ABILITIES
        self._items = [dict(i) for i in patch.items] or ITEMS
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self.shaper = RewardShaper(self.reward_cfg)
        self.meta: EpisodeMeta | None = None
        self._reset_world(seed)

    def set_opponent(self, opponent) -> None:
        self.opponent = opponent

    # ------------------------------------------------------------------ world
    def _reset_world(self, seed):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.agent = _Hero((0.15, 0.15), (0.10, 0.10), self._hp_max, self._mana_max)
        self.enemy = _Hero((0.85, 0.85), (0.90, 0.90), self._hp_max, self._mana_max)
        self.enemy_tower_max = self.diff["tower_hp"]
        self.ally_tower_max = TOWER_HP
        self.enemy_tower_hp = self.enemy_tower_max
        self.ally_tower_hp = self.ally_tower_max
        self.creeps: list[_Creep] = []
        for i in range(NUM_CREEPS):
            f = (i + 1) / (NUM_CREEPS + 1)
            jitter = self.rng.uniform(-0.03, 0.03, size=2)
            self.creeps.append(_Creep(pos=(0.5 + jitter[0], 0.5 + jitter[1] + (f - 0.5) * 0.2)))
        self.done = False
        self.outcome = 0  # +1 win / -1 loss / 0 undecided (set at episode end)
        self._prev_phi = self._phi()

    def _phi(self) -> float:
        """Potential for shaping: progress toward the win condition (spec 6,
        potential-based so it telescopes to zero and cannot be farmed)."""
        tower_progress = 1.0 - self.enemy_tower_hp / self.enemy_tower_max
        net_kills = self.agent.kills - self.agent.deaths
        return tower_progress + 0.15 * net_kills

    def reset(self, seed: int | None = None):
        if seed is None:
            seed = int(self.rng.integers(0, 2**31 - 1))
        self._reset_world(seed)
        self.shaper = RewardShaper(self.reward_cfg)
        self.meta = EpisodeMeta(
            episode_id=uuid.uuid4().hex,
            patch_id=self.patch_id,
            seed=seed,
            curriculum_phase=self.phase,
        )
        return self._observe(), self._info()

    # ------------------------------------------------------------- observation
    def _entity_rows(self, me: _Hero, foe: _Hero, foe_tower_pos, foe_tower_hp,
                     foe_tower_max, my_tower_pos, my_tower_hp, my_tower_max,
                     show_foe: bool):
        rows: list = [None] * NUM_ENTITY_SLOTS

        if show_foe:
            vis = _dist(me.pos, foe.pos) <= VISION_RANGE
            rows[SLOT_FOE_HERO] = dict(
                present=True, is_enemy=True, is_creep=False, visible=vis and foe.alive,
                pos=foe.pos if (vis and foe.alive) else np.zeros(2),
                hp_ratio=(foe.hp / self._hp_max) if (vis and foe.alive) else 0.0,
                hero_id=HERO_ID_ENEMY, alive=foe.alive,
                in_range=foe.alive and _dist(me.pos, foe.pos) <= self._atk_range,
            )

        for i, c in enumerate(self.creeps):
            rows[SLOT_CREEP_0 + i] = dict(
                present=True, is_enemy=True, is_creep=True, visible=True,
                pos=c.pos, hp_ratio=(c.hp / CREEP_HP) if c.alive else 0.0,
                hero_id=HERO_ID_CREEP, alive=c.alive,
                in_range=c.alive and _dist(me.pos, c.pos) <= self._atk_range,
            )

        rows[SLOT_ENEMY_TOWER] = dict(
            present=True, is_enemy=True, is_creep=False, visible=True,
            pos=np.asarray(foe_tower_pos), hp_ratio=foe_tower_hp / foe_tower_max,
            hero_id=HERO_ID_TOWER, alive=foe_tower_hp > 0,
            in_range=foe_tower_hp > 0 and _dist(me.pos, foe_tower_pos) <= self._atk_range,
        )
        rows[SLOT_ALLY_TOWER] = dict(
            present=True, is_enemy=False, is_creep=False, visible=True,
            pos=np.asarray(my_tower_pos), hp_ratio=my_tower_hp / my_tower_max,
            hero_id=HERO_ID_TOWER, alive=my_tower_hp > 0, in_range=False,
        )
        return rows

    def _observe(self, mirror: bool = False):
        """Egocentric observation. ``mirror=False`` -> agent's view;
        ``mirror=True`` -> enemy's view with positions point-mirrored."""
        if not mirror:
            me, foe = self.agent, self.enemy
            foe_tower_pos, my_tower_pos = self.enemy.tower_pos, self.agent.tower_pos
            foe_tower_hp, foe_tower_max = self.enemy_tower_hp, self.enemy_tower_max
            my_tower_hp, my_tower_max = self.ally_tower_hp, self.ally_tower_max
        else:
            me, foe = self.enemy, self.agent
            foe_tower_pos, my_tower_pos = self.agent.tower_pos, self.enemy.tower_pos
            foe_tower_hp, foe_tower_max = self.ally_tower_hp, self.ally_tower_max
            my_tower_hp, my_tower_max = self.enemy_tower_hp, self.enemy_tower_max

        show_foe = self.phase != "F0" or mirror
        obs = empty_observation()

        me_pos = _mirror(me.pos) if mirror else np.asarray(me.pos, dtype=np.float64)
        obs["agent_scalars"] = encode_agent_scalars(
            hp_ratio=np.clip(me.hp / self._hp_max, 0, 1),
            mana_ratio=np.clip(me.mana / self._mana_max, 0, 1),
            level_norm=me.level / MAX_LEVEL,
            gold_norm=np.clip(me.gold / 3000.0, 0, 1),
            xp_norm=np.clip(me.xp / (220.0 * MAX_LEVEL), 0, 1),
            pos_x=np.clip(me_pos[0], 0, 1),
            pos_y=np.clip(me_pos[1], 0, 1),
            facing=(me.facing % (2 * math.pi)) / (2 * math.pi),
            sim_time_norm=self.t / self.max_steps,
        )

        ab_ready = np.zeros(NUM_ABILITIES, dtype=bool)
        for i, ab in enumerate(self._abilities[:NUM_ABILITIES]):
            ready = me.ability_cd[i] == 0 and me.mana >= ab["mana"]
            ab_ready[i] = ready
            obs["abilities"][i] = [
                float(ready),
                1.0 - min(1.0, me.ability_cd[i] / max(1, ab["cd"])),
                float(me.mana >= ab["mana"]),
                (i + 1) / (NUM_ABILITIES + 1),
            ]
            obs["ability_effect_ids"][i] = i + 1

        it_ready = np.zeros(NUM_ITEMS, dtype=bool)
        for i in range(NUM_ITEMS):
            has = i < len(self._items)
            ready = has and me.item_cd[i] == 0
            it_ready[i] = ready
            obs["items"][i] = [float(has), float(ready), (i + 1) / (NUM_ITEMS + 1)]

        rows = self._entity_rows(
            me, foe, foe_tower_pos, foe_tower_hp, foe_tower_max,
            my_tower_pos, my_tower_hp, my_tower_max, show_foe,
        )
        ent_present = np.zeros(NUM_ENTITY_SLOTS, dtype=bool)
        ent_enemy = np.zeros(NUM_ENTITY_SLOTS, dtype=bool)
        ent_alive = np.zeros(NUM_ENTITY_SLOTS, dtype=bool)
        ent_range = np.zeros(NUM_ENTITY_SLOTS, dtype=bool)
        for s, r in enumerate(rows):
            if r is None:
                continue
            ent_present[s] = True
            ent_enemy[s] = r["is_enemy"]
            ent_alive[s] = r["alive"]
            ent_range[s] = r["in_range"]
            r_pos = _mirror(r["pos"]) if mirror else np.asarray(r["pos"])
            rel = r_pos - me_pos
            obs["entities"][s] = [
                1.0, float(r["is_enemy"]), float(r["is_creep"]), float(r["visible"]),
                float(np.clip(rel[0] * 0.5 + 0.5, 0, 1)),
                float(np.clip(rel[1] * 0.5 + 0.5, 0, 1)),
                float(np.clip(r["hp_ratio"], 0, 1)),
                float(np.clip(_dist(me.pos, r["pos"]) / 1.5, 0, 1)),
            ]
            obs["entity_hero_ids"][s] = r["hero_id"]

        obs["masks"] = build_masks(
            entity_present=ent_present, entity_is_enemy=ent_enemy,
            entity_alive=ent_alive, entity_in_range=ent_range,
            ability_ready=ab_ready, item_ready=it_ready,
        )
        return obs

    def _info(self):
        return {
            "episode_id": self.meta.episode_id if self.meta else None,
            "patch_id": self.patch_id,
            "schema_version": self.meta.schema_version if self.meta else None,
            "phase": self.phase,
            "t": self.t,
            "agent_kills": self.agent.kills,
            "agent_deaths": self.agent.deaths,
            "enemy_tower_hp": self.enemy_tower_hp,
            "ally_tower_hp": self.ally_tower_hp,
            "net_worth": self.agent.gold + self.agent.xp,
            "enemy_net_worth": self.enemy.gold + self.enemy.xp,
            "outcome": self.outcome,
        }

    def enemy_observation(self):
        """Mirrored egocentric observation for the opponent policy."""
        return self._observe(mirror=True)

    # -------------------------------------------------------------------- step
    def step(self, action: Action, opponent_action: Action | None = None):
        """``opponent_action`` (in the mirrored enemy frame) overrides both the
        built-in scripted AI and ``self.opponent`` -- used by the rollout loop so
        opponent forward passes can be batched across envs."""
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        self.t += 1
        a = self.agent
        rin = RewardInputs()
        pre_nw = a.gold + a.xp

        self._hero_upkeep(a)

        if a.alive:
            self._apply_action(a, self.enemy, action, rin, is_learner=True)

        # --- opponent
        if self.phase != "F0":
            if opponent_action is not None:
                self._hero_upkeep(self.enemy)
                if self.enemy.alive:
                    self._apply_enemy_action(opponent_action, rin)
            elif self.opponent is None:
                self._enemy_ai(rin)
            else:
                self._hero_upkeep(self.enemy)
                if self.enemy.alive:
                    self._apply_enemy_action(self.opponent(self._observe(mirror=True)), rin)

        self._creep_tick()
        if self.phase != "F0":
            self._tower_ai(rin)

        # --- economy / xp deltas
        post_nw = a.gold + a.xp
        rin.d_net_worth = post_nw - pre_nw
        rin.d_xp = 0.0
        rin.d_objective -= 0.005  # tiny time cost (spec 6)
        a.maybe_level()

        # --- potential-based shaping
        phi_new = self._phi()
        rin.potential_delta = 0.99 * phi_new - self._prev_phi
        self._prev_phi = phi_new

        # --- terminal check
        terminal = self._check_terminal()
        rin.terminal = terminal
        if terminal != 0:
            self.done = True
            self.outcome = terminal
        elif self.t >= self.max_steps:
            self.done = True
            nw_diff = (a.gold + a.xp) - (self.enemy.gold + self.enemy.xp)
            margin = ((a.kills - a.deaths)
                      + 2.0 * (self.enemy_tower_max - self.enemy_tower_hp) / self.enemy_tower_max
                      - 2.0 * (self.ally_tower_max - self.ally_tower_hp) / self.ally_tower_max
                      + float(np.clip(nw_diff / 2500.0, -1.5, 1.5)))
            wm = self.diff["win_margin"]
            self.outcome = 1 if margin >= wm else (-1 if margin <= -wm else 0)
            rin.terminal = 0

        reward, comps = self.shaper.step(rin)
        obs = self._observe()
        info = self._info()
        info["reward_components"] = comps
        truncated = self.done and terminal == 0
        terminated = self.done and terminal != 0
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------- mechanics
    def _hero_upkeep(self, h: _Hero):
        for i in range(NUM_ABILITIES):
            h.ability_cd[i] = max(0, h.ability_cd[i] - 1)
        for i in range(NUM_ITEMS):
            h.item_cd[i] = max(0, h.item_cd[i] - 1)
        if h.alive:
            h.hp = min(self._hp_max, h.hp + 3.0)
            h.mana = min(self._mana_max, h.mana + 4.0)
        else:
            h.respawn_timer -= 1
            if h.respawn_timer == 0:
                h.hp = self._hp_max
                h.mana = self._mana_max
                h.pos = h.tower_pos.copy()

    def _move_toward(self, hero: _Hero, target):
        target = np.asarray(target, dtype=np.float64)
        d = target - hero.pos
        n = np.linalg.norm(d)
        if n > 1e-6:
            hero.facing = math.atan2(d[1], d[0])
            step = min(self._move_speed, n)
            hero.pos = np.clip(hero.pos + d / n * step, 0.0, 1.0)

    # --- learner (agent) action -------------------------------------------------
    def _apply_action(self, actor: _Hero, foe: _Hero, action: Action,
                      rin: RewardInputs, is_learner: bool):
        t = ActionType(action.action_type)
        if t == ActionType.MOVE and action.target_xy is not None:
            self._move_toward(actor, action.target_xy)
        elif t == ActionType.ATTACK and action.target_entity_slot >= 0:
            self._slot_attack(actor, foe, action.target_entity_slot, self._hero_dmg, rin, is_learner)
        elif t == ActionType.CAST and action.ability_slot >= 0:
            self._cast(actor, foe, action.ability_slot, action.target_entity_slot, rin, is_learner)
        elif t == ActionType.ITEM and action.item_slot >= 0:
            self._use_item(actor, foe, action.item_slot, rin, is_learner)
        # STOP: nothing

    def _slot_attack(self, actor: _Hero, foe: _Hero, slot: int, dmg: float,
                     rin: RewardInputs, is_learner: bool):
        if slot == SLOT_FOE_HERO:
            if foe.alive and _dist(actor.pos, foe.pos) <= self._atk_range:
                self._hero_takes_damage(foe, actor, dmg, rin, foe_is_agent=not is_learner)
        elif SLOT_CREEP_0 <= slot < SLOT_CREEP_0 + NUM_CREEPS:
            c = self.creeps[slot - SLOT_CREEP_0]
            if c.alive and _dist(actor.pos, c.pos) <= self._atk_range:
                c.hp -= dmg
                if c.hp <= 0:
                    c.alive = False
                    c.respawn = 20
                    actor.gold += CREEP_BOUNTY
                    actor.xp += CREEP_XP
                    actor.maybe_level()
        elif slot == SLOT_ENEMY_TOWER:
            self._tower_attack(actor, foe_tower=True, dmg=dmg, actor_is_agent=is_learner, rin=rin)
        elif slot == SLOT_ALLY_TOWER:
            self._tower_attack(actor, foe_tower=False, dmg=dmg, actor_is_agent=is_learner, rin=rin)

    def _cast(self, actor: _Hero, foe: _Hero, ab_slot: int, ent_slot: int,
              rin: RewardInputs, is_learner: bool):
        if ab_slot < 0 or ab_slot >= NUM_ABILITIES:
            return
        ab = self._abilities[ab_slot]
        if actor.ability_cd[ab_slot] > 0 or actor.mana < ab["mana"]:
            return
        actor.mana -= ab["mana"]
        actor.ability_cd[ab_slot] = ab["cd"]
        if ab["kind"] == "heal":
            actor.hp = min(self._hp_max, actor.hp + ab["heal"])
            return
        # nuke
        if ent_slot == SLOT_FOE_HERO:
            if foe.alive and _dist(actor.pos, foe.pos) <= ab["range"]:
                self._hero_takes_damage(foe, actor, ab["dmg"], rin, foe_is_agent=not is_learner)
        elif SLOT_CREEP_0 <= ent_slot < SLOT_CREEP_0 + NUM_CREEPS:
            c = self.creeps[ent_slot - SLOT_CREEP_0]
            if c.alive and _dist(actor.pos, c.pos) <= ab["range"]:
                c.hp -= ab["dmg"]
                if c.hp <= 0:
                    c.alive = False
                    c.respawn = 20
                    actor.gold += CREEP_BOUNTY
                    actor.xp += CREEP_XP

    def _use_item(self, actor: _Hero, foe: _Hero, slot: int, rin: RewardInputs, is_learner: bool):
        if slot < 0 or slot >= len(self._items) or actor.item_cd[slot] > 0:
            return
        it = self._items[slot]
        actor.item_cd[slot] = it["cd"]
        if it["kind"] == "heal":
            actor.hp = min(self._hp_max, actor.hp + it["heal"])
        elif it["kind"] == "nuke":
            if foe.alive and _dist(actor.pos, foe.pos) <= it["range"]:
                self._hero_takes_damage(foe, actor, it["dmg"], rin, foe_is_agent=not is_learner)

    def _hero_takes_damage(self, victim: _Hero, attacker: _Hero, dmg: float,
                           rin: RewardInputs, foe_is_agent: bool):
        """``foe_is_agent`` == the victim is the learner's agent hero."""
        if not victim.alive:
            return
        victim.hp -= dmg
        if victim.hp <= 0:
            victim.respawn_timer = RESPAWN_STEPS
            victim.deaths += 1
            attacker.kills += 1
            victim.gold = max(0.0, victim.gold - DEATH_GOLD_LOSS)
            if foe_is_agent:
                rin.combat_event -= 1.0   # agent died
            else:
                attacker.gold += KILL_BOUNTY
                rin.combat_event += 1.0   # agent killed the enemy

    def _tower_attack(self, actor: _Hero, foe_tower: bool, dmg: float,
                      actor_is_agent: bool, rin: RewardInputs):
        # "foe_tower" is relative to the actor -> resolve to a concrete tower.
        hits_enemy_tower = foe_tower if actor_is_agent else (not foe_tower)
        tower_pos = self.enemy.tower_pos if hits_enemy_tower else self.agent.tower_pos
        if _dist(actor.pos, tower_pos) > self._atk_range:
            return
        if hits_enemy_tower:
            if self.enemy_tower_hp <= 0:
                return
            before = self.enemy_tower_hp
            self.enemy_tower_hp = max(0.0, self.enemy_tower_hp - dmg)
            if actor_is_agent:
                rin.d_structure += (before - self.enemy_tower_hp) / self.enemy_tower_max
        else:
            if self.ally_tower_hp <= 0:
                return
            before = self.ally_tower_hp
            self.ally_tower_hp = max(0.0, self.ally_tower_hp - dmg)
            if not actor_is_agent:
                rin.d_structure -= (before - self.ally_tower_hp) / self.ally_tower_max

    # --- external opponent ----------------------------------------------------
    def _apply_enemy_action(self, action: Action, rin: RewardInputs):
        # The enemy's action is expressed in its mirrored egocentric frame.
        t = ActionType(action.action_type)
        if t == ActionType.MOVE and action.target_xy is not None:
            self._move_toward(self.enemy, _mirror(action.target_xy))
        else:
            self._apply_action(self.enemy, self.agent, action, rin, is_learner=False)

    # --- built-in scripted opponent -----------------------------------------
    def _enemy_ai(self, rin: RewardInputs):
        e = self.enemy
        self._hero_upkeep(e)
        if not e.alive:
            return
        a = self.agent
        dmg_scale = self.diff["dmg_scale"]
        aggr = self.diff["enemy"]

        if e.hp < self.diff["retreat_hp"] * self._hp_max:
            self._move_toward(e, e.tower_pos)
            return
        if a.alive and _dist(e.pos, a.pos) <= self._atk_range and self.rng.random() < 0.4 + 0.6 * aggr:
            self._hero_takes_damage(a, e, self._hero_dmg * dmg_scale, rin, foe_is_agent=True)
            return
        if (a.alive and _dist(e.pos, a.pos) <= self._abilities[0]["range"] and e.mana >= self._abilities[0]["mana"]
                and e.ability_cd[0] == 0 and self.rng.random() < aggr):
            e.mana -= self._abilities[0]["mana"]
            e.ability_cd[0] = self._abilities[0]["cd"]
            self._hero_takes_damage(a, e, self._abilities[0]["dmg"] * dmg_scale, rin, foe_is_agent=True)
            return
        best, bestd = None, 1e9
        for i, c in enumerate(self.creeps):
            if c.alive:
                d = _dist(e.pos, c.pos)
                if d < bestd:
                    best, bestd = i, d
        if best is not None:
            if bestd <= self._atk_range:
                c = self.creeps[best]
                c.hp -= self._hero_dmg
                if c.hp <= 0:
                    c.alive = False
                    c.respawn = 20
                    e.gold += CREEP_BOUNTY
                    e.xp += CREEP_XP
                    e.maybe_level()
            else:
                self._move_toward(e, self.creeps[best].pos)
        else:
            self._move_toward(e, np.array([0.5, 0.5]))

    def _tower_ai(self, rin: RewardInputs):
        a, e = self.agent, self.enemy
        zap = self.diff["tower_zap"]
        if zap <= 0:
            return
        if a.alive and _dist(a.pos, self.enemy.tower_pos) <= self._atk_range and self.enemy_tower_hp > 0:
            self._hero_takes_damage(a, e, TOWER_ATTACK_DMG * zap, rin, foe_is_agent=True)
        if e.alive and _dist(e.pos, self.agent.tower_pos) <= self._atk_range and self.ally_tower_hp > 0:
            e.hp -= TOWER_ATTACK_DMG * zap
            if e.hp <= 0:
                e.respawn_timer = RESPAWN_STEPS
                e.deaths += 1
                a.kills += 1
                rin.combat_event += 1.0

    def _creep_tick(self):
        for c in self.creeps:
            if not c.alive:
                c.respawn -= 1
                if c.respawn <= 0:
                    c.alive = True
                    c.hp = CREEP_HP
            else:
                c.hp = min(CREEP_HP, c.hp + 1.0)

    def _check_terminal(self) -> int:
        a = self.agent
        if self.enemy_tower_hp <= 0:
            return 1
        if self.ally_tower_hp <= 0:
            return -1
        if a.kills - a.deaths >= WIN_SCORE_MARGIN:
            return 1
        if a.deaths - a.kills >= WIN_SCORE_MARGIN:
            return -1
        return 0
