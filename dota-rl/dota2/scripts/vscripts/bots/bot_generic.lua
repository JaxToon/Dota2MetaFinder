--[[
  bot_generic.lua  --  spec section 11 / blueprint step 2 (Observation Harvester).

  Skeleton only. When the API probe (scripts/api_probe.lua) confirms the needed
  fields exist, this module serialises the observation each Think() tick and
  ships it over a socket to the Python env server, then applies the action token
  it receives back (see rl_action_handler.lua).

  Until then the repo trains on dota_rl/env/mock_env.py and this file is inert.
]]

local OBS_SCHEMA_VERSION = 1
local TICK = 0

local function build_observation(bot)
  return {
    schema_version = OBS_SCHEMA_VERSION,
    step_id = TICK,
    sim_time = GameTime(),
    agent = {
      hp_ratio = bot:GetHealth() / bot:GetMaxHealth(),
      mana_ratio = bot:GetMana() / math.max(1, bot:GetMaxMana()),
      pos = { bot:GetLocation().x, bot:GetLocation().y },
      facing = bot:GetFacing(),
      level = bot:GetLevel(),
      gold = bot:GetGold(),
    },
    -- entities / abilities / masks: fill in after api_probe validation
  }
end

function Think()
  local bot = GetBot()
  TICK = TICK + 1
  local obs = build_observation(bot)
  -- TODO: send `obs` to Python via transport/zmq_bridge.py and await an action
  -- require("bots.rl_action_handler").apply(bot, action)
end

return { Think = Think, build_observation = build_observation }
