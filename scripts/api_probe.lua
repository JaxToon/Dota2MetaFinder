--[[
  api_probe.lua  --  spec ticket #3 / milestone M0 (blueprint section 0).

  FIRST real deliverable. Drop this in
    <SteamLibrary>/steamapps/common/dota 2 beta/game/dota/scripts/vscripts/bots/
  launch a LOCAL custom lobby (bot match) with launch options:
    -dev -vconsole +dota_force_gamemode 1

  It does NOT train anything. It enumerates which observation fields the PUBLIC
  Bot Scripting API actually exposes, and writes an "access matrix" so we can
  decide: continue with the real client, or pivot to the custom simulator
  (dota_rl/env/mock_env.py) which is what this repo ships today.
]]

local PROBE = {}
local probed = false

local function try(label, fn)
  local ok, val = pcall(fn)
  local available = ok and val ~= nil
  PROBE[label] = { available = available, sample = available and tostring(val) or nil }
  print(string.format("[api_probe] %-32s %s", label, available and "OK" or "MISSING"))
end

local function probe_once(bot)
  if probed then return end
  probed = true
  print("[api_probe] ===== Dota 2 Bot API access matrix =====")

  try("self.GetHealth",            function() return bot:GetHealth() end)
  try("self.GetMana",              function() return bot:GetMana() end)
  try("self.GetLocation",          function() return bot:GetLocation() end)
  try("self.GetFacing",            function() return bot:GetFacing() end)
  try("self.GetLevel",             function() return bot:GetLevel() end)
  try("self.GetGold",              function() return bot:GetGold() end)

  try("ability.GetCooldownTimeRemaining", function()
    return bot:GetAbilityInSlot(0):GetCooldownTimeRemaining() end)
  try("ability.GetManaCost",       function() return bot:GetAbilityInSlot(0):GetManaCost() end)

  -- the critical unknowns: do we see hidden / fogged enemy state?
  try("enemy.GetHealth (fogged)",  function()
    local e = bot:GetNearbyHeroes(2000, true, BOT_MODE_NONE)[1]; return e and e:GetHealth() end)
  try("enemy.ability cooldown",    function()
    local e = bot:GetNearbyHeroes(2000, true, BOT_MODE_NONE)[1]
    return e and e:GetAbilityInSlot(0):GetCooldownTimeRemaining() end)
  try("enemy.GetLocation when not visible", function()
    local e = bot:GetNearbyHeroes(2000, true, BOT_MODE_NONE)[1]
    return e and (not e:CanBeSeen()) and e:GetLocation() end)

  try("GetUnitList(UNIT_LIST_ENEMY_CREEPS)", function()
    return #GetUnitList(UNIT_LIST_ENEMY_CREEPS) end)
  try("RealTime tick",             function() return GameTime() end)

  -- dump for the access matrix
  local f = io.open("api_probe_result.json", "w")
  if f then
    f:write("{\n")
    local first = true
    for k, v in pairs(PROBE) do
      if not first then f:write(",\n") end
      first = false
      f:write(string.format('  "%s": {"available": %s}', k, tostring(v.available)))
    end
    f:write("\n}\n")
    f:close()
    print("[api_probe] wrote api_probe_result.json")
  end
  print("[api_probe] ================================================")
  print("[api_probe] If many enemy/fogged fields are MISSING, use the custom")
  print("[api_probe] simulator (dota_rl/env/mock_env.py) for training.")
end

function Think()
  local bot = GetBot()
  probe_once(bot)
  bot:Action_ClearActions(true)   -- stay put; this is a probe, not a player
end

return { Think = Think }
