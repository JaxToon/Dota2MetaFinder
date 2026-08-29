--[[
  rl_action_handler.lua  --  spec section 11 / blueprint step 4.

  Receives a structured action token (matching dota_rl/env/action.py) from the
  Python learner and executes it precisely in-engine. Skeleton only; wired up
  after the API probe (M0) passes.
]]

local ActionType = { STOP = 0, MOVE = 1, ATTACK = 2, CAST = 3, ITEM = 4 }

local function resolve_entity(bot, slot)
  -- slot -> handle mapping must match the Python-side fixed slot layout
  local enemies = bot:GetNearbyHeroes(2000, true, BOT_MODE_NONE)
  return enemies[slot + 1]
end

local function apply(bot, action)
  local t = action.action_type
  if t == ActionType.STOP then
    bot:Action_ClearActions(true)
  elseif t == ActionType.MOVE then
    bot:Action_MoveToLocation(Vector(action.target_xy[1], action.target_xy[2], 0))
  elseif t == ActionType.ATTACK then
    local e = resolve_entity(bot, action.target_entity_slot)
    if e then bot:Action_AttackUnit(e, true) end
  elseif t == ActionType.CAST then
    local ab = bot:GetAbilityInSlot(action.ability_slot)
    local e = resolve_entity(bot, action.target_entity_slot)
    if ab and e then bot:Action_UseAbilityOnEntity(ab, e) end
  elseif t == ActionType.ITEM then
    local item = bot:GetItemInSlot(action.item_slot)
    if item then bot:Action_UseAbility(item) end
  end
end

return { apply = apply, ActionType = ActionType }
