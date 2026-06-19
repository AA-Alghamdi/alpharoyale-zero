//! Combat system: targeting, attacking, projectiles.

use crate::arena::Pos;
use crate::engine::*;
use crate::entity::*;

/// Update targeting for all entities
pub fn update_targeting(engine: &mut BattleEngine) {
    // Collect positions of all alive entities for target lookup
    let troop_infos: Vec<(EntityId, u8, Pos, bool, bool)> = engine.troops.iter()
        .filter(|t| t.is_alive() && t.state != EntityState::Deploying)
        .map(|t| (t.id, t.player_id, t.pos, t.is_air(), false))
        .collect();

    let building_infos: Vec<(EntityId, u8, Pos, bool, bool)> = engine.buildings.iter()
        .filter(|b| b.is_alive())
        .map(|b| (b.id, b.player_id, b.pos, false, b.is_tower))
        .collect();

    // Troop targeting
    for troop in &mut engine.troops {
        if !troop.is_alive() || troop.state == EntityState::Deploying || troop.is_stunned() {
            continue;
        }

        // Check if current target is still valid
        if let Some(tid) = troop.target_id {
            let valid = troop_infos.iter().any(|(id, _, _, _, _)| *id == tid)
                || building_infos.iter().any(|(id, _, _, _, _)| *id == tid);
            if !valid {
                troop.target_id = None;
                troop.state = EntityState::Idle;
            }
        }

        // Find new target if needed
        if troop.target_id.is_none() {
            let target = find_best_target(
                troop.pos, troop.player_id, troop.sight_range,
                troop.targets_only_buildings(), troop.can_attack_air(), troop.can_attack_ground(),
                &troop_infos, &building_infos,
            );
            troop.target_id = target;
            if target.is_some() {
                troop.state = EntityState::Moving;
            }
        }
    }

    // Building targeting
    for i in 0..engine.buildings.len() {
        let b = &engine.buildings[i];
        if !b.is_alive() || b.damage == 0 {
            continue;
        }
        if b.tower_type == TowerType::King && !engine.players[b.player_id as usize].king_activated {
            continue;
        }

        let player_id = b.player_id;
        let pos = b.pos;
        let range = b.range;
        let target_mode = b.target_mode;

        // Validate current target
        if let Some(tid) = engine.buildings[i].target_id {
            let valid = troop_infos.iter().any(|(id, _, p, _, _)| {
                *id == tid && p.dist(pos) <= range as f32
            });
            if !valid {
                engine.buildings[i].target_id = None;
            }
        }

        // Find new target
        if engine.buildings[i].target_id.is_none() {
            let can_air = matches!(target_mode, TargetMode::Both | TargetMode::Air);
            let can_ground = matches!(target_mode, TargetMode::Both | TargetMode::Ground | TargetMode::Buildings);

            let mut best_id: Option<EntityId> = None;
            let mut best_dist = f32::MAX;

            for (id, pid, tpos, is_air, _) in &troop_infos {
                if *pid == player_id { continue; }
                if *is_air && !can_air { continue; }
                if !*is_air && !can_ground { continue; }
                let d = pos.dist(*tpos);
                if d <= range as f32 && d < best_dist {
                    best_dist = d;
                    best_id = Some(*id);
                }
            }

            engine.buildings[i].target_id = best_id;
        }
    }
}

fn find_best_target(
    pos: Pos, player_id: u8, sight_range: i32,
    buildings_only: bool, can_air: bool, can_ground: bool,
    troop_infos: &[(EntityId, u8, Pos, bool, bool)],
    building_infos: &[(EntityId, u8, Pos, bool, bool)],
) -> Option<EntityId> {
    let enemy_id = 1 - player_id;
    let sr = sight_range as f32;
    let mut best_id: Option<EntityId> = None;
    let mut best_dist = f32::MAX;

    if buildings_only {
        for (id, pid, bpos, _, _) in building_infos {
            if *pid != enemy_id { continue; }
            let d = pos.dist(*bpos);
            if d < best_dist {
                best_dist = d;
                best_id = Some(*id);
            }
        }
        return best_id;
    }

    // Troops
    for (id, pid, tpos, is_air, _) in troop_infos {
        if *pid != enemy_id { continue; }
        if *is_air && !can_air { continue; }
        if !*is_air && !can_ground { continue; }
        let d = pos.dist(*tpos);
        if d <= sr && d < best_dist {
            best_dist = d;
            best_id = Some(*id);
        }
    }

    // Buildings
    for (id, pid, bpos, _, _) in building_infos {
        if *pid != enemy_id { continue; }
        let d = pos.dist(*bpos);
        if d <= sr && d < best_dist {
            best_dist = d;
            best_id = Some(*id);
        }
    }

    // If nothing in sight, target nearest tower
    if best_id.is_none() {
        for (id, pid, bpos, _, is_tower) in building_infos {
            if *pid != enemy_id || !*is_tower { continue; }
            let d = pos.dist(*bpos);
            if d < best_dist {
                best_dist = d;
                best_id = Some(*id);
            }
        }
    }

    best_id
}

/// Update combat: process attacks for all entities
pub fn update_combat(engine: &mut BattleEngine) {
    // Collect entity positions for range checks
    let entity_positions: Vec<(EntityId, Pos)> = engine.troops.iter()
        .filter(|t| t.is_alive())
        .map(|t| (t.id, t.pos))
        .chain(engine.buildings.iter().filter(|b| b.is_alive()).map(|b| (b.id, b.pos)))
        .collect();

    let pos_map: std::collections::HashMap<EntityId, Pos> = entity_positions.into_iter().collect();

    // Troop attacks — collect results
    let mut attacks: Vec<(EntityId, i32, Pos, i32, u8)> = Vec::new();

    for troop in &mut engine.troops {
        if !troop.is_alive() || troop.state == EntityState::Deploying || troop.is_stunned() {
            continue;
        }

        if let Some(target_id) = troop.target_id {
            if let Some(&tp) = pos_map.get(&target_id) {
                let dist = troop.pos.dist(tp);
                let range = troop.range as f32 + troop.collision_radius as f32;

                if dist <= range {
                    troop.state = EntityState::Attacking;
                    troop.attack_timer -= MS_PER_TICK;

                    if troop.attack_timer <= 0 {
                        let mut dmg = troop.damage;
                        if troop.is_charging && troop.charge_range > 0 {
                            dmg = dmg * troop.charge_damage_mult / 100;
                            troop.is_charging = false;
                        }
                        attacks.push((target_id, dmg, troop.pos, troop.area_damage_radius, troop.player_id));
                        troop.attack_timer = troop.effective_hit_speed();
                    }
                }
            }
        }
    }

    // Building attacks
    for building in &mut engine.buildings {
        if !building.is_alive() || building.damage == 0 || building.target_id.is_none() {
            continue;
        }
        building.attack_timer -= MS_PER_TICK;
        if building.attack_timer <= 0 {
            let tid = building.target_id.unwrap();
            attacks.push((tid, building.damage, building.pos, 0, building.player_id));
            building.attack_timer = building.hit_speed_ms;
        }
    }

    // Apply attacks
    for (target_id, damage, _attacker_pos, splash_radius, attacker_player_id) in attacks {
        if splash_radius > 0 {
            // Area damage
            if let Some(&tp) = pos_map.get(&target_id) {
                let radius_sq = (splash_radius as i64) * (splash_radius as i64);
                let enemy_id = 1 - attacker_player_id;
                for troop in &mut engine.troops {
                    if troop.player_id == enemy_id && troop.is_alive() {
                        if troop.pos.dist_sq(tp) <= radius_sq {
                            apply_damage_to_troop(troop, damage);
                        }
                    }
                }
                for building in &mut engine.buildings {
                    if building.player_id == enemy_id && building.is_alive() {
                        if building.pos.dist_sq(tp) <= radius_sq {
                            building.hp -= damage;
                        }
                    }
                }
            }
        } else {
            // Single target
            let mut hit = false;
            for troop in &mut engine.troops {
                if troop.id == target_id {
                    apply_damage_to_troop(troop, damage);
                    hit = true;
                    break;
                }
            }
            if !hit {
                for building in &mut engine.buildings {
                    if building.id == target_id {
                        building.hp -= damage;
                        break;
                    }
                }
            }
        }
    }
}

fn apply_damage_to_troop(troop: &mut Troop, damage: i32) {
    if troop.shield_hp > 0 {
        troop.shield_hp -= damage;
        if troop.shield_hp < 0 {
            troop.hp += troop.shield_hp;
            troop.shield_hp = 0;
        }
        return;
    }
    troop.hp -= damage;
}

/// Update projectiles in flight
pub fn update_projectiles(engine: &mut BattleEngine) {
    // Collect entity positions for homing
    let pos_map: std::collections::HashMap<EntityId, Pos> = engine.troops.iter()
        .filter(|t| t.is_alive())
        .map(|t| (t.id, t.pos))
        .chain(engine.buildings.iter().filter(|b| b.is_alive()).map(|b| (b.id, b.pos)))
        .collect();

    let mut impacts: Vec<usize> = Vec::new();
    let mut impact_data: Vec<(Pos, i32, i32, u8, Option<EntityId>)> = Vec::new();

    for (idx, proj) in engine.projectiles.iter_mut().enumerate() {
        let speed_per_tick = proj.speed * MS_PER_TICK / 1000;

        // Update homing target
        if proj.homing {
            if let Some(tid) = proj.target_id {
                if let Some(&tp) = pos_map.get(&tid) {
                    proj.target_pos = tp;
                }
            }
        }

        proj.pos = proj.pos.move_toward(proj.target_pos, speed_per_tick);

        if proj.pos.dist(proj.target_pos) < 200.0 {
            impacts.push(idx);
            impact_data.push((proj.pos, proj.damage, proj.splash_radius, proj.player_id, proj.target_id));
        }
    }

    // Apply impacts
    for (impact_pos, damage, splash, player_id, target_id) in &impact_data {
        let enemy_id = 1 - *player_id;
        if *splash > 0 {
            let radius_sq = (*splash as i64) * (*splash as i64);
            for troop in &mut engine.troops {
                if troop.player_id == enemy_id && troop.is_alive() && troop.pos.dist_sq(*impact_pos) <= radius_sq {
                    apply_damage_to_troop(troop, *damage);
                }
            }
            for building in &mut engine.buildings {
                if building.player_id == enemy_id && building.is_alive() && building.pos.dist_sq(*impact_pos) <= radius_sq {
                    building.hp -= *damage;
                }
            }
        } else if let Some(tid) = target_id {
            for troop in &mut engine.troops {
                if troop.id == *tid { apply_damage_to_troop(troop, *damage); break; }
            }
            for building in &mut engine.buildings {
                if building.id == *tid { building.hp -= *damage; break; }
            }
        }
    }

    // Remove impacted projectiles
    let mut idx = 0;
    engine.projectiles.retain(|_| {
        let keep = !impacts.contains(&idx);
        idx += 1;
        keep
    });
}
