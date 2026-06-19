//! Combat system: targeting, attacking, projectiles.

use crate::arena::Pos;
use crate::engine::*;
use crate::entity::*;

/// Find position of an entity by ID (linear scan — fast for typical 10-30 entities)
#[inline]
fn find_pos(troops: &[Troop], buildings: &[Building], id: EntityId) -> Option<Pos> {
    for t in troops {
        if t.id == id && t.is_alive() { return Some(t.pos); }
    }
    for b in buildings {
        if b.id == id && b.is_alive() { return Some(b.pos); }
    }
    None
}

/// Update targeting for all entities
pub fn update_targeting(engine: &mut BattleEngine) {
    // Collect snapshot of alive entity info to avoid borrow conflicts
    let troop_infos: Vec<(EntityId, u8, Pos, bool)> = engine.troops.iter()
        .filter(|t| t.is_alive() && t.state != EntityState::Deploying)
        .map(|t| (t.id, t.player_id, t.pos, t.is_air()))
        .collect();

    let building_infos: Vec<(EntityId, u8, Pos, bool)> = engine.buildings.iter()
        .filter(|b| b.is_alive())
        .map(|b| (b.id, b.player_id, b.pos, b.is_tower))
        .collect();

    // Troop targeting
    for troop in &mut engine.troops {
        if !troop.is_alive() || troop.state == EntityState::Deploying || troop.is_stunned() {
            continue;
        }

        // Check if current target is still valid
        if let Some(tid) = troop.target_id {
            let valid = troop_infos.iter().any(|(id, _, _, _)| *id == tid)
                || building_infos.iter().any(|(id, _, _, _)| *id == tid);
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
    let king_activated = [engine.players[0].king_activated, engine.players[1].king_activated];
    for building in &mut engine.buildings {
        if !building.is_alive() || building.damage == 0 {
            continue;
        }
        if building.tower_type == TowerType::King && !king_activated[building.player_id as usize] {
            continue;
        }

        let pos = building.pos;
        let range = building.range;
        let player_id = building.player_id;
        let target_mode = building.target_mode;

        // Validate current target
        let range_sq = (range as i64) * (range as i64);
        if let Some(tid) = building.target_id {
            let valid = troop_infos.iter().any(|(id, _, p, _)| {
                *id == tid && p.dist_sq(pos) <= range_sq
            });
            if !valid {
                building.target_id = None;
            }
        }

        // Find new target
        if building.target_id.is_none() {
            let can_air = matches!(target_mode, TargetMode::Both | TargetMode::Air);
            let can_ground = matches!(target_mode, TargetMode::Both | TargetMode::Ground | TargetMode::Buildings);

            let mut best_id: Option<EntityId> = None;
            let mut best_dist_sq = i64::MAX;

            for &(id, pid, tpos, is_air) in &troop_infos {
                if pid == player_id { continue; }
                if is_air && !can_air { continue; }
                if !is_air && !can_ground { continue; }
                let d = pos.dist_sq(tpos);
                if d <= range_sq && d < best_dist_sq {
                    best_dist_sq = d;
                    best_id = Some(id);
                }
            }

            building.target_id = best_id;
        }
    }
}

fn find_best_target(
    pos: Pos, player_id: u8, sight_range: i32,
    buildings_only: bool, can_air: bool, can_ground: bool,
    troop_infos: &[(EntityId, u8, Pos, bool)],
    building_infos: &[(EntityId, u8, Pos, bool)],
) -> Option<EntityId> {
    let enemy_id = 1 - player_id;
    let sr_sq = (sight_range as i64) * (sight_range as i64);
    let mut best_id: Option<EntityId> = None;
    let mut best_dist_sq = i64::MAX;

    if buildings_only {
        for &(id, pid, bpos, _) in building_infos {
            if pid != enemy_id { continue; }
            let d = pos.dist_sq(bpos);
            if d < best_dist_sq {
                best_dist_sq = d;
                best_id = Some(id);
            }
        }
        return best_id;
    }

    // Troops
    for &(id, pid, tpos, is_air) in troop_infos {
        if pid != enemy_id { continue; }
        if is_air && !can_air { continue; }
        if !is_air && !can_ground { continue; }
        let d = pos.dist_sq(tpos);
        if d <= sr_sq && d < best_dist_sq {
            best_dist_sq = d;
            best_id = Some(id);
        }
    }

    // Buildings
    for &(id, pid, bpos, _) in building_infos {
        if pid != enemy_id { continue; }
        let d = pos.dist_sq(bpos);
        if d <= sr_sq && d < best_dist_sq {
            best_dist_sq = d;
            best_id = Some(id);
        }
    }

    // If nothing in sight, target nearest tower
    if best_id.is_none() {
        for &(id, pid, bpos, is_tower) in building_infos {
            if pid != enemy_id || !is_tower { continue; }
            let d = pos.dist_sq(bpos);
            if d < best_dist_sq {
                best_dist_sq = d;
                best_id = Some(id);
            }
        }
    }

    best_id
}

/// Update combat: process attacks for all entities
pub fn update_combat(engine: &mut BattleEngine) {
    // Collect entity positions once for range checks (avoid HashMap — linear scan is faster for <50 entities)
    struct PosEntry { id: EntityId, pos: Pos }
    let mut pos_entries: Vec<PosEntry> = Vec::with_capacity(engine.troops.len() + engine.buildings.len());
    for t in &engine.troops {
        if t.is_alive() { pos_entries.push(PosEntry { id: t.id, pos: t.pos }); }
    }
    for b in &engine.buildings {
        if b.is_alive() { pos_entries.push(PosEntry { id: b.id, pos: b.pos }); }
    }

    #[inline]
    fn lookup_pos(entries: &[PosEntry], id: EntityId) -> Option<Pos> {
        entries.iter().find(|e| e.id == id).map(|e| e.pos)
    }

    // Troop attacks — collect results to avoid aliasing
    let mut attacks: Vec<(EntityId, i32, i32, u8)> = Vec::with_capacity(16); // (target_id, damage, splash_radius, player_id)

    for troop in &mut engine.troops {
        if !troop.is_alive() || troop.state == EntityState::Deploying || troop.is_stunned() {
            continue;
        }

        if let Some(target_id) = troop.target_id {
            if let Some(tp) = lookup_pos(&pos_entries, target_id) {
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
                        attacks.push((target_id, dmg, troop.area_damage_radius, troop.player_id));
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
            attacks.push((tid, building.damage, 0, building.player_id));
            building.attack_timer = building.hit_speed_ms;
        }
    }

    // Apply attacks
    for &(target_id, damage, splash_radius, attacker_player_id) in &attacks {
        if splash_radius > 0 {
            if let Some(tp) = lookup_pos(&pos_entries, target_id) {
                let radius_sq = (splash_radius as i64) * (splash_radius as i64);
                let enemy_id = 1 - attacker_player_id;
                for troop in &mut engine.troops {
                    if troop.player_id == enemy_id && troop.is_alive() && troop.pos.dist_sq(tp) <= radius_sq {
                        apply_damage_to_troop(troop, damage);
                    }
                }
                for building in &mut engine.buildings {
                    if building.player_id == enemy_id && building.is_alive() && building.pos.dist_sq(tp) <= radius_sq {
                        building.hp -= damage;
                    }
                }
            }
        } else {
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

#[inline]
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
    let mut impacts: Vec<usize> = Vec::new();
    let mut impact_data: Vec<(Pos, i32, i32, u8, Option<EntityId>)> = Vec::new();

    for (idx, proj) in engine.projectiles.iter_mut().enumerate() {
        let speed_per_tick = proj.speed * MS_PER_TICK / 1000;

        // Update homing target position
        if proj.homing {
            if let Some(tid) = proj.target_id {
                if let Some(tp) = find_pos(&engine.troops, &engine.buildings, tid) {
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
    for &(impact_pos, damage, splash, player_id, target_id) in &impact_data {
        let enemy_id = 1 - player_id;
        if splash > 0 {
            let radius_sq = (splash as i64) * (splash as i64);
            for troop in &mut engine.troops {
                if troop.player_id == enemy_id && troop.is_alive() && troop.pos.dist_sq(impact_pos) <= radius_sq {
                    apply_damage_to_troop(troop, damage);
                }
            }
            for building in &mut engine.buildings {
                if building.player_id == enemy_id && building.is_alive() && building.pos.dist_sq(impact_pos) <= radius_sq {
                    building.hp -= damage;
                }
            }
        } else if let Some(tid) = target_id {
            for troop in &mut engine.troops {
                if troop.id == tid { apply_damage_to_troop(troop, damage); break; }
            }
            for building in &mut engine.buildings {
                if building.id == tid { building.hp -= damage; break; }
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
