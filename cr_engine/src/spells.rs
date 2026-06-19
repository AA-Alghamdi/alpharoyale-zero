//! Spell system: instant spells, persistent effects, special mechanics.

use crate::arena::Pos;
use crate::engine::*;
use crate::entity::*;

/// Cast a spell (from spells_other.csv)
pub fn cast_spell(engine: &mut BattleEngine, player_id: u8, spell_name: &str, pos: Pos) {
    match spell_name {
        "Fireball" => {
            engine.pending_spells.push(PendingSpell {
                player_id,
                name: "Fireball".to_string(),
                pos,
                radius: 2500,
                damage: 325,
                delay_ms: 500,
                duration_ms: 0,
                tick_damage: 0,
                effect: SpellEffect::Damage,
            });
        }
        "Arrows" => {
            engine.pending_spells.push(PendingSpell {
                player_id,
                name: "Arrows".to_string(),
                pos,
                radius: 4000,
                damage: 115,
                delay_ms: 1500, // arrows have travel time
                duration_ms: 0,
                tick_damage: 0,
                effect: SpellEffect::Damage,
            });
        }
        "Zap" => {
            engine.pending_spells.push(PendingSpell {
                player_id,
                name: "Zap".to_string(),
                pos,
                radius: 2500,
                damage: 75,
                delay_ms: 200,
                duration_ms: 0,
                tick_damage: 0,
                effect: SpellEffect::DamageAndStun,
            });
        }
        "Lightning" => {
            engine.pending_spells.push(PendingSpell {
                player_id,
                name: "Lightning".to_string(),
                pos,
                radius: 3500,
                damage: 652,
                delay_ms: 500,
                duration_ms: 0,
                tick_damage: 0,
                effect: SpellEffect::DamageAndStun,
            });
        }
        "Rocket" => {
            engine.pending_spells.push(PendingSpell {
                player_id,
                name: "Rocket".to_string(),
                pos,
                radius: 2000,
                damage: 700,
                delay_ms: 2000,
                duration_ms: 0,
                tick_damage: 0,
                effect: SpellEffect::Damage,
            });
        }
        "Freeze" => {
            engine.pending_spells.push(PendingSpell {
                player_id,
                name: "Freeze".to_string(),
                pos,
                radius: 3000,
                damage: 76,
                delay_ms: 0,
                duration_ms: 4000,
                tick_damage: 0,
                effect: SpellEffect::Freeze,
            });
        }
        "Poison" => {
            engine.pending_spells.push(PendingSpell {
                player_id,
                name: "Poison".to_string(),
                pos,
                radius: 3500,
                damage: 0,
                delay_ms: 0,
                duration_ms: 8000,
                tick_damage: 36, // damage per tick
                effect: SpellEffect::Poison,
            });
        }
        "Rage" => {
            engine.pending_spells.push(PendingSpell {
                player_id,
                name: "Rage".to_string(),
                pos,
                radius: 5000,
                damage: 0,
                delay_ms: 0,
                duration_ms: 7500,
                tick_damage: 0,
                effect: SpellEffect::Rage,
            });
        }
        "Tornado" => {
            engine.pending_spells.push(PendingSpell {
                player_id,
                name: "Tornado".to_string(),
                pos,
                radius: 5500,
                damage: 0,
                delay_ms: 0,
                duration_ms: 2500,
                tick_damage: 14,
                effect: SpellEffect::Tornado,
            });
        }
        "Log" => {
            engine.pending_spells.push(PendingSpell {
                player_id,
                name: "Log".to_string(),
                pos,
                radius: 3900,
                damage: 96,
                delay_ms: 400,
                duration_ms: 0,
                tick_damage: 0,
                effect: SpellEffect::Log,
            });
        }
        "Heal" => {
            engine.pending_spells.push(PendingSpell {
                player_id,
                name: "Heal".to_string(),
                pos,
                radius: 3000,
                damage: 0,
                delay_ms: 0,
                duration_ms: 2000,
                tick_damage: -65, // negative = healing
                effect: SpellEffect::Heal,
            });
        }
        "GoblinBarrel" => {
            // Spawns 3 goblins at the target position
            engine.spawn_troops(player_id, "Goblin", 3, pos);
        }
        "Graveyard" => {
            engine.pending_spells.push(PendingSpell {
                player_id,
                name: "Graveyard".to_string(),
                pos,
                radius: 5000,
                damage: 0,
                delay_ms: 0,
                duration_ms: 10000,
                tick_damage: 0,
                effect: SpellEffect::Graveyard,
            });
        }
        "Clone" => {
            // Clone all friendly troops in radius
            let radius_sq = (3000i64) * (3000i64);
            let mut clones: Vec<(u8, String, Pos)> = Vec::new();
            for t in &engine.troops {
                if t.player_id == player_id && t.is_alive() {
                    if t.pos.dist_sq(pos) <= radius_sq {
                        clones.push((player_id, t.character_name.clone(), t.pos));
                    }
                }
            }
            for (pid, name, cpos) in clones {
                if let Some(data) = engine.game_data.characters.get(&name) {
                    let mut cloned = engine.make_troop(pid, &data.clone(), cpos);
                    cloned.hp = 1;
                    cloned.max_hp = 1;
                    cloned.state = EntityState::Idle;
                    cloned.deploy_timer = 0;
                    engine.troops.push(cloned);
                }
            }
        }
        "Mirror" => {
            // Mirror: replay last card at +1 level cost
            // Simplified: do nothing (would need to track last played card)
        }
        _ => {}
    }
}

/// Update pending spell effects
pub fn update_pending_spells(engine: &mut BattleEngine) {
    let mut finished = Vec::new();
    let mut graveyard_spawns: Vec<(u8, Pos)> = Vec::new();

    for (idx, spell) in engine.pending_spells.iter_mut().enumerate() {
        // Count down delay
        if spell.delay_ms > 0 {
            spell.delay_ms -= MS_PER_TICK;
            if spell.delay_ms > 0 {
                continue;
            }
        }

        let enemy_id = 1 - spell.player_id;
        let radius_sq = (spell.radius as i64) * (spell.radius as i64);

        match spell.effect {
            SpellEffect::Damage => {
                // One-time area damage
                for troop in &mut engine.troops {
                    if troop.player_id == enemy_id && troop.is_alive() {
                        if troop.pos.dist_sq(spell.pos) <= radius_sq {
                            troop.hp -= spell.damage;
                        }
                    }
                }
                for building in &mut engine.buildings {
                    if building.player_id == enemy_id && building.is_alive() {
                        if building.pos.dist_sq(spell.pos) <= radius_sq {
                            building.hp -= spell.damage;
                        }
                    }
                }
                finished.push(idx);
            }
            SpellEffect::DamageAndStun => {
                // Damage + 0.5s stun
                for troop in &mut engine.troops {
                    if troop.player_id == enemy_id && troop.is_alive() {
                        if troop.pos.dist_sq(spell.pos) <= radius_sq {
                            troop.hp -= spell.damage;
                            troop.statuses.push(StatusEffect {
                                kind: StatusKind::Stun,
                                remaining_ms: 500,
                            });
                            // Reset inferno-type ramp damage
                            troop.attack_timer = troop.hit_speed_ms;
                        }
                    }
                }
                for building in &mut engine.buildings {
                    if building.player_id == enemy_id && building.is_alive() {
                        if building.pos.dist_sq(spell.pos) <= radius_sq {
                            building.hp -= spell.damage;
                        }
                    }
                }
                finished.push(idx);
            }
            SpellEffect::Freeze => {
                // Apply freeze status to all enemies in area
                if spell.duration_ms > 0 {
                    // First tick: apply initial damage + freeze
                    if spell.damage > 0 {
                        for troop in &mut engine.troops {
                            if troop.player_id == enemy_id && troop.is_alive() {
                                if troop.pos.dist_sq(spell.pos) <= radius_sq {
                                    troop.hp -= spell.damage;
                                    troop.statuses.push(StatusEffect {
                                        kind: StatusKind::Freeze,
                                        remaining_ms: spell.duration_ms,
                                    });
                                }
                            }
                        }
                        spell.damage = 0; // only damage once
                    }
                    spell.duration_ms -= MS_PER_TICK;
                    if spell.duration_ms <= 0 {
                        finished.push(idx);
                    }
                } else {
                    finished.push(idx);
                }
            }
            SpellEffect::Poison => {
                if spell.duration_ms > 0 {
                    // Apply poison tick damage
                    for troop in &mut engine.troops {
                        if troop.player_id == enemy_id && troop.is_alive() {
                            if troop.pos.dist_sq(spell.pos) <= radius_sq {
                                troop.hp -= spell.tick_damage;
                            }
                        }
                    }
                    for building in &mut engine.buildings {
                        if building.player_id == enemy_id && building.is_alive() {
                            if building.pos.dist_sq(spell.pos) <= radius_sq {
                                building.hp -= spell.tick_damage;
                            }
                        }
                    }
                    spell.duration_ms -= MS_PER_TICK;
                    if spell.duration_ms <= 0 {
                        finished.push(idx);
                    }
                } else {
                    finished.push(idx);
                }
            }
            SpellEffect::Rage => {
                if spell.duration_ms > 0 {
                    // Apply rage to friendly troops in area
                    for troop in &mut engine.troops {
                        if troop.player_id == spell.player_id && troop.is_alive() {
                            if troop.pos.dist_sq(spell.pos) <= radius_sq {
                                if !troop.statuses.iter().any(|s| matches!(s.kind, StatusKind::Rage(_))) {
                                    troop.statuses.push(StatusEffect {
                                        kind: StatusKind::Rage(140), // 40% speed boost
                                        remaining_ms: 500, // reapplied each tick
                                    });
                                }
                            }
                        }
                    }
                    spell.duration_ms -= MS_PER_TICK;
                    if spell.duration_ms <= 0 {
                        finished.push(idx);
                    }
                } else {
                    finished.push(idx);
                }
            }
            SpellEffect::Tornado => {
                if spell.duration_ms > 0 {
                    // Pull enemies toward center + tick damage
                    for troop in &mut engine.troops {
                        if troop.player_id == enemy_id && troop.is_alive() {
                            if troop.pos.dist_sq(spell.pos) <= radius_sq {
                                troop.hp -= spell.tick_damage;
                                // Pull toward center
                                troop.pos = troop.pos.move_toward(spell.pos, 100);
                            }
                        }
                    }
                    spell.duration_ms -= MS_PER_TICK;
                    if spell.duration_ms <= 0 {
                        finished.push(idx);
                    }
                } else {
                    finished.push(idx);
                }
            }
            SpellEffect::Heal => {
                if spell.duration_ms > 0 {
                    // Heal friendly troops
                    for troop in &mut engine.troops {
                        if troop.player_id == spell.player_id && troop.is_alive() {
                            if troop.pos.dist_sq(spell.pos) <= radius_sq {
                                troop.hp = (troop.hp + (-spell.tick_damage)).min(troop.max_hp);
                            }
                        }
                    }
                    spell.duration_ms -= MS_PER_TICK;
                    if spell.duration_ms <= 0 {
                        finished.push(idx);
                    }
                } else {
                    finished.push(idx);
                }
            }
            SpellEffect::Log => {
                // Damage + knockback in area
                for troop in &mut engine.troops {
                    if troop.player_id == enemy_id && troop.is_alive() && !troop.is_air() {
                        if troop.pos.dist_sq(spell.pos) <= radius_sq {
                            troop.hp -= spell.damage;
                            // Knockback: push away from spell center
                            let push_dir = if troop.pos.y > spell.pos.y { 1500 } else { -1500 };
                            troop.pos.y += push_dir;
                            troop.pos.y = troop.pos.y.clamp(200, crate::arena::ARENA_HEIGHT - 200);
                        }
                    }
                }
                finished.push(idx);
            }
            SpellEffect::Graveyard => {
                if spell.duration_ms > 0 {
                    // Spawn a skeleton every ~500ms in the area
                    if spell.duration_ms % 500 < MS_PER_TICK {
                        // Random position within radius
                        let angle = (spell.duration_ms as f32 * 0.7).sin();
                        let r = (spell.radius as f32) * 0.6;
                        let sx = spell.pos.x + (r * angle.cos()) as i32;
                        let sy = spell.pos.y + (r * angle.sin()) as i32;
                        graveyard_spawns.push((spell.player_id, Pos::new(sx, sy)));
                    }
                    spell.duration_ms -= MS_PER_TICK;
                    if spell.duration_ms <= 0 {
                        finished.push(idx);
                    }
                } else {
                    finished.push(idx);
                }
            }
            _ => {
                finished.push(idx);
            }
        }
    }

    // Remove finished spells (reverse order to maintain indices)
    finished.sort_unstable();
    finished.dedup();
    for idx in finished.into_iter().rev() {
        engine.pending_spells.remove(idx);
    }

    // Process graveyard spawns
    for (player_id, pos) in graveyard_spawns {
        engine.spawn_troops(player_id, "Skeleton", 1, pos);
    }
}
