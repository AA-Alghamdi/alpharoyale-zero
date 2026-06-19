//! Movement system: pathfinding, bridge crossing, collision.

use crate::arena::*;
use crate::engine::BattleEngine;
use crate::entity::*;

/// Update movement for all troops
pub fn update_movement(engine: &mut BattleEngine) {
    // Collect target positions for all entities (to avoid borrow issues)
    let pos_map: std::collections::HashMap<EntityId, Pos> = engine.troops.iter()
        .filter(|t| t.is_alive())
        .map(|t| (t.id, t.pos))
        .chain(engine.buildings.iter().filter(|b| b.is_alive()).map(|b| (b.id, b.pos)))
        .collect();

    for troop in &mut engine.troops {
        if !troop.is_alive() || troop.state == EntityState::Deploying || troop.state == EntityState::Attacking {
            continue;
        }
        if troop.is_stunned() {
            continue;
        }

        let speed = troop.effective_speed();
        if speed == 0 {
            continue;
        }

        // Get target position from the snapshot
        let target_pos = troop.target_id.and_then(|tid| pos_map.get(&tid).copied());

        let dest = if let Some(tp) = target_pos {
            if troop.move_type == MoveType::Ground {
                compute_ground_path(troop.pos, tp, troop.player_id)
            } else {
                tp
            }
        } else {
            // No target — walk toward enemy side
            let forward_y = if troop.player_id == 0 {
                troop.pos.y + 1000
            } else {
                troop.pos.y - 1000
            };
            Pos::new(troop.pos.x, forward_y)
        };

        // Charge mechanics
        if troop.charge_range > 0 && target_pos.is_some() {
            let tp = target_pos.unwrap();
            let dist_to_target = troop.pos.dist(tp);
            if dist_to_target <= troop.charge_range as f32 && !troop.is_charging {
                troop.is_charging = true;
                let charge_pos = troop.pos.move_toward(tp, speed * 2);
                troop.pos = Pos::new(
                    charge_pos.x.clamp(200, ARENA_WIDTH - 200),
                    charge_pos.y.clamp(200, ARENA_HEIGHT - 200),
                );
                continue;
            }
        }

        // Normal movement
        let new_pos = troop.pos.move_toward(dest, speed);
        let clamped = Pos::new(
            new_pos.x.clamp(200, ARENA_WIDTH - 200),
            new_pos.y.clamp(200, ARENA_HEIGHT - 200),
        );

        // Ground units can't walk through river
        if troop.move_type == MoveType::Ground && in_river(clamped.y) && !on_bridge(clamped.x) {
            continue;
        }

        troop.pos = clamped;

        // Check if in range of target
        if let Some(tp) = target_pos {
            let dist = troop.pos.dist(tp);
            if dist <= troop.range as f32 + troop.collision_radius as f32 {
                troop.state = EntityState::Attacking;
            } else {
                troop.state = EntityState::Moving;
            }
        }
    }
}

/// Compute ground path from current position to target, routing through bridges if needed
fn compute_ground_path(from: Pos, to: Pos, player_id: u8) -> Pos {
    let going_up = player_id == 0;

    let crosses_river = if going_up {
        from.y < RIVER_Y_MIN && to.y > RIVER_Y_MAX
    } else {
        from.y > RIVER_Y_MAX && to.y < RIVER_Y_MIN
    };

    if !crosses_river {
        return to;
    }

    if (going_up && from.y < RIVER_Y_MIN) || (!going_up && from.y > RIVER_Y_MAX) {
        let bridge = nearest_bridge_waypoint(from, going_up);
        if from.dist(bridge) < 500.0 {
            return to;
        }
        return bridge;
    }

    to
}
