//! Movement system: pathfinding, bridge crossing, collision.

use crate::arena::*;
use crate::engine::BattleEngine;
use crate::entity::*;

/// Update movement for all troops
pub fn update_movement(engine: &mut BattleEngine) {
    // Collect target positions snapshot
    let pos_snapshot: Vec<(EntityId, Pos)> = engine.troops.iter()
        .filter(|t| t.is_alive())
        .map(|t| (t.id, t.pos))
        .chain(engine.buildings.iter().filter(|b| b.is_alive()).map(|b| (b.id, b.pos)))
        .collect();

    #[inline]
    fn lookup(snapshot: &[(EntityId, Pos)], id: EntityId) -> Option<Pos> {
        snapshot.iter().find(|(eid, _)| *eid == id).map(|(_, p)| *p)
    }

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

        let target_pos = troop.target_id.and_then(|tid| lookup(&pos_snapshot, tid));

        // Determine immediate movement destination
        let dest = if troop.move_type == MoveType::Air {
            // Air units fly directly
            target_pos.unwrap_or_else(|| {
                let forward_y = if troop.player_id == 0 { troop.pos.y + 1000 } else { troop.pos.y - 1000 };
                Pos::new(troop.pos.x, forward_y)
            })
        } else {
            // Ground units need bridge routing
            let final_target = target_pos.unwrap_or_else(|| {
                let forward_y = if troop.player_id == 0 { ARENA_HEIGHT - 3000 } else { 3000 };
                Pos::new(troop.pos.x, forward_y)
            });
            compute_bridge_route(troop.pos, final_target, troop.player_id)
        };

        // Charge mechanics
        if troop.charge_range > 0 && !troop.is_charging {
            if let Some(tp) = target_pos {
                let dist_to_target = troop.pos.dist(tp);
                if dist_to_target <= troop.charge_range as f32 {
                    troop.is_charging = true;
                    let charge_pos = troop.pos.move_toward(tp, speed * 2);
                    troop.pos = Pos::new(
                        charge_pos.x.clamp(200, ARENA_WIDTH - 200),
                        charge_pos.y.clamp(200, ARENA_HEIGHT - 200),
                    );
                    continue;
                }
            }
        }

        let new_pos = troop.pos.move_toward(dest, speed);
        troop.pos = Pos::new(
            new_pos.x.clamp(200, ARENA_WIDTH - 200),
            new_pos.y.clamp(200, ARENA_HEIGHT - 200),
        );

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

/// Route ground units through bridges. Returns the immediate waypoint to move toward.
fn compute_bridge_route(from: Pos, to: Pos, player_id: u8) -> Pos {
    let going_up = player_id == 0;

    // Check if path crosses the river
    let needs_bridge = if going_up {
        from.y < RIVER_Y_MIN && to.y > RIVER_Y_MIN
    } else {
        from.y > RIVER_Y_MAX && to.y < RIVER_Y_MAX
    };

    if !needs_bridge {
        // If we're IN the river on a bridge, keep going straight through
        if in_river(from.y) && on_bridge(from.x) {
            return to;
        }
        return to;
    }

    // We need to cross — find the nearest bridge
    let left_bridge_center = (LEFT_BRIDGE_X_MIN + LEFT_BRIDGE_X_MAX) / 2; // 4000
    let right_bridge_center = (RIGHT_BRIDGE_X_MIN + RIGHT_BRIDGE_X_MAX) / 2; // 14000

    let bridge_x = if (from.x - left_bridge_center).abs() <= (from.x - right_bridge_center).abs() {
        left_bridge_center
    } else {
        right_bridge_center
    };

    // Route through 3 waypoints:
    // 1. Approach bridge at own side of river
    // 2. Cross bridge (stay on bridge x while traversing river y range)
    // 3. Exit on the other side

    if going_up {
        // Phase 1: Not at bridge x yet — go to bridge entrance
        if (from.x - bridge_x).abs() > 300 || from.y < RIVER_Y_MIN - 500 {
            return Pos::new(bridge_x, RIVER_Y_MIN - 200);
        }
        // Phase 2: At bridge entrance — cross the river staying on bridge_x
        if from.y <= RIVER_Y_MAX + 200 {
            return Pos::new(bridge_x, RIVER_Y_MAX + 500);
        }
        // Phase 3: Past the river — go to target
        return to;
    } else {
        // Phase 1: Not at bridge x yet — go to bridge entrance
        if (from.x - bridge_x).abs() > 300 || from.y > RIVER_Y_MAX + 500 {
            return Pos::new(bridge_x, RIVER_Y_MAX + 200);
        }
        // Phase 2: At bridge entrance — cross the river staying on bridge_x
        if from.y >= RIVER_Y_MIN - 200 {
            return Pos::new(bridge_x, RIVER_Y_MIN - 500);
        }
        // Phase 3: Past the river — go to target
        return to;
    }
}
