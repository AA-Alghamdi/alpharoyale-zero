//! Arena geometry, positions, and pathfinding.
//!
//! CR arena is 18x32 tiles. Internal coordinates use millitiles (1 tile = 1000 millitiles).
//! Arena width: 18000 mt, height: 32000 mt.
//! River at y=15500-16500, bridges at x=3000-5000 and x=13000-15000.

/// Millitiles per tile
pub const MT_PER_TILE: i32 = 1000;

pub const ARENA_WIDTH: i32 = 18000;
pub const ARENA_HEIGHT: i32 = 32000;

// River boundaries
pub const RIVER_Y_MIN: i32 = 15000;
pub const RIVER_Y_MAX: i32 = 17000;

// Bridge positions (x ranges)
pub const LEFT_BRIDGE_X_MIN: i32 = 2500;
pub const LEFT_BRIDGE_X_MAX: i32 = 5500;
pub const RIGHT_BRIDGE_X_MIN: i32 = 12500;
pub const RIGHT_BRIDGE_X_MAX: i32 = 15500;

// Tower positions (from RetroRoyale's Encode method)
pub const P0_KING_X: i32 = 9000;
pub const P0_KING_Y: i32 = 3000;
pub const P0_LEFT_PRINCESS_X: i32 = 3500;
pub const P0_LEFT_PRINCESS_Y: i32 = 6500;
pub const P0_RIGHT_PRINCESS_X: i32 = 14500;
pub const P0_RIGHT_PRINCESS_Y: i32 = 6500;

pub const P1_KING_X: i32 = 9000;
pub const P1_KING_Y: i32 = 29000;
pub const P1_LEFT_PRINCESS_X: i32 = 3500;
pub const P1_LEFT_PRINCESS_Y: i32 = 25500;
pub const P1_RIGHT_PRINCESS_X: i32 = 14500;
pub const P1_RIGHT_PRINCESS_Y: i32 = 25500;

/// Position in millitiles
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
}

impl Pos {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Squared distance (avoids sqrt for comparisons)
    pub fn dist_sq(self, other: Pos) -> i64 {
        let dx = (self.x - other.x) as i64;
        let dy = (self.y - other.y) as i64;
        dx * dx + dy * dy
    }

    /// Actual distance in millitiles
    pub fn dist(self, other: Pos) -> f32 {
        (self.dist_sq(other) as f32).sqrt()
    }

    /// Move toward target by `step` millitiles, returns new position
    pub fn move_toward(self, target: Pos, step: i32) -> Pos {
        let dx = target.x - self.x;
        let dy = target.y - self.y;
        let d = ((dx as i64 * dx as i64 + dy as i64 * dy as i64) as f32).sqrt();
        if d < 1.0 {
            return target;
        }
        let ratio = step as f32 / d;
        if ratio >= 1.0 {
            return target;
        }
        Pos {
            x: self.x + (dx as f32 * ratio) as i32,
            y: self.y + (dy as f32 * ratio) as i32,
        }
    }
}

/// Check if a position is on a bridge
pub fn on_bridge(x: i32) -> bool {
    (x >= LEFT_BRIDGE_X_MIN && x <= LEFT_BRIDGE_X_MAX)
        || (x >= RIGHT_BRIDGE_X_MIN && x <= RIGHT_BRIDGE_X_MAX)
}

/// Check if a position is in the river zone
pub fn in_river(y: i32) -> bool {
    y >= RIVER_Y_MIN && y <= RIVER_Y_MAX
}

/// Check if a ground entity can pass through this position
pub fn is_passable_ground(x: i32, y: i32) -> bool {
    if x < 0 || x > ARENA_WIDTH || y < 0 || y > ARENA_HEIGHT {
        return false;
    }
    if in_river(y) && !on_bridge(x) {
        return false;
    }
    true
}

/// Get the nearest bridge waypoint for crossing the river
pub fn nearest_bridge_waypoint(from: Pos, going_up: bool) -> Pos {
    let bridge_y = if going_up { RIVER_Y_MAX + 500 } else { RIVER_Y_MIN - 500 };

    let left_bridge_center = (LEFT_BRIDGE_X_MIN + LEFT_BRIDGE_X_MAX) / 2;
    let right_bridge_center = (RIGHT_BRIDGE_X_MIN + RIGHT_BRIDGE_X_MAX) / 2;

    let left_dist = ((from.x - left_bridge_center) as i64).abs();
    let right_dist = ((from.x - right_bridge_center) as i64).abs();

    if left_dist <= right_dist {
        Pos::new(left_bridge_center, bridge_y)
    } else {
        Pos::new(right_bridge_center, bridge_y)
    }
}

/// Get deployment zone for a player (valid Y range for placing cards)
pub fn deploy_zone(player_id: u8) -> (i32, i32) {
    if player_id == 0 {
        // Player 0 deploys in bottom half
        (500, RIVER_Y_MIN - 500)
    } else {
        // Player 1 deploys in top half
        (RIVER_Y_MAX + 500, ARENA_HEIGHT - 500)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dist() {
        let a = Pos::new(0, 0);
        let b = Pos::new(3000, 4000);
        assert!((a.dist(b) - 5000.0).abs() < 1.0);
    }

    #[test]
    fn test_bridge() {
        assert!(on_bridge(4000));
        assert!(on_bridge(14000));
        assert!(!on_bridge(9000));
    }

    #[test]
    fn test_passable() {
        assert!(is_passable_ground(9000, 5000)); // normal ground
        assert!(!is_passable_ground(9000, 16000)); // river, no bridge
        assert!(is_passable_ground(4000, 16000)); // on bridge
    }
}
