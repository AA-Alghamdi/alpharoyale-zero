//! Entity types: Troops, Buildings, Projectiles, and Spells.

use crate::arena::Pos;

pub type EntityId = u32;

/// What kind of target this entity can attack
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetMode {
    Ground,
    Air,
    Both,
    Buildings, // e.g. Giant, Hog Rider
}

/// Movement type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MoveType {
    Ground,
    Air,
}

/// Entity state in the simulation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntityState {
    Deploying,   // waiting for deploy time
    Idle,        // no target, moving forward
    Moving,      // has target, moving toward it
    Attacking,   // in range, attacking
    Dead,
}

/// Status effect on an entity
#[derive(Debug, Clone)]
pub struct StatusEffect {
    pub kind: StatusKind,
    pub remaining_ms: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StatusKind {
    Stun,
    Freeze,
    Slow(i32),    // speed multiplier pct (e.g., 35 = 35% speed)
    Rage(i32),    // speed boost pct (e.g., 140 = 140% speed)
    Poison(i32),  // damage per second
    Shield(i32),  // shield HP remaining
}

/// A troop entity
#[derive(Debug, Clone)]
pub struct Troop {
    pub id: EntityId,
    pub character_name: String,
    pub player_id: u8,
    pub pos: Pos,
    pub state: EntityState,
    pub move_type: MoveType,
    pub target_mode: TargetMode,

    // Stats (from CSV data)
    pub max_hp: i32,
    pub hp: i32,
    pub damage: i32,
    pub range: i32,           // millitiles
    pub sight_range: i32,     // millitiles
    pub speed: i32,           // millitiles per tick
    pub hit_speed_ms: i32,    // ms between attacks
    pub load_time_ms: i32,    // ms for first hit to land
    pub deploy_time_ms: i32,  // ms before unit becomes active
    pub area_damage_radius: i32,
    pub collision_radius: i32,
    pub mass: i32,
    pub attack_pushback: i32,
    pub crown_tower_damage_pct: i32,

    // Death effects
    pub death_damage: i32,
    pub death_damage_radius: i32,
    pub death_spawn: String,  // character name to spawn on death
    pub death_spawn_count: i32,

    // Charge mechanic
    pub charge_range: i32,
    pub is_charging: bool,
    pub charge_damage_mult: i32, // percentage, 200 = double damage

    // Timers (in ms)
    pub deploy_timer: i32,
    pub attack_timer: i32,     // counts down to next attack
    pub lifetime_timer: i32,   // for units with limited lifetime

    // Targeting
    pub target_id: Option<EntityId>,
    pub target_pos: Option<Pos>, // for movement waypoint

    // Status effects
    pub statuses: Vec<StatusEffect>,

    // Shield (Dark Prince, Guards)
    pub shield_hp: i32,

    // Projectile info
    pub projectile_name: String,
    pub multiple_projectiles: i32,

    // Spawner (Witch summons Skeletons)
    pub summon_character: String,
    pub summon_number: i32,
    pub summon_timer: i32,
    pub summon_interval_ms: i32,
}

impl Troop {
    /// Effective speed considering status effects
    pub fn effective_speed(&self) -> i32 {
        if self.statuses.iter().any(|s| matches!(s.kind, StatusKind::Stun | StatusKind::Freeze)) {
            return 0;
        }
        let mut speed = self.speed;
        for status in &self.statuses {
            match status.kind {
                StatusKind::Slow(pct) => speed = speed * pct / 100,
                StatusKind::Rage(pct) => speed = speed * pct / 100,
                _ => {}
            }
        }
        speed
    }

    /// Effective attack speed considering status effects
    pub fn effective_hit_speed(&self) -> i32 {
        let mut hs = self.hit_speed_ms;
        for status in &self.statuses {
            match status.kind {
                StatusKind::Rage(pct) => hs = hs * 100 / pct,
                _ => {}
            }
        }
        hs.max(100) // minimum 100ms
    }

    pub fn is_air(&self) -> bool {
        self.move_type == MoveType::Air
    }

    pub fn is_alive(&self) -> bool {
        self.state != EntityState::Dead
    }

    pub fn is_stunned(&self) -> bool {
        self.statuses.iter().any(|s| matches!(s.kind, StatusKind::Stun | StatusKind::Freeze))
    }

    pub fn can_attack_air(&self) -> bool {
        matches!(self.target_mode, TargetMode::Air | TargetMode::Both)
    }

    pub fn can_attack_ground(&self) -> bool {
        matches!(self.target_mode, TargetMode::Ground | TargetMode::Both | TargetMode::Buildings)
    }

    pub fn targets_only_buildings(&self) -> bool {
        self.target_mode == TargetMode::Buildings
    }
}

/// A building entity (towers, spawners, etc.)
#[derive(Debug, Clone)]
pub struct Building {
    pub id: EntityId,
    pub building_name: String,
    pub player_id: u8,
    pub pos: Pos,
    pub state: EntityState,

    pub max_hp: i32,
    pub hp: i32,
    pub damage: i32,
    pub range: i32,
    pub hit_speed_ms: i32,
    pub lifetime_timer: i32, // ms, 0 = permanent (king/princess towers)

    pub attack_timer: i32,
    pub target_id: Option<EntityId>,
    pub target_mode: TargetMode,

    pub is_tower: bool,        // king or princess tower
    pub tower_type: TowerType,
    pub statuses: Vec<StatusEffect>,

    // Spawner buildings (Goblin Hut, Tombstone, etc.)
    pub summon_character: String,
    pub summon_number: i32,
    pub summon_timer: i32,
    pub summon_interval_ms: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TowerType {
    King,
    LeftPrincess,
    RightPrincess,
    None,
}

impl Building {
    pub fn is_alive(&self) -> bool {
        self.state != EntityState::Dead
    }
}

/// A projectile in flight
#[derive(Debug, Clone)]
pub struct Projectile {
    pub id: EntityId,
    pub player_id: u8,
    pub pos: Pos,
    pub target_pos: Pos,
    pub target_id: Option<EntityId>,
    pub speed: i32,        // millitiles per tick
    pub damage: i32,
    pub splash_radius: i32, // 0 = single target
    pub pushback: i32,
    pub homing: bool,
    pub is_spell: bool,     // true for Fireball, Arrows, etc.
    pub spell_name: String, // for spells: identifies special effects
}

/// Pending spell effect (area damage after delay)
#[derive(Debug, Clone)]
pub struct PendingSpell {
    pub player_id: u8,
    pub name: String,
    pub pos: Pos,
    pub radius: i32,
    pub damage: i32,
    pub delay_ms: i32,     // ms until effect triggers
    pub duration_ms: i32,  // for persistent spells (Poison, Graveyard)
    pub tick_damage: i32,  // damage per tick for persistent spells
    pub effect: SpellEffect,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpellEffect {
    Damage,          // Fireball, Arrows, Rocket
    DamageAndStun,   // Zap, Lightning
    Freeze,
    Rage,
    Poison,
    Tornado,
    Heal,
    Clone,
    Graveyard,       // spawns skeletons over time
    Log,             // pushback + damage in a line
}
