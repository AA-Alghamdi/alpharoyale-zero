//! Core battle engine — the tick loop that drives everything.

use std::path::Path;

use crate::arena::*;
use crate::combat;
use crate::data::{scale_stat, CharacterData, GameData};
use crate::entity::*;
use crate::movement;
use crate::spells;

/// Milliseconds per tick. CR runs at ~20 ticks/sec in battle.
pub const MS_PER_TICK: i32 = 50;

/// Ticks per second
pub const TICKS_PER_SEC: i32 = 1000 / MS_PER_TICK;

/// Battle duration in ticks (3 minutes regular + 1 min overtime + 1 min sudden death)
pub const REGULAR_TIME_TICKS: i32 = 180 * TICKS_PER_SEC;       // 3 minutes
pub const OVERTIME_START_TICKS: i32 = REGULAR_TIME_TICKS;       // 3:00
pub const DOUBLE_ELIXIR_TICKS: i32 = 120 * TICKS_PER_SEC;      // 2:00
pub const OVERTIME_END_TICKS: i32 = 240 * TICKS_PER_SEC;        // 4:00
pub const SUDDEN_DEATH_TICKS: i32 = 300 * TICKS_PER_SEC;        // 5:00
pub const MAX_TICKS: i32 = 360 * TICKS_PER_SEC;                 // 6:00 tiebreaker

/// Elixir constants
pub const MAX_ELIXIR: f32 = 10.0;
pub const ELIXIR_PER_TICK: f32 = 1.0 / (2.8 * TICKS_PER_SEC as f32); // 1 elixir per 2.8 sec
pub const DOUBLE_ELIXIR_MULT: f32 = 2.0;
pub const TRIPLE_ELIXIR_MULT: f32 = 3.0;

/// Player state
#[derive(Debug, Clone)]
pub struct PlayerState {
    pub player_id: u8,
    pub elixir: f32,
    pub deck: Vec<String>,        // 8 card names
    pub hand: [usize; 4],         // indices into deck
    pub next_card: usize,         // deck index of next card
    pub crowns: i32,
    pub king_activated: bool,
}

impl PlayerState {
    pub fn new(player_id: u8, deck: Vec<String>) -> Self {
        assert!(deck.len() == 8, "Deck must have 8 cards");
        Self {
            player_id,
            elixir: 5.0, // start with 5 elixir
            hand: [0, 1, 2, 3],
            next_card: 4,
            crowns: 0,
            king_activated: false,
            deck,
        }
    }

    pub fn cycle_card(&mut self, hand_index: usize) {
        self.hand[hand_index] = self.next_card;
        self.next_card = (self.next_card + 1) % self.deck.len();
    }

    pub fn card_name(&self, hand_index: usize) -> &str {
        &self.deck[self.hand[hand_index]]
    }
}

/// The complete battle state
#[derive(Debug, Clone)]
pub struct BattleEngine {
    pub game_data: GameData,
    pub tick: i32,
    pub players: [PlayerState; 2],
    pub troops: Vec<Troop>,
    pub buildings: Vec<Building>,
    pub projectiles: Vec<Projectile>,
    pub pending_spells: Vec<PendingSpell>,
    pub next_entity_id: EntityId,
    pub game_over: bool,
    pub winner: Option<u8>,  // 0, 1, or None for draw
    pub overtime: bool,
}

/// Command from an agent
#[derive(Debug, Clone)]
pub struct Command {
    pub player_id: u8,
    pub card_hand_index: usize, // 0-3
    pub x: i32,                 // millitiles
    pub y: i32,                 // millitiles
}

impl BattleEngine {
    /// Create a new battle
    pub fn new(game_data: GameData, deck0: Vec<String>, deck1: Vec<String>) -> Self {
        let mut engine = Self {
            game_data,
            tick: 0,
            players: [
                PlayerState::new(0, deck0),
                PlayerState::new(1, deck1),
            ],
            troops: Vec::with_capacity(128),
            buildings: Vec::with_capacity(32),
            projectiles: Vec::with_capacity(64),
            pending_spells: Vec::with_capacity(16),
            next_entity_id: 100,
            game_over: false,
            winner: None,
            overtime: false,
        };
        engine.create_towers();
        engine
    }

    /// Load game data from a directory at Level 1 (raw stats).
    pub fn from_data_dir(data_dir: &Path, deck0: Vec<String>, deck1: Vec<String>) -> Result<Self, String> {
        let game_data = GameData::load(data_dir)?;
        Ok(Self::new(game_data, deck0, deck1))
    }

    /// Load game data from a directory scaled to a target level.
    pub fn from_data_dir_at_level(data_dir: &Path, deck0: Vec<String>, deck1: Vec<String>, level: i32) -> Result<Self, String> {
        let game_data = GameData::load_at_level(data_dir, level)?;
        Ok(Self::new(game_data, deck0, deck1))
    }

    fn next_id(&mut self) -> EntityId {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        id
    }

    fn create_towers(&mut self) {
        // Towers have their own stat table, not the same as troop scaling.
        // We store the known Level 11 (Tournament Standard) values directly.
        // Source: Clash Royale wiki, verified against game client.
        let (king_hp, king_dmg, princess_hp, princess_dmg) = match self.game_data.level {
            11 => (4008, 109, 2534, 109),
            14 => (5544, 152, 3514, 152),
            1  => (2400, 50, 1400, 50),
            _  => {
                // Approximate: interpolate between L1 and L14
                let t = (self.game_data.level - 1).max(0) as f64 / 13.0;
                let interp = |lo: i32, hi: i32| -> i32 { (lo as f64 + t * (hi - lo) as f64).round() as i32 };
                (interp(2400, 5544), interp(50, 152), interp(1400, 3514), interp(50, 152))
            }
        };

        // Player 0 towers (bottom)
        let t0k = Self::make_tower_static(&mut self.next_entity_id, 0, TowerType::King, Pos::new(P0_KING_X, P0_KING_Y), king_hp, king_dmg, 7000);
        let t0l = Self::make_tower_static(&mut self.next_entity_id, 0, TowerType::LeftPrincess, Pos::new(P0_LEFT_PRINCESS_X, P0_LEFT_PRINCESS_Y), princess_hp, princess_dmg, 7500);
        let t0r = Self::make_tower_static(&mut self.next_entity_id, 0, TowerType::RightPrincess, Pos::new(P0_RIGHT_PRINCESS_X, P0_RIGHT_PRINCESS_Y), princess_hp, princess_dmg, 7500);
        // Player 1 towers (top)
        let t1k = Self::make_tower_static(&mut self.next_entity_id, 1, TowerType::King, Pos::new(P1_KING_X, P1_KING_Y), king_hp, king_dmg, 7000);
        let t1l = Self::make_tower_static(&mut self.next_entity_id, 1, TowerType::LeftPrincess, Pos::new(P1_LEFT_PRINCESS_X, P1_LEFT_PRINCESS_Y), princess_hp, princess_dmg, 7500);
        let t1r = Self::make_tower_static(&mut self.next_entity_id, 1, TowerType::RightPrincess, Pos::new(P1_RIGHT_PRINCESS_X, P1_RIGHT_PRINCESS_Y), princess_hp, princess_dmg, 7500);
        self.buildings.extend([t0k, t0l, t0r, t1k, t1l, t1r]);
    }

    fn make_tower_static(next_id: &mut EntityId, player_id: u8, tower_type: TowerType, pos: Pos, hp: i32, damage: i32, range: i32) -> Building {
        let id = *next_id;
        *next_id += 1;
        let name = match tower_type {
            TowerType::King => "KingTower",
            TowerType::LeftPrincess | TowerType::RightPrincess => "PrincessTower",
            TowerType::None => "Building",
        };
        Building {
            id,
            building_name: name.to_string(),
            player_id,
            pos,
            state: if tower_type == TowerType::King { EntityState::Idle } else { EntityState::Attacking },
            max_hp: hp,
            hp,
            damage,
            range,
            hit_speed_ms: 800,
            lifetime_timer: 0,
            attack_timer: 0,
            target_id: None,
            target_mode: TargetMode::Both,
            is_tower: true,
            tower_type,
            statuses: Vec::new(),
            summon_character: String::new(),
            summon_number: 0,
            summon_timer: 0,
            summon_interval_ms: 0,
        }
    }

    /// Process a command (play a card)
    pub fn execute_command(&mut self, cmd: &Command) -> bool {
        let player = &self.players[cmd.player_id as usize];
        let card_name = player.card_name(cmd.card_hand_index).to_string();

        // Check deploy zone
        let (y_min, y_max) = deploy_zone(cmd.player_id);
        let deploy_y = cmd.y.clamp(y_min, y_max);
        let deploy_x = cmd.x.clamp(500, ARENA_WIDTH - 500);
        let deploy_pos = Pos::new(deploy_x, deploy_y);

        // Clone data lookups to avoid borrow conflicts with self
        let spell_char = self.game_data.spell_characters.get(&card_name).cloned();
        let spell_other = self.game_data.spell_others.get(&card_name).cloned();
        let spell_building = self.game_data.spell_buildings.get(&card_name).cloned();

        // Check if it's a character spell
        if let Some(spell_data) = spell_char {
            let cost = spell_data.mana_cost;
            if self.players[cmd.player_id as usize].elixir < cost as f32 {
                return false;
            }
            self.players[cmd.player_id as usize].elixir -= cost as f32;

            self.spawn_troops(cmd.player_id, &spell_data.summon_character, spell_data.summon_number, deploy_pos);

            if !spell_data.summon_character_second.is_empty() && spell_data.summon_character_second_count > 0 {
                self.spawn_troops(cmd.player_id, &spell_data.summon_character_second, spell_data.summon_character_second_count, deploy_pos);
            }

            self.players[cmd.player_id as usize].cycle_card(cmd.card_hand_index);
            return true;
        }

        // Check if it's an other spell (Fireball, Arrows, etc.)
        if let Some(spell_data) = spell_other {
            let cost = spell_data.mana_cost;
            if self.players[cmd.player_id as usize].elixir < cost as f32 {
                return false;
            }
            self.players[cmd.player_id as usize].elixir -= cost as f32;

            spells::cast_spell(self, cmd.player_id, &card_name, deploy_pos);

            self.players[cmd.player_id as usize].cycle_card(cmd.card_hand_index);
            return true;
        }

        // Check if it's a building spell
        if let Some(building_data) = spell_building {
            let cost = building_data.mana_cost;
            if self.players[cmd.player_id as usize].elixir < cost as f32 {
                return false;
            }
            self.players[cmd.player_id as usize].elixir -= cost as f32;

            self.spawn_building(cmd.player_id, &building_data.summon_character, deploy_pos);

            self.players[cmd.player_id as usize].cycle_card(cmd.card_hand_index);
            return true;
        }

        false
    }

    /// Spawn troops at a position (with slight spread for multiples)
    pub fn spawn_troops(&mut self, player_id: u8, char_name: &str, count: i32, center: Pos) {
        let char_data = match self.game_data.characters.get(char_name) {
            Some(d) => d.clone(),
            None => return,
        };

        for i in 0..count {
            // Spread units slightly
            let offset_x = if count > 1 { (i - count / 2) * 300 } else { 0 };
            let offset_y = if count > 2 { (i / 2) * 300 - 150 } else { 0 };
            let pos = Pos::new(
                (center.x + offset_x).clamp(500, ARENA_WIDTH - 500),
                center.y + offset_y,
            );

            let troop = self.make_troop(player_id, &char_data, pos);
            self.troops.push(troop);
        }
    }

    pub fn make_troop(&mut self, player_id: u8, data: &CharacterData, pos: Pos) -> Troop {
        let id = self.next_id();
        let move_type = if data.flying_height > 0 { MoveType::Air } else { MoveType::Ground };
        let target_mode = if data.target_only_buildings {
            TargetMode::Buildings
        } else if data.attacks_air && data.attacks_ground {
            TargetMode::Both
        } else if data.attacks_air {
            TargetMode::Air
        } else {
            TargetMode::Ground
        };

        // Speed conversion: CSV speed value → millitiles per tick
        // CR speed values: 45=slow, 60=medium, 90=fast, 120=very fast
        // At 20 ticks/sec, a "medium" (60) unit moves ~60 tiles in ~20 seconds = 3 tiles/sec = 150 mt/tick
        let speed_mt_per_tick = data.speed * 5 / 2;

        // Determine death spawn from character data
        let (death_spawn, death_spawn_count) = self.get_death_spawn(&data.name);

        Troop {
            id,
            character_name: data.name.clone(),
            player_id,
            pos,
            state: EntityState::Deploying,
            move_type,
            target_mode,
            max_hp: data.hitpoints,
            hp: data.hitpoints,
            damage: data.damage,
            range: data.range,
            sight_range: data.sight_range,
            speed: speed_mt_per_tick,
            hit_speed_ms: data.hit_speed,
            load_time_ms: data.load_time,
            deploy_time_ms: data.deploy_time,
            area_damage_radius: data.area_damage_radius,
            collision_radius: data.collision_radius,
            mass: data.mass,
            attack_pushback: data.attack_pushback,
            crown_tower_damage_pct: data.crown_tower_damage_pct,
            death_damage: data.death_damage,
            death_damage_radius: data.death_damage_radius,
            death_spawn,
            death_spawn_count,
            charge_range: data.charge_range,
            is_charging: false,
            charge_damage_mult: 200, // Prince does 2x on charge
            deploy_timer: data.deploy_time,
            attack_timer: 0,
            lifetime_timer: data.lifetime,
            target_id: None,
            target_pos: None,
            statuses: Vec::new(),
            shield_hp: 0, // set by specific card logic
            projectile_name: data.projectile.clone(),
            multiple_projectiles: data.multiple_projectiles,
            summon_character: data.summon_character.clone(),
            summon_number: data.summon_number,
            summon_timer: 0,
            summon_interval_ms: data.special_attack_interval,
        }
    }

    fn get_death_spawn(&self, name: &str) -> (String, i32) {
        match name {
            "Golem" => ("Golemite".to_string(), 2),
            "LavaHound" => ("LavaPups".to_string(), 6),
            "GiantSkeleton" => (String::new(), 0), // death DAMAGE, not spawn
            "SkeletonBalloon" => ("Skeleton".to_string(), 3),
            "BattleRam" => ("Barbarian".to_string(), 2),
            _ => (String::new(), 0),
        }
    }

    fn spawn_building(&mut self, player_id: u8, building_name: &str, pos: Pos) {
        let id = self.next_id();
        // Try to get building stats from characters table
        let (hp, damage, range, hit_speed, lifetime, target_mode, summon_char, summon_num, summon_interval) =
            match building_name {
                "Cannon" => (742, 92, 5500, 800, 30000, TargetMode::Ground, "", 0, 0),
                "Tesla" => (452, 96, 5500, 800, 35000, TargetMode::Both, "", 0, 0),
                "InfernoTower" => (800, 20, 6000, 400, 35000, TargetMode::Both, "", 0, 0),
                "BombTower" => (900, 100, 6000, 1800, 35000, TargetMode::Ground, "", 0, 0),
                "Mortar" => (612, 108, 11500, 5000, 30000, TargetMode::Ground, "", 0, 0),
                "Xbow" => (850, 26, 11500, 250, 40000, TargetMode::Both, "", 0, 0),
                "GoblinHut" => (640, 0, 0, 0, 60000, TargetMode::Ground, "SpearGoblin", 2, 5000),
                "BarbarianHut" => (960, 0, 0, 0, 60000, TargetMode::Ground, "Barbarian", 2, 14000),
                "Tombstone" => (240, 0, 0, 0, 40000, TargetMode::Ground, "Skeleton", 1, 2900),
                "FirespiritHut" => (480, 0, 0, 0, 50000, TargetMode::Ground, "FireSpirits", 2, 10000),
                "ElixirCollector" => (590, 0, 0, 0, 70000, TargetMode::Ground, "", 0, 0),
                _ => (500, 50, 5500, 1000, 30000, TargetMode::Both, "", 0, 0),
            };

        self.buildings.push(Building {
            id,
            building_name: building_name.to_string(),
            player_id,
            pos,
            state: EntityState::Deploying,
            max_hp: hp,
            hp,
            damage,
            range,
            hit_speed_ms: hit_speed,
            lifetime_timer: lifetime,
            attack_timer: 0,
            target_id: None,
            target_mode,
            is_tower: false,
            tower_type: TowerType::None,
            statuses: Vec::new(),
            summon_character: summon_char.to_string(),
            summon_number: summon_num,
            summon_timer: summon_interval, // first spawn after interval
            summon_interval_ms: summon_interval,
        });
    }

    /// Advance the simulation by one tick
    pub fn step(&mut self) {
        if self.game_over {
            return;
        }

        self.tick += 1;

        // 1. Regenerate elixir
        self.update_elixir();

        // 2. Update deploy timers
        self.update_deploy_timers();

        // 3. Update status effects
        self.update_status_effects();

        // 4. Targeting
        combat::update_targeting(self);

        // 5. Movement
        movement::update_movement(self);

        // 6. Combat (attacks)
        combat::update_combat(self);

        // 7. Update projectiles
        combat::update_projectiles(self);

        // 8. Process pending spells
        spells::update_pending_spells(self);

        // 9. Process deaths (death damage, death spawns)
        self.process_deaths();

        // 10. Update building spawners
        self.update_spawners();

        // 11. Update lifetimes
        self.update_lifetimes();

        // 12. Clean up dead entities
        self.cleanup_dead();

        // 13. Check win conditions
        self.check_win_conditions();
    }

    /// Run multiple ticks
    pub fn step_n(&mut self, n: i32) {
        for _ in 0..n {
            self.step();
            if self.game_over {
                break;
            }
        }
    }

    fn update_elixir(&mut self) {
        let mult = if self.tick >= OVERTIME_START_TICKS {
            if self.tick >= OVERTIME_END_TICKS { TRIPLE_ELIXIR_MULT } else { DOUBLE_ELIXIR_MULT }
        } else if self.tick >= DOUBLE_ELIXIR_TICKS {
            DOUBLE_ELIXIR_MULT
        } else {
            1.0
        };

        for p in &mut self.players {
            p.elixir = (p.elixir + ELIXIR_PER_TICK * mult).min(MAX_ELIXIR);
        }
    }

    fn update_deploy_timers(&mut self) {
        for troop in &mut self.troops {
            if troop.state == EntityState::Deploying {
                troop.deploy_timer -= MS_PER_TICK;
                if troop.deploy_timer <= 0 {
                    troop.state = EntityState::Idle;
                }
            }
        }
        for building in &mut self.buildings {
            if building.state == EntityState::Deploying {
                building.lifetime_timer -= MS_PER_TICK;
                // Buildings become active after a brief delay
                building.state = EntityState::Idle;
            }
        }
    }

    fn update_status_effects(&mut self) {
        for troop in &mut self.troops {
            troop.statuses.retain_mut(|s| {
                s.remaining_ms -= MS_PER_TICK;
                // Apply poison tick damage
                if let StatusKind::Poison(dps) = s.kind {
                    troop.hp -= dps * MS_PER_TICK / 1000;
                }
                s.remaining_ms > 0
            });
        }
        for building in &mut self.buildings {
            building.statuses.retain_mut(|s| {
                s.remaining_ms -= MS_PER_TICK;
                if let StatusKind::Poison(dps) = s.kind {
                    building.hp -= dps * MS_PER_TICK / 1000;
                }
                s.remaining_ms > 0
            });
        }
    }

    fn process_deaths(&mut self) {
        let mut spawns: Vec<(u8, String, i32, Pos)> = Vec::new();

        for troop in &mut self.troops {
            if troop.hp <= 0 && troop.state != EntityState::Dead {
                troop.state = EntityState::Dead;

                // Death damage
                if troop.death_damage > 0 && troop.death_damage_radius > 0 {
                    let _radius_sq = (troop.death_damage_radius as i64) * (troop.death_damage_radius as i64);
                    // Damage all enemy entities in radius
                    // We'll collect targets to avoid borrow issues
                    // This is handled by recording and applying after
                }

                // Death spawns
                if !troop.death_spawn.is_empty() && troop.death_spawn_count > 0 {
                    spawns.push((
                        troop.player_id,
                        troop.death_spawn.clone(),
                        troop.death_spawn_count,
                        troop.pos,
                    ));
                }
            }
        }

        // Apply death damage
        let death_damage_sources: Vec<(u8, Pos, i32, i32)> = self.troops.iter()
            .filter(|t| t.state == EntityState::Dead && t.death_damage > 0 && t.death_damage_radius > 0)
            .map(|t| (t.player_id, t.pos, t.death_damage, t.death_damage_radius))
            .collect();

        for (player_id, pos, damage, radius) in death_damage_sources {
            let radius_sq = (radius as i64) * (radius as i64);
            let enemy_id = 1 - player_id;

            for troop in &mut self.troops {
                if troop.player_id == enemy_id && troop.is_alive() {
                    if troop.pos.dist_sq(pos) <= radius_sq {
                        troop.hp -= damage;
                    }
                }
            }
            for building in &mut self.buildings {
                if building.player_id == enemy_id && building.is_alive() {
                    if building.pos.dist_sq(pos) <= radius_sq {
                        building.hp -= damage;
                    }
                }
            }
        }

        // Process building deaths
        for building in &mut self.buildings {
            if building.hp <= 0 && building.state != EntityState::Dead {
                building.state = EntityState::Dead;
                // Check for crown scoring
                if building.is_tower {
                    let enemy_id = 1 - building.player_id;
                    match building.tower_type {
                        TowerType::King => {
                            self.players[enemy_id as usize].crowns = 3; // instant win
                        }
                        TowerType::LeftPrincess | TowerType::RightPrincess => {
                            self.players[enemy_id as usize].crowns += 1;
                            // Activate king tower
                            self.players[building.player_id as usize].king_activated = true;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Spawn death spawns
        for (player_id, char_name, count, pos) in spawns {
            self.spawn_troops(player_id, &char_name, count, pos);
        }
    }

    fn update_spawners(&mut self) {
        let mut spawns: Vec<(u8, String, i32, Pos)> = Vec::new();

        // Building spawners
        for building in &mut self.buildings {
            if !building.is_alive() || building.summon_character.is_empty() || building.summon_interval_ms == 0 {
                continue;
            }
            building.summon_timer -= MS_PER_TICK;
            if building.summon_timer <= 0 {
                building.summon_timer = building.summon_interval_ms;
                spawns.push((
                    building.player_id,
                    building.summon_character.clone(),
                    building.summon_number,
                    Pos::new(building.pos.x, building.pos.y + if building.player_id == 0 { 1500 } else { -1500 }),
                ));
            }
        }

        // Troop spawners (Witch)
        for troop in &mut self.troops {
            if !troop.is_alive() || troop.summon_character.is_empty() || troop.summon_interval_ms == 0 {
                continue;
            }
            troop.summon_timer -= MS_PER_TICK;
            if troop.summon_timer <= 0 {
                troop.summon_timer = troop.summon_interval_ms;
                spawns.push((
                    troop.player_id,
                    troop.summon_character.clone(),
                    troop.summon_number,
                    troop.pos,
                ));
            }
        }

        for (player_id, char_name, count, pos) in spawns {
            self.spawn_troops(player_id, &char_name, count, pos);
        }
    }

    fn update_lifetimes(&mut self) {
        for troop in &mut self.troops {
            if troop.lifetime_timer > 0 && troop.is_alive() {
                troop.lifetime_timer -= MS_PER_TICK;
                if troop.lifetime_timer <= 0 {
                    troop.hp = 0;
                    troop.state = EntityState::Dead;
                }
            }
        }
        for building in &mut self.buildings {
            if building.lifetime_timer > 0 && building.is_alive() && !building.is_tower {
                building.lifetime_timer -= MS_PER_TICK;
                if building.lifetime_timer <= 0 {
                    building.hp = 0;
                    building.state = EntityState::Dead;
                }
            }
        }
    }

    fn cleanup_dead(&mut self) {
        self.troops.retain(|t| t.state != EntityState::Dead);
        // Keep dead towers for crown tracking, remove dead non-tower buildings
        self.buildings.retain(|b| b.is_tower || b.state != EntityState::Dead);
        self.projectiles.retain(|p| p.damage > 0 || p.speed > 0); // remove spent projectiles
    }

    fn check_win_conditions(&mut self) {
        let c0 = self.players[0].crowns;
        let c1 = self.players[1].crowns;

        // 3-crown win
        if c0 >= 3 {
            self.game_over = true;
            self.winner = Some(0);
            return;
        }
        if c1 >= 3 {
            self.game_over = true;
            self.winner = Some(1);
            return;
        }

        // Overtime
        if self.tick >= OVERTIME_START_TICKS && !self.overtime {
            self.overtime = true;
            // If tied at end of regular time, go to overtime
        }

        // End of overtime — most crowns wins
        if self.tick >= OVERTIME_END_TICKS {
            if c0 != c1 {
                self.game_over = true;
                self.winner = if c0 > c1 { Some(0) } else { Some(1) };
                return;
            }
            // If still tied, sudden death — next crown wins
        }

        // Sudden death: any crown after overtime ends wins
        if self.tick > OVERTIME_END_TICKS && c0 != c1 {
            self.game_over = true;
            self.winner = if c0 > c1 { Some(0) } else { Some(1) };
            return;
        }

        // Tiebreaker at max time — lowest HP tower percentage
        if self.tick >= MAX_TICKS {
            self.game_over = true;
            if c0 != c1 {
                self.winner = if c0 > c1 { Some(0) } else { Some(1) };
            } else {
                // Compare remaining tower HP percentages
                let hp0 = self.total_tower_hp_pct(0);
                let hp1 = self.total_tower_hp_pct(1);
                if (hp0 - hp1).abs() < 0.001 {
                    self.winner = None; // draw
                } else {
                    self.winner = if hp0 > hp1 { Some(0) } else { Some(1) };
                }
            }
        }
    }

    fn total_tower_hp_pct(&self, player_id: u8) -> f32 {
        let mut total_hp = 0i32;
        let mut total_max = 0i32;
        for b in &self.buildings {
            if b.player_id == player_id && b.is_tower {
                total_hp += b.hp.max(0);
                total_max += b.max_hp;
            }
        }
        if total_max == 0 { return 0.0; }
        total_hp as f32 / total_max as f32
    }

    /// Get current game time in seconds
    pub fn time_seconds(&self) -> f32 {
        self.tick as f32 * MS_PER_TICK as f32 / 1000.0
    }

    /// Get state observation for RL (flat vector)
    pub fn get_observation(&self, player_id: u8) -> Vec<f32> {
        let mut obs = Vec::with_capacity(2048);

        // Player state (20 features)
        let p = &self.players[player_id as usize];
        let e = &self.players[1 - player_id as usize];
        obs.push(p.elixir / MAX_ELIXIR);
        obs.push(e.elixir / MAX_ELIXIR); // opponent elixir (hidden in real game)
        obs.push(p.crowns as f32 / 3.0);
        obs.push(e.crowns as f32 / 3.0);
        obs.push(self.time_seconds() / 360.0);
        obs.push(if self.overtime { 1.0 } else { 0.0 });

        // Tower HPs (6 towers × 2 values)
        for b in &self.buildings {
            if b.is_tower {
                obs.push(b.hp as f32 / b.max_hp as f32);
            }
        }

        // Pad to fixed size
        while obs.len() < 20 {
            obs.push(0.0);
        }

        // Entity features (up to 64 entities × 10 features each)
        let mut entity_count = 0;
        for troop in &self.troops {
            if entity_count >= 64 { break; }
            if !troop.is_alive() { continue; }
            obs.push(troop.pos.x as f32 / ARENA_WIDTH as f32);
            obs.push(troop.pos.y as f32 / ARENA_HEIGHT as f32);
            obs.push(troop.hp as f32 / troop.max_hp as f32);
            obs.push(if troop.player_id == player_id { 1.0 } else { 0.0 });
            obs.push(troop.damage as f32 / 1000.0);
            obs.push(troop.range as f32 / 10000.0);
            obs.push(troop.speed as f32 / 500.0);
            obs.push(if troop.is_air() { 1.0 } else { 0.0 });
            obs.push(if troop.is_stunned() { 1.0 } else { 0.0 });
            obs.push(troop.area_damage_radius as f32 / 5000.0);
            entity_count += 1;
        }

        // Pad entities to 64 × 10 = 640
        while obs.len() < 20 + 640 {
            obs.push(0.0);
        }

        obs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_decks() -> (Vec<String>, Vec<String>) {
        let deck0 = vec![
            "Knight", "Archer", "Giant", "Musketeer",
            "Fireball", "Valkyrie", "Prince", "Witch",
        ].into_iter().map(String::from).collect();
        let deck1 = vec![
            "Knight", "Archer", "Giant", "Musketeer",
            "Fireball", "Valkyrie", "Prince", "Witch",
        ].into_iter().map(String::from).collect();
        (deck0, deck1)
    }

    #[test]
    fn test_create_battle() {
        let data = GameData::load(Path::new("gamedata")).unwrap();
        let (d0, d1) = test_decks();
        let engine = BattleEngine::new(data, d0, d1);

        assert_eq!(engine.tick, 0);
        assert_eq!(engine.buildings.len(), 6); // 3 towers each
        assert!(!engine.game_over);
        assert_eq!(engine.players[0].elixir, 5.0);
    }

    #[test]
    fn test_deploy_troop() {
        let data = GameData::load(Path::new("gamedata")).unwrap();
        let (d0, d1) = test_decks();
        let mut engine = BattleEngine::new(data, d0, d1);

        let cmd = Command {
            player_id: 0,
            card_hand_index: 0, // Knight (3 elixir)
            x: 9000,
            y: 10000,
        };
        assert!(engine.execute_command(&cmd));
        assert_eq!(engine.troops.len(), 1);
        assert_eq!(engine.troops[0].character_name, "Knight");
        assert_eq!(engine.players[0].elixir, 2.0); // 5 - 3
    }

    #[test]
    fn test_step_advances_time() {
        let data = GameData::load(Path::new("gamedata")).unwrap();
        let (d0, d1) = test_decks();
        let mut engine = BattleEngine::new(data, d0, d1);

        engine.step();
        assert_eq!(engine.tick, 1);
        assert!(engine.players[0].elixir > 5.0); // elixir regenerated
    }

    #[test]
    fn test_observation() {
        let data = GameData::load(Path::new("gamedata")).unwrap();
        let (d0, d1) = test_decks();
        let engine = BattleEngine::new(data, d0, d1);
        let obs = engine.get_observation(0);
        assert_eq!(obs.len(), 660); // 20 + 640
    }
}
