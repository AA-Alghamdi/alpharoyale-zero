//! CSV game data parsing for authentic card stats.
//!
//! Parses characters.csv, spells_characters.csv, spells_buildings.csv,
//! spells_other.csv from the Supercell APK game data.

use std::collections::HashMap;
use std::path::Path;

// ---- Level scaling ----
// CR uses these exact multipliers per level. Tournament Standard = Level 11.
// Source: Supercell APK data, validated against wiki.
const LEVEL_MULTIPLIERS: [f64; 14] = [
    1.0,    // Level 1
    1.1,    // Level 2
    1.21,   // Level 3
    1.331,  // Level 4
    1.4641, // Level 5
    1.6105, // Level 6
    1.7716, // Level 7
    1.9487, // Level 8
    2.1436, // Level 9
    2.3579, // Level 10
    2.5937, // Level 11
    2.8531, // Level 12
    3.1384, // Level 13
    3.4523, // Level 14
];

/// Starting level per rarity (Level 1 in the CSV = this rarity's base level)
fn rarity_base_level(rarity: &str) -> i32 {
    match rarity {
        "Common" => 1,
        "Rare" => 3,
        "Epic" => 6,
        "Legendary" => 9,
        "Champion" | "Hero" => 11,
        _ => 1,
    }
}

/// Scale a stat from Level 1 to the target level, accounting for rarity.
/// Common cards at Level 1 in CSV are truly Level 1.
/// Rare cards at Level 1 in CSV correspond to gameplay Level 3, etc.
pub fn scale_stat(base_value: i32, rarity: &str, target_level: i32) -> i32 {
    if base_value == 0 {
        return 0;
    }
    let base_level = rarity_base_level(rarity);
    // How many upgrades from base to target
    let upgrades = (target_level - base_level).max(0) as usize;
    if upgrades == 0 {
        return base_value;
    }
    // Each upgrade = ×1.1
    let mult = 1.1_f64.powi(upgrades as i32);
    (base_value as f64 * mult).round() as i32
}

/// Character stats from characters.csv (the actual unit — Knight, Archer, etc.)
#[derive(Debug, Clone)]
pub struct CharacterData {
    pub name: String,
    pub rarity: String,
    pub sight_range: i32,     // millitiles
    pub deploy_time: i32,     // ms
    pub charge_range: i32,    // millitiles, 0 = no charge
    pub speed: i32,           // units/tick
    pub hitpoints: i32,
    pub hit_speed: i32,       // ms between attacks
    pub load_time: i32,       // ms before first hit connects
    pub damage: i32,
    pub crown_tower_damage_pct: i32,
    pub range: i32,           // millitiles
    pub attacks_ground: bool,
    pub attacks_air: bool,
    pub death_damage_radius: i32,
    pub death_damage: i32,
    pub death_pushback: i32,
    pub attack_pushback: i32,
    pub lifetime: i32,        // ms, 0 = permanent
    pub area_damage_radius: i32,
    pub target_only_buildings: bool,
    pub collision_radius: i32,
    pub mass: i32,
    pub flying_height: i32,   // 0 = ground unit
    pub multiple_projectiles: i32,
    pub projectile: String,
    pub special_attack_interval: i32,
    pub summon_character: String, // for spawners like Witch
    pub summon_number: i32,
}

impl CharacterData {
    /// Scale HP and damage stats to a target level.
    pub fn scale_to_level(&mut self, target_level: i32) {
        self.hitpoints = scale_stat(self.hitpoints, &self.rarity, target_level);
        self.damage = scale_stat(self.damage, &self.rarity, target_level);
        self.death_damage = scale_stat(self.death_damage, &self.rarity, target_level);
    }
}

impl Default for CharacterData {
    fn default() -> Self {
        Self {
            name: String::new(),
            rarity: String::new(),
            sight_range: 5500,
            deploy_time: 1000,
            charge_range: 0,
            speed: 60,
            hitpoints: 100,
            hit_speed: 1000,
            load_time: 0,
            damage: 50,
            crown_tower_damage_pct: 0,
            range: 1000,
            attacks_ground: true,
            attacks_air: false,
            death_damage_radius: 0,
            death_damage: 0,
            death_pushback: 0,
            attack_pushback: 0,
            lifetime: 0,
            area_damage_radius: 0,
            target_only_buildings: false,
            collision_radius: 100,
            mass: 500,
            flying_height: 0,
            multiple_projectiles: 0,
            projectile: String::new(),
            special_attack_interval: 0,
            summon_character: String::new(),
            summon_number: 0,
        }
    }
}

/// Spell card data from spells_characters.csv (the playable card)
#[derive(Debug, Clone)]
pub struct SpellCharacterData {
    pub name: String,
    pub rarity: String,
    pub mana_cost: i32,
    pub summon_character: String,
    pub summon_number: i32,
    pub summon_character_second: String,
    pub summon_character_second_count: i32,
    pub radius: i32,
    pub can_place_on_buildings: bool,
}

/// Spell data from spells_other.csv (Fireball, Arrows, etc.)
#[derive(Debug, Clone)]
pub struct SpellOtherData {
    pub name: String,
    pub mana_cost: i32,
    pub radius: i32,
    pub instant_damage: i32,
    pub duration_seconds: i32,
    pub pushback: i32,
    pub effect: String,
}

/// Building data from spells_buildings.csv
#[derive(Debug, Clone)]
pub struct SpellBuildingData {
    pub name: String,
    pub mana_cost: i32,
    pub summon_character: String,
}

/// Projectile data from projectiles.csv
#[derive(Debug, Clone)]
pub struct ProjectileData {
    pub name: String,
    pub speed: i32,
    pub damage: i32,
    pub radius: i32,
    pub pushback: i32,
    pub homing: bool,
}

/// All game data loaded from CSVs
#[derive(Debug, Clone)]
pub struct GameData {
    pub characters: HashMap<String, CharacterData>,
    pub spell_characters: HashMap<String, SpellCharacterData>,
    pub spell_others: HashMap<String, SpellOtherData>,
    pub spell_buildings: HashMap<String, SpellBuildingData>,
    pub projectiles: HashMap<String, ProjectileData>,
    pub level: i32,
}

fn parse_int(s: &str) -> i32 {
    s.trim().parse::<i32>().unwrap_or(0)
}

fn parse_bool(s: &str) -> bool {
    matches!(s.trim().to_lowercase().as_str(), "true" | "1" | "yes")
}

impl GameData {
    /// Load game data from CSVs with stats at Level 1 (raw from APK).
    pub fn load(data_dir: &Path) -> Result<Self, String> {
        Self::load_at_level(data_dir, 1)
    }

    /// Load game data and scale all stats to `level` (Tournament Standard = 11).
    pub fn load_at_level(data_dir: &Path, level: i32) -> Result<Self, String> {
        let mut characters = Self::load_characters(data_dir)?;
        let spell_characters = Self::load_spell_characters(data_dir)?;
        let spell_others = Self::load_spell_others(data_dir)?;
        let spell_buildings = Self::load_spell_buildings(data_dir)?;
        let projectiles = Self::load_projectiles(data_dir)?;

        // Scale character stats to target level
        if level > 1 {
            for char_data in characters.values_mut() {
                char_data.scale_to_level(level);
            }
        }

        Ok(Self {
            characters,
            spell_characters,
            spell_others,
            spell_buildings,
            projectiles,
            level,
        })
    }

    fn load_characters(data_dir: &Path) -> Result<HashMap<String, CharacterData>, String> {
        let path = data_dir.join("characters.csv");
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(&path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;

        let headers = rdr.headers().map_err(|e| e.to_string())?.clone();

        // Skip the type-definition row (second row in SC CSVs)
        let mut records = rdr.records();
        let _ = records.next(); // skip type row

        let mut map = HashMap::new();

        for result in records {
            let record = result.map_err(|e| e.to_string())?;
            let get = |field: &str| -> String {
                headers.iter().position(|h| h == field)
                    .and_then(|i| record.get(i))
                    .unwrap_or("")
                    .to_string()
            };

            let name = get("Name");
            if name.is_empty() || name.starts_with("NOTINUSE") {
                continue;
            }

            let data = CharacterData {
                name: name.clone(),
                rarity: get("Rarity"),
                sight_range: parse_int(&get("SightRange")),
                deploy_time: parse_int(&get("DeployTime")),
                charge_range: parse_int(&get("ChargeRange")),
                speed: parse_int(&get("Speed")),
                hitpoints: parse_int(&get("Hitpoints")),
                hit_speed: parse_int(&get("HitSpeed")),
                load_time: parse_int(&get("LoadTime")),
                damage: parse_int(&get("Damage")),
                crown_tower_damage_pct: parse_int(&get("CrownTowerDamagePercent")),
                range: parse_int(&get("Range")),
                attacks_ground: parse_bool(&get("AttacksGround")),
                attacks_air: parse_bool(&get("AttacksAir")),
                death_damage_radius: parse_int(&get("DeathDamageRadius")),
                death_damage: parse_int(&get("DeathDamage")),
                death_pushback: parse_int(&get("DeathPushBack")),
                attack_pushback: parse_int(&get("AttackPushBack")),
                lifetime: parse_int(&get("LifeTime")),
                area_damage_radius: parse_int(&get("AreaDamageRadius")),
                target_only_buildings: parse_bool(&get("TargetOnlyBuildings")),
                collision_radius: parse_int(&get("CollisionRadius")),
                mass: parse_int(&get("Mass")),
                flying_height: parse_int(&get("FlyingHeight")),
                multiple_projectiles: parse_int(&get("MultipleProjectiles")),
                projectile: get("Projectile"),
                special_attack_interval: parse_int(&get("SpecialAttackInterval")),
                summon_character: get("AttachedCharacter"),
                summon_number: 0,
            };

            map.insert(name, data);
        }

        Ok(map)
    }

    fn load_spell_characters(data_dir: &Path) -> Result<HashMap<String, SpellCharacterData>, String> {
        let path = data_dir.join("spells_characters.csv");
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(&path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;

        let headers = rdr.headers().map_err(|e| e.to_string())?.clone();
        let mut records = rdr.records();
        let _ = records.next(); // skip type row

        let mut map = HashMap::new();

        for result in records {
            let record = result.map_err(|e| e.to_string())?;
            let get = |field: &str| -> String {
                headers.iter().position(|h| h == field)
                    .and_then(|i| record.get(i))
                    .unwrap_or("")
                    .to_string()
            };

            let name = get("Name");
            if name.is_empty() || name.starts_with("NOTINUSE") {
                continue;
            }

            let data = SpellCharacterData {
                name: name.clone(),
                rarity: get("Rarity"),
                mana_cost: parse_int(&get("ManaCost")),
                summon_character: get("SummonCharacter"),
                summon_number: {
                    let n = parse_int(&get("SummonNumber"));
                    if n == 0 { 1 } else { n }
                },
                summon_character_second: get("SummonCharacterSecond"),
                summon_character_second_count: parse_int(&get("SummonCharacterSecondCount")),
                radius: parse_int(&get("Radius")),
                can_place_on_buildings: parse_bool(&get("CanPlaceOnBuildings")),
            };

            map.insert(name, data);
        }

        Ok(map)
    }

    fn load_spell_others(data_dir: &Path) -> Result<HashMap<String, SpellOtherData>, String> {
        let path = data_dir.join("spells_other.csv");
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(&path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;

        let headers = rdr.headers().map_err(|e| e.to_string())?.clone();
        let mut records = rdr.records();
        let _ = records.next();

        let mut map = HashMap::new();

        for result in records {
            let record = result.map_err(|e| e.to_string())?;
            let get = |field: &str| -> String {
                headers.iter().position(|h| h == field)
                    .and_then(|i| record.get(i))
                    .unwrap_or("")
                    .to_string()
            };

            let name = get("Name");
            if name.is_empty() || name.starts_with("NotInUse") {
                continue;
            }

            let data = SpellOtherData {
                name: name.clone(),
                mana_cost: parse_int(&get("ManaCost")),
                radius: parse_int(&get("Radius")),
                instant_damage: parse_int(&get("InstantDamage")),
                duration_seconds: parse_int(&get("DurationSeconds")),
                pushback: parse_int(&get("Pushback")),
                effect: get("Effect"),
            };

            map.insert(name, data);
        }

        Ok(map)
    }

    fn load_spell_buildings(data_dir: &Path) -> Result<HashMap<String, SpellBuildingData>, String> {
        let path = data_dir.join("spells_buildings.csv");
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(&path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;

        let headers = rdr.headers().map_err(|e| e.to_string())?.clone();
        let mut records = rdr.records();
        let _ = records.next();

        let mut map = HashMap::new();

        for result in records {
            let record = result.map_err(|e| e.to_string())?;
            let get = |field: &str| -> String {
                headers.iter().position(|h| h == field)
                    .and_then(|i| record.get(i))
                    .unwrap_or("")
                    .to_string()
            };

            let name = get("Name");
            if name.is_empty() || name.starts_with("NOT_IN_USE") {
                continue;
            }

            let data = SpellBuildingData {
                name: name.clone(),
                mana_cost: parse_int(&get("ManaCost")),
                summon_character: get("SummonCharacter"),
            };

            map.insert(name, data);
        }

        Ok(map)
    }

    fn load_projectiles(data_dir: &Path) -> Result<HashMap<String, ProjectileData>, String> {
        let path = data_dir.join("projectiles.csv");
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(&path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;

        let headers = rdr.headers().map_err(|e| e.to_string())?.clone();
        let mut records = rdr.records();
        let _ = records.next();

        let mut map = HashMap::new();

        for result in records {
            let record = result.map_err(|e| e.to_string())?;
            let get = |field: &str| -> String {
                headers.iter().position(|h| h == field)
                    .and_then(|i| record.get(i))
                    .unwrap_or("")
                    .to_string()
            };

            let name = get("Name");
            if name.is_empty() {
                continue;
            }

            let data = ProjectileData {
                name: name.clone(),
                speed: parse_int(&get("Speed")),
                damage: parse_int(&get("Damage")),
                radius: parse_int(&get("Radius")),
                pushback: parse_int(&get("Pushback")),
                homing: parse_bool(&get("Homing")),
            };

            map.insert(name, data);
        }

        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_gamedata() {
        let data = GameData::load(Path::new("gamedata")).unwrap();
        assert!(data.characters.contains_key("Knight"));
        assert!(data.characters.contains_key("Archer"));
        assert!(data.characters.contains_key("Giant"));

        let knight = &data.characters["Knight"];
        assert_eq!(knight.hitpoints, 660);
        assert_eq!(knight.damage, 75);
        assert_eq!(knight.speed, 60);
        assert!(knight.attacks_ground);
        assert!(!knight.attacks_air);

        assert!(data.spell_characters.contains_key("Knight"));
        assert_eq!(data.spell_characters["Knight"].mana_cost, 3);

        assert!(data.spell_others.contains_key("Fireball"));
        assert_eq!(data.spell_others["Fireball"].mana_cost, 4);

        println!("Loaded {} characters, {} spell cards, {} spells, {} buildings, {} projectiles",
            data.characters.len(),
            data.spell_characters.len(),
            data.spell_others.len(),
            data.spell_buildings.len(),
            data.projectiles.len(),
        );
    }

    #[test]
    fn test_load_gamedata_v2() {
        let path = Path::new("gamedata_v2");
        if !path.exists() {
            eprintln!("gamedata_v2 not found, skipping");
            return;
        }
        let data = GameData::load(path).unwrap();

        // Should have significantly more characters than v1
        assert!(data.characters.len() > 80, "Expected 80+ characters, got {}", data.characters.len());
        assert!(data.spell_characters.len() > 80, "Expected 80+ spell chars, got {}", data.spell_characters.len());

        // New cards should be present
        assert!(data.characters.contains_key("ElectroGiant") || data.characters.contains_key("ElectroDragon"),
            "Expected new cards in v2 data");

        println!("V2: {} characters, {} spell cards, {} spells, {} buildings, {} projectiles",
            data.characters.len(),
            data.spell_characters.len(),
            data.spell_others.len(),
            data.spell_buildings.len(),
            data.projectiles.len(),
        );
    }

    #[test]
    fn test_level_scaling() {
        // Knight is Common: Level 1 HP=660, Level 11 should be ~1712 (660×1.1^10)
        let hp = scale_stat(660, "Common", 11);
        assert!((hp - 1712).abs() <= 10, "Knight L11 HP {} not near 1712", hp);

        // Giant is Rare: Level 1 HP=1900, rarity base=3, so L11 = 8 upgrades = 1900×1.1^8
        let giant_hp = scale_stat(1900, "Rare", 11);
        assert!((giant_hp - 4073).abs() <= 20, "Giant L11 HP {} not near 4073", giant_hp);

        // Level 1 should be unchanged for Common
        assert_eq!(scale_stat(660, "Common", 1), 660);

        // Zero stays zero
        assert_eq!(scale_stat(0, "Common", 11), 0);

        // Load at level 11 and verify
        let data = GameData::load_at_level(Path::new("gamedata"), 11).unwrap();
        let knight = &data.characters["Knight"];
        assert!(knight.hitpoints > 1500, "Knight L11 HP {} should be >1500", knight.hitpoints);
        println!("Knight L11: HP={}, Damage={}", knight.hitpoints, knight.damage);
    }
}
