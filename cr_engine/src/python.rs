//! Python bindings via PyO3

use pyo3::prelude::*;
use std::path::Path;

use crate::data::GameData;
use crate::engine::{BattleEngine, Command};

#[pyclass(name = "CREngine")]
struct PyCREngine {
    engine: BattleEngine,
}

#[pymethods]
impl PyCREngine {
    #[new]
    #[pyo3(signature = (data_dir, deck0, deck1, level=11))]
    fn new(data_dir: &str, deck0: Vec<String>, deck1: Vec<String>, level: i32) -> PyResult<Self> {
        let game_data = GameData::load_at_level(Path::new(data_dir), level)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(Self {
            engine: BattleEngine::new(game_data, deck0, deck1),
        })
    }

    /// Advance simulation by one tick
    fn step(&mut self) {
        self.engine.step();
    }

    /// Advance simulation by n ticks
    fn step_n(&mut self, n: i32) {
        self.engine.step_n(n);
    }

    /// Play a card
    #[pyo3(signature = (player_id, hand_index, x, y))]
    fn play_card(&mut self, player_id: u8, hand_index: usize, x: i32, y: i32) -> bool {
        self.engine.execute_command(&Command {
            player_id,
            card_hand_index: hand_index,
            x,
            y,
        })
    }

    /// Get current tick
    #[getter]
    fn tick(&self) -> i32 {
        self.engine.tick
    }

    /// Is the game over?
    #[getter]
    fn game_over(&self) -> bool {
        self.engine.game_over
    }

    /// Winner (0 or 1, or -1 for draw/ongoing)
    #[getter]
    fn winner(&self) -> i32 {
        self.engine.winner.map(|w| w as i32).unwrap_or(-1)
    }

    /// Get elixir for a player
    fn get_elixir(&self, player_id: u8) -> f32 {
        self.engine.players[player_id as usize].elixir
    }

    /// Get crowns for a player
    fn get_crowns(&self, player_id: u8) -> i32 {
        self.engine.players[player_id as usize].crowns
    }

    /// Get current hand (card names) for a player
    fn get_hand(&self, player_id: u8) -> Vec<String> {
        let p = &self.engine.players[player_id as usize];
        (0..4).map(|i| p.card_name(i).to_string()).collect()
    }

    /// Get next card name for a player
    fn get_next_card(&self, player_id: u8) -> String {
        let p = &self.engine.players[player_id as usize];
        p.deck[p.next_card].clone()
    }

    /// Get observation vector for RL
    fn get_observation(&self, player_id: u8) -> Vec<f32> {
        self.engine.get_observation(player_id)
    }

    /// Get time in seconds
    fn time_seconds(&self) -> f32 {
        self.engine.time_seconds()
    }

    /// Get troop count
    fn troop_count(&self) -> usize {
        self.engine.troops.len()
    }

    /// Get building count
    fn building_count(&self) -> usize {
        self.engine.buildings.len()
    }

    /// Get total troops alive per player
    fn troops_alive(&self, player_id: u8) -> usize {
        self.engine.troops.iter()
            .filter(|t| t.player_id == player_id && t.is_alive())
            .count()
    }

    /// Get tower HP [king, left_princess, right_princess] for a player
    fn get_tower_hp(&self, player_id: u8) -> Vec<i32> {
        let mut hps = vec![0i32; 3];
        for b in &self.engine.buildings {
            if b.player_id == player_id && b.is_tower {
                let idx = match b.tower_type {
                    crate::entity::TowerType::King => 0,
                    crate::entity::TowerType::LeftPrincess => 1,
                    crate::entity::TowerType::RightPrincess => 2,
                    _ => continue,
                };
                hps[idx] = b.hp.max(0);
            }
        }
        hps
    }

    /// Clone the engine state (for MCTS)
    fn clone_state(&self) -> PyCREngine {
        PyCREngine {
            engine: self.engine.clone(),
        }
    }

    /// Get all entity positions and info for visualization/debugging
    fn get_entities(&self) -> Vec<(u32, u8, i32, i32, i32, i32, String, bool)> {
        let mut entities = Vec::new();
        for t in &self.engine.troops {
            if t.is_alive() {
                entities.push((
                    t.id,
                    t.player_id,
                    t.pos.x,
                    t.pos.y,
                    t.hp,
                    t.max_hp,
                    t.character_name.clone(),
                    t.is_air(),
                ));
            }
        }
        entities
    }

    /// Run a benchmark: step N ticks and return time in ms
    fn benchmark(&mut self, n_ticks: i32) -> f64 {
        let start = std::time::Instant::now();
        for _ in 0..n_ticks {
            self.engine.step();
            if self.engine.game_over {
                break;
            }
        }
        start.elapsed().as_secs_f64() * 1000.0
    }
}

/// Python module
#[pymodule]
fn cr_engine_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCREngine>()?;
    Ok(())
}
