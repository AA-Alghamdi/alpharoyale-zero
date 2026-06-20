use std::path::Path;
use std::time::Instant;

use cr_engine_native::data::GameData;
use cr_engine_native::engine::{BattleEngine, Command};

fn main() {
    let data = GameData::load(Path::new("gamedata")).unwrap();

    let deck0: Vec<String> = vec![
        "Knight", "Archer", "Giant", "Musketeer",
        "Fireball", "Valkyrie", "Prince", "Witch",
    ].into_iter().map(String::from).collect();

    let deck1: Vec<String> = vec![
        "Knight", "Archer", "Goblins", "MiniPekka",
        "Arrows", "Bomber", "Musketeer", "Giant",
    ].into_iter().map(String::from).collect();

    // Benchmark 1: Empty tick speed
    {
        let mut engine = BattleEngine::new(data.clone(), deck0.clone(), deck1.clone());
        let n = 1_000_000;
        let start = Instant::now();
        engine.step_n(n);
        let elapsed = start.elapsed();
        let tps = n as f64 / elapsed.as_secs_f64();
        println!("Empty ticks:     {:.0} ticks/sec ({} ticks in {:.2}ms)", tps, n, elapsed.as_secs_f64() * 1000.0);
    }

    // Benchmark 2: Battle with troops deployed periodically
    {
        let mut engine = BattleEngine::new(data.clone(), deck0.clone(), deck1.clone());
        engine.execute_command(&Command { player_id: 0, card_hand_index: 0, x: 9000, y: 10000 });
        engine.execute_command(&Command { player_id: 0, card_hand_index: 1, x: 4000, y: 8000 });
        engine.execute_command(&Command { player_id: 1, card_hand_index: 0, x: 9000, y: 22000 });
        engine.execute_command(&Command { player_id: 1, card_hand_index: 1, x: 14000, y: 24000 });

        let n = 100_000;
        let start = Instant::now();
        for i in 0..n {
            engine.step();
            if i % 1000 == 0 && !engine.game_over {
                let hand = (i / 1000 % 4) as usize;
                let _ = engine.execute_command(&Command { player_id: 0, card_hand_index: hand, x: 9000, y: 10000 });
                let _ = engine.execute_command(&Command { player_id: 1, card_hand_index: hand, x: 9000, y: 22000 });
            }
            if engine.game_over { break; }
        }
        let elapsed = start.elapsed();
        let actual_ticks = engine.tick;
        let tps = actual_ticks as f64 / elapsed.as_secs_f64();
        println!("Battle ticks:    {:.0} ticks/sec ({} ticks in {:.2}ms, {} troops alive)",
            tps, actual_ticks, elapsed.as_secs_f64() * 1000.0, engine.troops.len());
    }

    // Benchmark 3: Full game simulation (realistic play — deploy every 3-5 sec)
    {
        let n_games = 1000;
        let start = Instant::now();
        let mut total_ticks = 0i64;
        let mut wins = [0i32; 3]; // [p0, p1, draw]
        for _ in 0..n_games {
            let mut engine = BattleEngine::new(data.clone(), deck0.clone(), deck1.clone());
            let mut deploy_cooldown = [0i32; 2];

            loop {
                engine.step();
                for pid in 0..2u8 {
                    deploy_cooldown[pid as usize] -= 1;
                    if deploy_cooldown[pid as usize] <= 0 && engine.players[pid as usize].elixir >= 4.0 {
                        let y = if pid == 0 { 10000 } else { 22000 };
                        let hand = (engine.tick as usize / 100 + pid as usize) % 4;
                        let _ = engine.execute_command(&Command { player_id: pid, card_hand_index: hand, x: 9000, y });
                        deploy_cooldown[pid as usize] = 60; // ~3 sec cooldown
                    }
                }
                if engine.game_over { break; }
                if engine.tick >= 7200 { break; }
            }
            total_ticks += engine.tick as i64;
            match engine.winner {
                Some(0) => wins[0] += 1,
                Some(1) => wins[1] += 1,
                _ => wins[2] += 1,
            }
        }
        let elapsed = start.elapsed();
        let total_tps = total_ticks as f64 / elapsed.as_secs_f64();
        let games_per_sec = n_games as f64 / elapsed.as_secs_f64();
        println!("Full games:      {:.0} ticks/sec, {:.1} games/sec ({} games in {:.2}s)",
            total_tps, games_per_sec, n_games, elapsed.as_secs_f64());
        println!("                 avg {:.0} ticks/game, P0 wins: {}, P1 wins: {}, draws: {}",
            total_ticks as f64 / n_games as f64, wins[0], wins[1], wins[2]);
    }

    println!("\nComparison:");
    println!("  Our Python crsim:    761 ticks/sec");
    println!("  clash-simulator:     732,000 ticks/sec (Python)");
    println!("  cr_engine (Rust):    see above");
}
