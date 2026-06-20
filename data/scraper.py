"""Clash Royale API scraper — BFS through player network to collect battle data.

Collects battle logs from the official Supercell API using a breadth-first
search through the player network. Starts from top-ladder players and expands
through their opponents.

Usage::

    python -m data.scraper --token YOUR_API_TOKEN --output battles.jsonl --max-battles 100000

Data format (JSONL): Each line is a JSON object with fields:
    - battle_time: str (ISO timestamp)
    - type: str (e.g. "PvP", "challenge")
    - deck_p0: list[int] (8 card global IDs)
    - deck_p1: list[int]
    - crowns_p0: int
    - crowns_p1: int
    - trophies_p0: int
    - trophies_p1: int
    - winner: int (0 or 1)
    - p0_tag: str
    - p1_tag: str
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import deque
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

# Use the RoyaleAPI proxy to avoid needing a static IP
DEFAULT_BASE_URL = "https://proxy.royaleapi.dev/v1"
OFFICIAL_BASE_URL = "https://api.clashroyale.com/v1"

RATE_LIMIT_DELAY = 0.05  # seconds between requests (~20 req/s)


def _encode_tag(tag: str) -> str:
    """URL-encode a player tag (# → %23)."""
    return tag.replace("#", "%23")


async def fetch_top_players(
    session: aiohttp.ClientSession,
    base_url: str,
    headers: dict,
    limit: int = 200,
) -> list[str]:
    """Fetch top ladder player tags from the global leaderboard."""
    url = f"{base_url}/locations/global/rankings/players?limit={limit}"
    async with session.get(url, headers=headers) as resp:
        if resp.status != 200:
            logger.warning("Failed to fetch leaderboard: %d", resp.status)
            return []
        data = await resp.json()
        return [item["tag"] for item in data.get("items", [])]


async def fetch_battles(
    session: aiohttp.ClientSession,
    base_url: str,
    headers: dict,
    tag: str,
) -> list[dict]:
    """Fetch a player's battle log (up to 25 battles)."""
    url = f"{base_url}/players/{_encode_tag(tag)}/battlelog"
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(url, headers=headers, timeout=timeout) as resp:
            if resp.status == 200:
                return await resp.json()
            if resp.status == 429:
                logger.warning("Rate limited, backing off...")
                await asyncio.sleep(5.0)
            return []
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.debug("Error fetching %s: %s", tag, e)
        return []


def _extract_battle_record(battle: dict) -> dict | None:
    """Extract a normalized battle record from the API response."""
    try:
        team = battle.get("team", [{}])
        opponent = battle.get("opponent", [{}])
        if not team or not opponent:
            return None

        p0 = team[0]
        p1 = opponent[0]

        deck_p0 = [card["id"] for card in p0.get("cards", [])]
        deck_p1 = [card["id"] for card in p1.get("cards", [])]

        if len(deck_p0) != 8 or len(deck_p1) != 8:
            return None

        crowns_p0 = p0.get("crownsEarned", 0)
        crowns_p1 = p1.get("crownsEarned", 0)

        if crowns_p0 > crowns_p1:
            winner = 0
        elif crowns_p1 > crowns_p0:
            winner = 1
        else:
            winner = -1  # draw

        return {
            "battle_time": battle.get("battleTime", ""),
            "type": battle.get("type", ""),
            "deck_p0": deck_p0,
            "deck_p1": deck_p1,
            "crowns_p0": crowns_p0,
            "crowns_p1": crowns_p1,
            "trophies_p0": p0.get("startingTrophies", 0),
            "trophies_p1": p1.get("startingTrophies", 0),
            "winner": winner,
            "p0_tag": p0.get("tag", ""),
            "p1_tag": p1.get("tag", ""),
        }
    except (KeyError, IndexError, TypeError):
        return None


async def bfs_scrape(
    token: str,
    output_path: str,
    max_battles: int = 100_000,
    base_url: str = DEFAULT_BASE_URL,
    seed_tags: list[str] | None = None,
) -> int:
    """BFS scrape through the player network.

    Returns the number of battles collected.
    """
    headers = {"Authorization": f"Bearer {token}"}
    visited: set[str] = set()
    queue: deque[str] = deque()
    battle_count = 0
    out_file = Path(output_path)

    async with aiohttp.ClientSession() as session:
        # Seed with top players
        if seed_tags:
            queue.extend(seed_tags)
        else:
            logger.info("Fetching top players from leaderboard...")
            top = await fetch_top_players(session, base_url, headers)
            queue.extend(top)
            logger.info("Seeded with %d top players", len(top))

        with open(out_file, "a") as f:
            while queue and battle_count < max_battles:
                tag = queue.popleft()
                if tag in visited:
                    continue
                visited.add(tag)

                battles = await fetch_battles(session, base_url, headers, tag)
                await asyncio.sleep(RATE_LIMIT_DELAY)

                for battle in battles:
                    record = _extract_battle_record(battle)
                    if record is None:
                        continue

                    f.write(json.dumps(record) + "\n")
                    battle_count += 1

                    # Add opponent to queue
                    opp_tag = record["p1_tag"]
                    if opp_tag and opp_tag not in visited:
                        queue.append(opp_tag)

                if battle_count % 1000 == 0:
                    logger.info(
                        "Progress: %d battles, %d players visited, %d in queue",
                        battle_count, len(visited), len(queue),
                    )

    logger.info("Scraping complete: %d battles from %d players", battle_count, len(visited))
    return battle_count


def load_battles(path: str) -> list[dict]:
    """Load battles from a JSONL file."""
    battles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                battles.append(json.loads(line))
    return battles


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape CR battles from the official API")
    parser.add_argument("--token", required=True, help="Supercell API token")
    parser.add_argument("--output", default="data/battles.jsonl", help="Output JSONL file")
    parser.add_argument("--max-battles", type=int, default=100_000, help="Max battles to collect")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(bfs_scrape(args.token, args.output, args.max_battles, args.base_url))


if __name__ == "__main__":
    main()
