# Dataset Pipeline

The dataset pipeline builds structured Clash Royale match records from gameplay
videos and local recordings. Its output is a `GameRecord`: one validated JSON
object per match, with timeline states, detected actions, decks, tower health,
elixir, result, confidence, and provenance.

## Flow

```text
discover -> download -> sample frames -> annotate frames -> segment games -> write records
```

## Commands

Run the offline smoke test:

```bash
python tools/dataset/smoke_test.py
```

Print the schema:

```bash
python tools/dataset/run.py schema
```

Discover candidate videos:

```bash
python tools/dataset/run.py discover --per-query 50 --out data/queue.jsonl
```

Process a local video:

```bash
python tools/dataset/run.py process-local \
  --file match.mp4 \
  --video-id local_match_001 \
  --provider mock \
  --fps 1
```

Split bulk work into two phases:

```bash
python tools/dataset/batch_download.py --max-videos 50 --fps 1
python tools/dataset/batch_annotate.py --max-videos 25 --batch-size 5 --patient
```

## Access Notes

Bulk YouTube download typically requires authenticated browser cookies, a local
JavaScript runtime for challenge handling, and conservative request pacing. The
download layer supports cookie files, browser-profile cookies, proxies, retry
limits, and low-resolution formats to keep extraction cheap.

Generated queues, raw videos, extracted frames, progress files, and game dumps
belong under `data/` and are intentionally ignored by Git. Curated, small,
schema-valid examples live in `samples/game_records/`.

## Quality Controls

- All records validate against `crpipe.schema.GameRecord`.
- Every frame carries confidence and raw annotation payloads for audits.
- Failed frames produce low-confidence placeholders instead of aborting a video.
- Game segmentation uses gameplay gaps, clock resets, and crown changes.
- Batch annotation supports retry/backoff and provider rotation for long runs.

## Training Use

The records can seed:

- Imitation warm-start datasets.
- Deck/style priors.
- Action-distribution audits.
- Simulator-to-video fidelity checks.
- Live bridge decision-log comparison.
