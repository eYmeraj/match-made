# Match Made

Find the rows in two files that describe the same entity (company, person, product) and
download the merged result. Built for files of a million rows or more.

- **Files**: CSV, TSV or Excel (.xlsx). Delimiter, encoding and sheet are detected and adjustable.
- **Comparisons**: pick column pairs, how to clean them (case, accents, legal suffixes,
  abbreviations, stopwords, replacements, number and date formats) and how to score them.
  Each scores 0-100; the match score is their weighted average. Auto-suggest is optional.
- **Blocking**: key rules (same first word, year, sounds-alike...) and similar-text rules
  (character n-gram TF-IDF) choose which pairs get scored.
- **Results**: one-to-one, best match per A row, or all pairs above the threshold. Tune the
  threshold without re-running, then download the merged file and the unmatched rows.

## Run it

```sh
docker compose up -d --build
```

Open http://localhost:8080 (set `MATCHMADE_PORT` to change the port). Uploaded files and
results live in the `data` volume.

## Develop

Backend (Python 3.12, [uv](https://docs.astral.sh/uv/)):

```sh
cd backend
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

Frontend (Node 24):

```sh
cd frontend
npm install
API_URL=http://localhost:8080 npm run dev   # proxies /api to the compose stack
npm run lint && npm run typecheck && npm test
npm run e2e                                 # needs the compose stack running
npm run gen:api                             # after changing backend API models
```

`frontend/src/api/schema.ts` is generated from the backend's OpenAPI schema; don't edit it.

## How it works

`web` (nginx) serves the React app and proxies `/api` to `api` (FastAPI). Runs are queued in
Redis and executed by `worker` (RQ): clean -> block -> score -> persist candidate pairs.
Pairs scoring below `threshold - 25` are discarded, so the results page can lower the
threshold by up to 25 points without a re-run.

To measure speed and accuracy on synthetic data with a known ground truth:

```sh
cd backend
uv run python scripts/generate_synthetic.py --rows 1000000 --out ../.scratch/syn1m
uv run python scripts/benchmark.py ../.scratch/syn1m
```

On an M-series laptop, 1M x 1M rows match in about 4 minutes.
