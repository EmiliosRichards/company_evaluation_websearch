## Manuav Company Evaluation (single LLM call + web search)

This repo is a small pipeline that:
- takes a **company website URL**
- runs **one** OpenAI call with **web search enabled**
- returns a **Manuav Fit score (0–10)** + recommendation + evidence (JSON)

### Setup

- **Install**:

```bash
python -m pip install -r requirements.txt
```

- **Create a `.env`** (recommended):

```bash
copy .env.example .env
```

Then edit `.env` and set `OPENAI_API_KEY`.

Optional (model override):

```bash
OPENAI_MODEL="gpt-4.1-mini"
```

Optional (rubric override / versioning):

```bash
MANUAV_RUBRIC_FILE="rubrics/manuav_rubric_v4_en.md"
```

### Run

```bash
python -m scripts.evaluate https://company.com
```

### Irene sample (9 rows: 3 low / 3 mid / 3 high)

Create the sample:

```bash
python -m scripts.make_irene_sample
```

Run the evaluator on the sample (writes JSONL + prints MAE):

```bash
python -m scripts.run_irene_sample
```

This also writes a **timestamped CSV** to `outputs/`. Add a suffix with `-s`:

```bash
python -m scripts.run_irene_sample -s baseline
```

Files:
- `data/Websearch Irene - Manuav AI Search.csv`: Irene’s full manual research
- `data/irene_sample_9.csv`: sampled 9-row subset
- `outputs/<timestamp>[_suffix].jsonl`: model results for the sample (JSONL)
- `outputs/<timestamp>[_suffix].csv`: flattened results for the sample (CSV)

### Output

`scripts.evaluate` prints strict JSON including:
- `manuav_fit_score` (0–10)
- `confidence` (low/medium/high)
- `reasoning` (why this score, per rubric)
- `sources_visited` (the pages relied on)


