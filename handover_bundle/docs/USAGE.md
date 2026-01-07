## Usage Guide

This repo has three main OpenAI entrypoints:
- `python -m scripts.evaluate` (single URL)
- `python -m scripts.evaluate_list` (CSV/TXT list → JSONL + CSV)
- `python -m scripts.trace_web_search` (investigate web-search behavior; debug/audit)

### Quick presets (recommended)

- **Baseline bulk run (cheap + stable)**

```bash
# Guardrail: allow at most 2 web-search tool invocations within a single evaluation call
MANUAV_MAX_TOOL_CALLS=2

# Optional: Flex tier (cheaper tokens; web-search billing is separate)
MANUAV_SERVICE_TIER=flex
MANUAV_OPENAI_TIMEOUT_SECONDS=900
```

- **Sticky-company help (still usually 1 query)**

```bash
MANUAV_MAX_TOOL_CALLS=2
MANUAV_SECOND_QUERY_ON_UNCERTAINTY=1
```

- **Gated fallback retry (only on low confidence)**

```bash
MANUAV_MAX_TOOL_CALLS=2
MANUAV_RETRY_DISAMBIGUATION_ON_LOW_CONFIDENCE=1
MANUAV_RETRY_MAX_TOOL_CALLS=3
```

Tip: the retry is a **second model call** and can add extra web-search queries **only** on the rows that trigger it (`confidence=low`).

### `scripts.evaluate` (single company)

Minimal:

```bash
python -m scripts.evaluate https://example.com
```

Most useful flags/envs:
- **Model/rubric**
  - env `OPENAI_MODEL` / flag `--model`
  - env `MANUAV_RUBRIC_FILE` / flag `--rubric-file`
- **Tool budget**
  - env `MANUAV_MAX_TOOL_CALLS` / flag `--max-tool-calls`
  - env `MANUAV_SECOND_QUERY_ON_UNCERTAINTY` / flag `--second-query-on-uncertainty`
- **Gated retry (optional)**
  - env `MANUAV_RETRY_DISAMBIGUATION_ON_LOW_CONFIDENCE` / flag `--retry-disambiguation-on-low-confidence`
  - env `MANUAV_RETRY_MAX_TOOL_CALLS` / flag `--retry-max-tool-calls`
- **Flex**
  - env `MANUAV_SERVICE_TIER` / flag `--service-tier`
  - env `MANUAV_OPENAI_TIMEOUT_SECONDS` / flag `--timeout-seconds`
  - env `MANUAV_FLEX_MAX_RETRIES` / flag `--flex-max-retries`
  - env `MANUAV_FLEX_FALLBACK_TO_AUTO` / flag `--flex-fallback-to-auto`
- **Prompt caching (optional)**
  - env `MANUAV_PROMPT_CACHE` / flag `--prompt-cache`
  - env `MANUAV_PROMPT_CACHE_RETENTION` / flag `--prompt-cache-retention`

Debug:

```bash
python -m scripts.evaluate https://example.com --debug-web-search
```

### `scripts.evaluate_list` (run a list)

Inputs:
- CSV (default): specify `--url-column` and `--name-column`
- TXT: one URL per line (`--input-format txt`)

Example: random sample of 200 unique URLs:

```bash
python -m scripts.evaluate_list ^
  --input "data/Manuav Company DE 5-30 B2B.csv" --input-format csv ^
  --url-column Website --name-column Company --score-column - --bucket-column - ^
  --random-sample 200 --seed 42 ^
  --max-tool-calls 2 --service-tier flex --resume
```

Most useful flags/envs:
- **Input**
  - env `MANUAV_INPUT_PATH` / flag `--input`
  - env `MANUAV_INPUT_FORMAT` / flag `--input-format` (`auto/csv/txt`)
  - env `MANUAV_CSV_DELIMITER` / flag `--csv-delimiter`
  - env `MANUAV_URL_COLUMN` / flag `--url-column`
  - env `MANUAV_NAME_COLUMN` / flag `--name-column`
  - env `MANUAV_SCORE_COLUMN` / flag `--score-column` (use `-` to disable)
  - env `MANUAV_BUCKET_COLUMN` / flag `--bucket-column` (use `-` to disable)
- **Sampling / run control**
  - env `MANUAV_RANDOM_SAMPLE` / flag `--random-sample`
  - env `MANUAV_SAMPLE_SEED` / flag `--seed`
  - env `MANUAV_DEDUPE` / flag `--dedupe`
  - env `MANUAV_RESUME` / flag `--resume`
  - env `MANUAV_LIMIT` / flag `--limit`
  - env `MANUAV_CONTINUE_ON_ERROR` / flag `--continue-on-error`
  - env `MANUAV_PROGRESS_EVERY` / flag `--progress-every`
  - flag `--sleep` (politeness / rate limiting)
- **Model / tool budget**
  - env `OPENAI_MODEL` / flag `--model`
  - env `MANUAV_MAX_TOOL_CALLS` / flag `--max-tool-calls`
  - env `MANUAV_SECOND_QUERY_ON_UNCERTAINTY` / flag `--second-query-on-uncertainty`
  - env `MANUAV_RETRY_DISAMBIGUATION_ON_LOW_CONFIDENCE` / flag `--retry-disambiguation-on-low-confidence`
  - env `MANUAV_RETRY_MAX_TOOL_CALLS` / flag `--retry-max-tool-calls`
- **Debug/audit**
  - env `MANUAV_DEBUG_WEB_SEARCH` / flag `--debug-web-search` (adds query/open breakdown fields + stores `web_search_debug` in JSONL)

Tip: for “investigatory” runs, enable `--debug-web-search` and consider using `scripts.trace_web_search --include-sources` on a small sample (not bulk).

### `scripts.trace_web_search` (investigate tool behavior)

This is specifically for understanding:
- how often the model uses 1 vs 2 queries
- query vs open/visit breakdown
- optional “sources” output for auditing (increases tokens)

Example:

```bash
python -m scripts.trace_web_search ^
  --input "data/Manuav Company DE 5-30 B2B.csv" --url-column Website --name-column Company ^
  --sample 25 --seed 42 --max-tool-calls 3 --service-tier flex --include-sources
```

### Billing/cost tips (OpenAI)

- **Web search billing**: OpenAI “Web Searches” appears to bill **query** calls at **$0.01 per query**. Opens/visits still count as tool calls, but do not appear to be billed the same way.
- **Flex**: token pricing can be discounted, but **web search tool costs are not discounted**.
- **Cost fields in outputs**:
  - `web_search_calls` is the **billed query count**
  - `web_search_tool_calls_total` is total tool invocations (query + open/visit)


