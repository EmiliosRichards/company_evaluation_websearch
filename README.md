## Manuav Company Evaluation (single LLM call + web search)

This repo is a small pipeline that:
- takes a **company website URL**
- runs **one** LLM call with **web search enabled**
- returns a **Manuav Fit score (0–10)** + recommendation + evidence (JSON)

Supported providers:
- **OpenAI** (using `gpt-4.1-mini`, `gpt-4o`, `gpt-5.x`, etc.) with built-in Web Search tool.
- **Google Gemini** (using `gemini-3-flash-preview`, `gemini-2.0-flash`, etc.) with Grounding with Google Search.

### Setup

- **Install**:

```bash
python -m pip install -r requirements.txt
python -m pip install google-genai  # Optional: for Gemini support
```

- **Create a `.env`** (recommended):

```bash
copy .env.example .env
```

Then edit `.env` and set `OPENAI_API_KEY` or `GEMINI_API_KEY`.

Optional (model override):

```bash
OPENAI_MODEL="gpt-4.1-mini"
# or for Gemini:
GEMINI_MODEL="gemini-3-flash-preview"
```

Optional (rubric override / versioning):

```bash
MANUAV_RUBRIC_FILE="rubrics/manuav_rubric_v4_en.md"
```

Optional (cap web-search/tool calls per company - OpenAI only):

```bash
MANUAV_MAX_TOOL_CALLS=8
```

Optional (pricing for cost reporting; USD per 1M tokens - OpenAI only):

```bash
MANUAV_PRICE_INPUT_PER_1M=1.75
MANUAV_PRICE_CACHED_INPUT_PER_1M=0.175
MANUAV_PRICE_OUTPUT_PER_1M=14.00
```

Optional (pricing for cost reporting; USD per 1M tokens - Gemini only):

```bash
GEMINI_PRICE_INPUT_PER_1M=0.50
GEMINI_PRICE_OUTPUT_PER_1M=3.00
GEMINI_PRICE_SEARCH_PER_1K=35.00
```

Optional (prompt caching to reduce repeated rubric/system input cost - OpenAI only):

```bash
MANUAV_PROMPT_CACHE=1
MANUAV_PROMPT_CACHE_RETENTION=24h
```

Optional (reasoning effort override; default is auto - OpenAI only):

```bash
MANUAV_REASONING_EFFORT=low
```

Optional (Flex processing - OpenAI only):

- Flex can be **cheaper** (Batch-rate token pricing) but **slower** and may occasionally return `429 Resource Unavailable` (not charged).
- If you enable Flex, consider increasing timeouts.

```bash
MANUAV_SERVICE_TIER=flex
MANUAV_OPENAI_TIMEOUT_SECONDS=900
MANUAV_FLEX_MAX_RETRIES=5
MANUAV_FLEX_FALLBACK_TO_AUTO=1
```

### Run (OpenAI)

```bash
python -m scripts.evaluate https://company.com
```

To also show estimated cost on stderr (JSON output remains on stdout):

```bash
python -m scripts.evaluate https://company.com
```

To suppress cost printing:

```bash
python -m scripts.evaluate https://company.com --no-cost
```

### Run (Google Gemini)

Ensure `GEMINI_API_KEY` is set.

```bash
python -m scripts.evaluate_gemini https://company.com
```

### Irene sample (9 rows: 3 low / 3 mid / 3 high)

Create the sample:

```bash
python -m scripts.make_irene_sample
```

Run the evaluator on the sample (writes JSONL + prints MAE):

```bash
python -m scripts.evaluate_list
```

This also writes a **timestamped CSV** to `outputs/`. Add a suffix with `-s`:

```bash
python -m scripts.evaluate_list -s baseline
```

Files:
- `data/Websearch Irene - Manuav AI Search.csv`: Irene’s full manual research
- `data/irene_sample_9.csv`: sampled 9-row subset
- `outputs/<timestamp>[_suffix].jsonl`: model results for the sample (JSONL)
- `outputs/<timestamp>[_suffix].csv`: flattened results for the sample (CSV)

### Output

Scripts print strict JSON including:
- `manuav_fit_score` (0–10)
- `confidence` (low/medium/high)
- `reasoning` (short: why this score, per rubric)

Note: sources/URLs are not included in the JSON output to save output tokens. For OpenAI, you can extract citations from the Responses API output annotations when web search is enabled.

### Batch API (OpenAI, async, ~50% cheaper)

You can run Irene samples via the Batch API (asynchronous, completes within 24h):

Important limitation:
- **Web search tools are not supported in the OpenAI Batch API** right now. If you include `tools` in batch requests, the batch will fail with `web_search_unsupported`.
- That means Batch is only suitable for runs where you provide all evidence in the prompt (no tool use), or for other non-tool workloads.

1) Create a batch job (writes an input JSONL, uploads it, creates the batch):

```bash
python -m scripts.run_irene_sample_batch create --sample data/irene_sample_9_seed99.csv --model gpt-5-mini-2025-08-07 --suffix seed99 --no-web-search
```

2) Check status:

```bash
python -m scripts.run_irene_sample_batch status <batch_id>
```

3) Fetch results (downloads output file and writes results CSV/JSONL):

```bash
python -m scripts.run_irene_sample_batch fetch <batch_id> --suffix seed99
```

Cost notes:
- Batch costs are discounted by ~50% vs sync; the fetch step applies a `MANUAV_BATCH_DISCOUNT` multiplier (default 0.5) to estimate total cost.
