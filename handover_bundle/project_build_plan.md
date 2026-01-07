### What you’re building (target system)
- **Input**: a single shop/company URL (domain).
- **Process**: one OpenAI Responses API call with **Web Search tool** enabled; optionally **Flex** + **prompt caching**.
- **Output per URL** (JSONL + CSV row, written incrementally): detected ecommerce platform + confidence + short reasoning + (optional) sources/citations + usage + cost + tool-call counts + retry metadata.

Recommendation: normalize terminology to **“shop system / ecommerce platform”** (your examples are platforms, not payment processors).

---

### Definition of Done (DoD)
- **Core behavior**
  - **URL in → platform out**: for every input URL, the system emits exactly one classification: `magento|shopware|woocommerce|shopify|other|unknown`.
  - **Failure handling**: unreachable/blocked/ambiguous sites must produce `unknown` (and still write a row) rather than crashing the run.
  - **Short reasoning**: reasoning stays within the defined limit (2–4 sentences, ≤600 chars).
- **Evidence & confidence**
  - **Evidence tiering** is implemented and reflected in output (e.g., `A|B|C`).
  - **Confidence mapping** is consistent with evidence tiers (`high/medium/low`) and uses `low` for ambiguous/unknown cases.
  - **No URL leakage**: reasoning must not contain raw URLs; any URLs belong only in optional debug-only sources outputs.
- **Output contract**
  - The OpenAI response is enforced via a **strict JSON Schema** (no extra fields).
  - JSONL output includes: input URL, platform classification, confidence, reasoning, and run metadata (usage + costs + tool-call counts).
  - CSV output includes flattened columns matching the JSONL record, plus cost and tool-count columns for analysis.
- **Costing & billing semantics**
  - Token cost is computed from OpenAI `usage` and env pricing inputs.
  - Web-search cost is computed from **billed query count** (not total tool invocations).
  - Flex discount (if enabled) is applied to **token cost only**, not web-search tool cost.
- **Tool budget controls**
  - `max_tool_calls` is plumbed through and acts as a guardrail.
  - The “soft” second-query-on-uncertainty toggle exists and does not force 2 queries for every company.
  - The “gated retry on low confidence” toggle exists; when it triggers:
    - totals (tool calls, billed query count, token usage, costs) are **aggregated** across attempts
    - output includes retry metadata (`retry_used`, `retry_selected`)
- **Operational features**
  - List runner supports: CSV/TXT input, column mapping, sampling, dedupe, resume, and continue-on-error.
  - Progress reporting and run duration fields are present.
- **Observability**
  - Debug mode can record query/open/unknown tool-call breakdown (and total tool calls).
  - Tracing script exists for small-sample audits (optional sources list).
- **Quality**
  - Tests updated to the new schema/rubric and pass (`pytest`).
  - README/usage docs updated for the new project prefix, toggles, and recommended presets.

### Comprehensive rubric (what “good” looks like)
You want a rubric that scores **detection correctness + evidence quality + determinism**. I’d structure it like this:

#### 1) Primary classification (required)
Pick exactly one:

- **Magento (PASS)**: Magento / Adobe Commerce detected
- **Shopware (PASS)**: Shopware detected
- **WooCommerce (WEAK PASS / not ideal)**: WooCommerce detected
- **Shopify (FAIL)**: Shopify detected
- **Other (FAIL)**: e.g., BigCommerce, Wix, Squarespace, custom, etc.
- **Unknown (FAIL)**: insufficient evidence / ambiguous / inaccessible

#### 2) Evidence strength (how confident we are)
Define evidence tiers (this is key for reliable automation):

- **Tier A (direct, best)**: First-party evidence from the site itself, e.g.
  - HTML markers, script paths, meta tags, cookies, asset URLs, known endpoints
  - Examples of “direct signals” you can tell the model to look for:
    - Magento: `/static/`, `mage/`, `Magento_`, `form_key`, `X-Magento-*`, “Magento 2” in page source, etc.
    - Shopware: `/widgets/`, `/bundles/storefront/`, “Shopware 6” hints, etc.
    - WooCommerce: `wp-content/plugins/woocommerce`, `woocommerce_params`, `wp-json/wc/`
    - Shopify: `cdn.shopify.com`, `Shopify.theme`, `myshopify.com`, `shopify-section`
- **Tier B (strong, third-party)**: reputable tech-profiler sources (BuiltWith, Wappalyzer, SimilarTech), ideally **2+ sources agree**
- **Tier C (weak)**: indirect hints (blog posts, job ads, agency case studies, single low-quality directory)

Rule: **Tier A overrides** Tier B/C if contradictory (because it’s first-party).

#### 3) Decision rules (avoid wrong “passes”)
- If **Shopify** is detected with Tier A or Tier B → **FAIL** even if “also mentions WooCommerce” elsewhere.
- If **multiple platforms** appear (common with migrations, blog mentions, agencies):
  - prefer **the platform used by the target domain**, not an agency’s other clients
  - if ambiguous → **Unknown (FAIL)** + low confidence
- If the **domain is dead / parked / redirects oddly** and no reliable evidence → **Unknown (FAIL)**.

#### 4) Confidence output (required)
Map to evidence tier:
- **high**: Tier A, or Tier B with strong agreement
- **medium**: Tier B single source, or Tier C with multiple aligned hints
- **low**: Tier C only, conflicting signals, or unknown

#### 5) Minimal output constraints
Keep reasoning short (you already have this pattern): **2–4 sentences, ≤600 chars**.

---

### Exact build plan (modular pipeline), based on this repo’s proven structure
You can reuse the same architecture with a few targeted swaps:

#### A) Core library module (copy + rename)
1. Copy `manuav_eval/` into the new repo, rename package to something like `shoptech_eval/`.
2. Update the exported functions in `__init__.py` to match new naming if desired.

**What you change:**
- `schema.py`: define a new strict output schema for “platform detection”.
- `rubric_loader.py`: point the default rubric to your new rubric file.
- `evaluator.py`: replace the Manuav prompt with the new rubric + instructions for platform detection.

#### B) Output schema (strict JSON)
Define a schema like:
- `input_url` (string)
- `final_platform` (enum: `magento|shopware|woocommerce|shopify|other|unknown`)
- `platform_family` (optional grouping if useful)
- `confidence` (`low|medium|high`)
- `evidence_tier` (`A|B|C`)
- `signals` (short list of strings, max N) — optional but useful
- `reasoning` (short string, max 600)
- optionally `sources` (debug-only, max 8), using the same “strict schema” pattern you already implemented

Keep `OUTPUT_SCHEMA_WITH_SOURCES` as a debug/audit option (same idea as this project).

#### C) Evaluator prompt (the “engine”)
In `evaluator.py`, rewrite only the prompt block to:
- explicitly define the rubric (above)
- instruct the model to:
  - attempt a **single query** by default
  - be conservative if evidence is missing
  - never output URLs in reasoning
- optionally keep your existing toggles:
  - `MANUAV_SECOND_QUERY_ON_UNCERTAINTY=1` (rename to your new prefix) = soft
  - `MANUAV_RETRY_DISAMBIGUATION_ON_LOW_CONFIDENCE=1` = hard gated retry (recommended)

#### D) CLI scripts (copy mostly as-is)
- `scripts/evaluate.py`: becomes `scripts/detect_platform.py` (or keep name) for single URL.
- `scripts/evaluate_list.py`: becomes `scripts/detect_platform_list.py` for CSV/TXT runs:
  - keep: sampling, dedupe, resume, JSONL+CSV streaming, cost fields, debug web-search counts
  - change: column names and output columns to match platform schema
- `scripts/trace_web_search.py`: keep for auditing + “include sources” sampling
- `scripts/analyze_run.py`: keep for summarizing costs/tool usage

#### E) Config model (.env + args)
Reuse the same pattern:
- env provides defaults, CLI overrides per run.
- keep the cost config approach: pricing is *input variables*, usage comes from API response.

**Key configs you’ll want (rename prefix):**
- **Core**: `OPENAI_API_KEY`, `OPENAI_MODEL`
- **Tool budget**: `*_MAX_TOOL_CALLS`
- **Soft extra query**: `*_SECOND_QUERY_ON_UNCERTAINTY`
- **Gated retry**: `*_RETRY_DISAMBIGUATION_ON_LOW_CONFIDENCE`, `*_RETRY_MAX_TOOL_CALLS`
- **Flex**: `*_SERVICE_TIER=flex`, `*_OPENAI_TIMEOUT_SECONDS=900`, `*_FLEX_MAX_RETRIES`, `*_FLEX_FALLBACK_TO_AUTO`
- **Prompt caching**: `*_PROMPT_CACHE=1`, `*_PROMPT_CACHE_RETENTION=24h`
- **Pricing for reporting**: `*_PRICE_INPUT_PER_1M`, `*_PRICE_CACHED_INPUT_PER_1M`, `*_PRICE_OUTPUT_PER_1M`, `*_PRICE_WEB_SEARCH_PER_1K`
- **List runner**: input path/format, column mapping, sampling/resume flags, `*_DEBUG_WEB_SEARCH`

Cost reminder: **Flex discounts tokens only; Web Search tool billing is separate** (your framework already models that correctly).

---

### Files in “handover bundle”

#### **Core (required)**
- `requirements.txt`
- `manuav_eval/` (entire directory)
  - `manuav_eval/__init__.py`
  - `manuav_eval/evaluator.py`
  - `manuav_eval/schema.py`
  - `manuav_eval/costing.py`
  - `manuav_eval/rubric_loader.py`
  - (optional depending on new project): `manuav_eval/openai_batch.py`, `manuav_eval/gemini_evaluator.py`
- `scripts/`
  - `scripts/evaluate.py`
  - `scripts/evaluate_list.py`
  - `scripts/trace_web_search.py`
  - `scripts/analyze_run.py`
  - `scripts/__init__.py`
- `rubrics/` (entire directory; you’ll add a new rubric file here)
- `README.md`
- `docs/USAGE.md`

#### **Dev quality (strongly recommended)**
- `tests/` (entire directory) — even if you later rewrite tests, this gives you the exact harness patterns.
- `scripts/run_tests_coverage.py`

#### **Config templates**
- `env_example.md` (copy it to `.env` in the new repo)

---

### The “minimum modifications” checklist for your new project
- Add new rubric: `rubrics/shop_platform_rubric_v1.md`
- Update default rubric path in `rubric_loader.py`
- Replace Manuav prompt text in `evaluator.py` with platform-detection instructions
- Replace schema in `schema.py` with platform output fields
- Update CSV fieldnames in the list runner to match new output
- Rename scripts (optional) + update docs

---

### Recommended `.env` for the new “shop platform detector” project
Keep the same structure, but **rename the prefix** from `MANUAV_…` to something project-specific (e.g. `SHOPTECH_…`). This prevents collisions when you run both repos on the same machine.

A clean mapping:

- **`OPENAI_API_KEY`, `OPENAI_MODEL`**: keep as-is.
- Rename:
  - `MANUAV_MAX_TOOL_CALLS` → `SHOPTECH_MAX_TOOL_CALLS`
  - `MANUAV_SECOND_QUERY_ON_UNCERTAINTY` → `SHOPTECH_SECOND_QUERY_ON_UNCERTAINTY`
  - `MANUAV_RETRY_DISAMBIGUATION_ON_LOW_CONFIDENCE` → `SHOPTECH_RETRY_DISAMBIGUATION_ON_LOW_CONFIDENCE`
  - `MANUAV_RETRY_MAX_TOOL_CALLS` (add this explicitly) → `SHOPTECH_RETRY_MAX_TOOL_CALLS=3`
  - Pricing:
    - `MANUAV_PRICE_INPUT_PER_1M` → `SHOPTECH_PRICE_INPUT_PER_1M`
    - `MANUAV_PRICE_CACHED_INPUT_PER_1M` → `SHOPTECH_PRICE_CACHED_INPUT_PER_1M`
    - `MANUAV_PRICE_OUTPUT_PER_1M` → `SHOPTECH_PRICE_OUTPUT_PER_1M`
    - `MANUAV_PRICE_WEB_SEARCH_PER_1K` → `SHOPTECH_PRICE_WEB_SEARCH_PER_1K`
  - Caching:
    - `MANUAV_PROMPT_CACHE` → `SHOPTECH_PROMPT_CACHE`
    - `MANUAV_PROMPT_CACHE_RETENTION` → `SHOPTECH_PROMPT_CACHE_RETENTION`
  - Flex:
    - `MANUAV_SERVICE_TIER` → `SHOPTECH_SERVICE_TIER`
    - `MANUAV_OPENAI_TIMEOUT_SECONDS` → `SHOPTECH_OPENAI_TIMEOUT_SECONDS`
    - `MANUAV_FLEX_MAX_RETRIES` → `SHOPTECH_FLEX_MAX_RETRIES`
    - `MANUAV_FLEX_FALLBACK_TO_AUTO` → `SHOPTECH_FLEX_FALLBACK_TO_AUTO`
    - `MANUAV_FLEX_TOKEN_DISCOUNT` → `SHOPTECH_FLEX_TOKEN_DISCOUNT`

(Under the hood you’ll just update `scripts/*` + `costing.py` env reads to use the new prefix.)

---

### Suggested defaults for your specific goal (platform detection)
Given “Magento / Shopware pass; WooCommerce weak; Shopify fail; Unknown fail”:

- **Tool budget**:
  - `SHOPTECH_MAX_TOOL_CALLS=2` (good baseline guardrail)
  - Leave `SHOPTECH_SECOND_QUERY_ON_UNCERTAINTY=1` as optional; it’s a soft nudge.
- **Fallback retry**:
  - Keep `SHOPTECH_RETRY_DISAMBIGUATION_ON_LOW_CONFIDENCE=1` **enabled** for production list runs, because a lot of “Unknown” will be due to ambiguity/redirects/blocked pages.
  - Set `SHOPTECH_RETRY_MAX_TOOL_CALLS=3` so the retry has headroom for 2 queries + possible open/visit.

---

### Prompt-caching note (your comment about incompatibility)
In this framework, prompt caching is **best-effort**:
- If the model supports it, it helps.
- If it doesn’t (some models/tiers), our evaluator already retries without `prompt_cache_retention` when needed.

So for the new project:
- Keep:
  - `SHOPTECH_PROMPT_CACHE=1`
  - `SHOPTECH_PROMPT_CACHE_RETENTION=24h`
- And rely on the existing “fallback if unsupported” behavior.

---

### What you should change in code for the new project (high-signal checklist)
- **Rename env prefix everywhere** (`MANUAV_` → `SHOPTECH_`) in:
  - `scripts/evaluate.py`
  - `scripts/evaluate_list.py`
  - `scripts/trace_web_search.py` (if you want it to inherit defaults)
  - `manuav_eval/costing.py`
- **New schema** in `schema.py`:
  - `final_platform` enum (`magento|shopware|woocommerce|shopify|other|unknown`)
  - `confidence`
  - `evidence_tier` (`A|B|C`)
  - `reasoning` (short)
  - optional debug `sources`
- **New rubric file** in `rubrics/` and point default in `rubric_loader.py`
- **Rewrite the evaluator prompt** in `evaluator.py` to:
  - search specifically for platform signals (Magento/Shopware/WooCommerce/Shopify markers)
  - enforce decision rules + evidence tiers

