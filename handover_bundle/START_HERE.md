## Start Here (handover bundle)

This folder is a **copyable framework**. The goal is to use it as a template for a *new* repo that detects which **ecommerce platform** a shop uses (Magento / Shopware / WooCommerce / Shopify / Other / Unknown) using:
- OpenAI Responses API
- Web Search tool enabled
- Flex processing (optional) to reduce token costs
- Prompt caching (optional) when supported
- Incremental output to **JSONL + CSV**, including usage + estimated costs

### What’s in this bundle
- **Core library**: `handover_bundle/manuav_eval/`
  - prompt + OpenAI call: `manuav_eval/evaluator.py`
  - strict JSON schema: `manuav_eval/schema.py`
  - cost estimation: `manuav_eval/costing.py`
  - rubric loading: `manuav_eval/rubric_loader.py`
- **CLIs/workflows**: `handover_bundle/scripts/`
  - single URL: `scripts/evaluate.py`
  - list runner (CSV/TXT → JSONL+CSV): `scripts/evaluate_list.py`
  - web-search tracer: `scripts/trace_web_search.py`
  - run analyzer: `scripts/analyze_run.py`
- **Rubrics**: `handover_bundle/rubrics/` (you will add a new rubric file for the new project)
- **Docs**: `handover_bundle/README.md`, `handover_bundle/docs/USAGE.md`
- **Tests**: `handover_bundle/tests/` (use these as patterns; update them as you refactor)

### How to use this bundle in a new repo (high-level)
1. **Copy the whole folder** contents into a new repository root.
2. **Create `.env`** from the template: copy `env_example.md` → `.env` (the CLIs use `.env` for defaults).
3. **Rename the project prefix** (recommended): replace `MANUAV_…` env vars with a new prefix (e.g. `SHOPTECH_…`) so it doesn’t clash with the Manuav project.
4. **Replace the business logic**:
   - update `manuav_eval/schema.py` to a new “platform detection” output schema
   - add a new rubric file under `rubrics/` and point `DEFAULT_RUBRIC_FILE` to it
   - rewrite the prompt in `manuav_eval/evaluator.py` to follow the new rubric and find platform signals
5. **Update list output columns** in `scripts/evaluate_list.py` (CSV headers + JSONL record fields) to match the new schema.
6. Keep the tool/cost infrastructure:
   - `max_tool_calls` as a guardrail (usually 2)
   - optional **soft** second-query toggle
   - optional **gated retry** toggle on low confidence (safe, only triggers on “sticky” cases)
   - cost estimation from OpenAI usage metadata + env pricing
7. **Run tests** and adjust them to the new schema/rubric:
   - `python -m pytest -q`

### What to follow next
- Read `handover_bundle/project_build_plan.md` and implement it in the new repo.
- Use `handover_bundle/docs/USAGE.md` as the source of truth for flags/env patterns and recommended presets.


