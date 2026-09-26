# Vera Signal-to-Action Engine

A hybrid merchant-engagement bot for the magicpin Vera challenge.

## What makes this different

The LLM is **not** the decision maker. The system first selects the strongest merchant/customer signal, creates an evidence-bound draft, and only then (optionally) asks an LLM to polish the language. A post-generation evidence guard rejects unsupported numbers/names, category taboo language, and internal implementation jargon.

Four deliberate design choices:

1. **Signal ranking** — trigger urgency, specificity, customer relevance, merchant state, and recency are combined to choose what deserves attention now. One proactive message per merchant per tick preserves attention.
2. **Evidence ledger / guard** — every factual detail is sourced from category, merchant, trigger, or customer context. The optional LLM cannot introduce new numeric facts.
3. **Attention memory** — trigger-level dedup plus merchant-level recent-send memory reduces repeated topics and spam.
4. **Conversation intelligence** — global auto-reply detection, explicit commitment detection, graceful stop handling, defer/wait behavior, and off-topic containment.

## API

- `GET /v1/healthz`
- `GET /v1/metadata`
- `POST /v1/context`
- `POST /v1/tick`
- `POST /v1/reply`

## Local run

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Then:

```bash
curl http://localhost:8080/v1/healthz
```

## Optional LLM polish

Set `ENABLE_LLM_POLISH=1`, `OPENAI_API_KEY`, and `OPENAI_MODEL`. The LLM receives the already selected draft plus allowed evidence. If the API fails or the output introduces an unsupported fact, the deterministic draft is returned instead.

## Generate canonical submission JSONL

The supplied challenge package has a deterministic dataset generator. After generating the expanded dataset beside this folder:

```bash
python scripts/generate_submission.py
```

This writes `submission.jsonl` with the 30 canonical test pairs.

## Tests

```bash
pytest -q
```

## Deployment

Render can use `render.yaml`, or Docker can use the included `Dockerfile`.

## Important submission note

Do not add external merchant/customer data sources. The challenge permits commercial LLM APIs but prohibits sending payload data to non-LLM external APIs. Keep context state ephemeral and rely only on the contexts pushed by the judge.
