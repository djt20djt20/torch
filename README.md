# Torch Take-Home Exercise

## Overview

This exercise is designed to take **2–3 hours**. We are not looking for a perfect solution — we are looking for how you think, the decisions you make, and how you approach building something that could realistically go to production.

Use whatever tools you like, including AI copilots such as Claude Code, Cursor, or GitHub Copilot. We actively encourage this — how you use these tools is part of what we are interested in.

Simple and clear beats complex and clever. If you are spending time gold-plating, stop.

---

## Context

You are joining Torch as the first AI/ML engineer. One of the first things we want to build is a system that helps reviewers assess incoming records more quickly and confidently.

When a new record comes in, a reviewer needs to answer two questions:

1. Is this record likely to result in a loss — i.e. will costs exceed the revenue it generates?
2. What do similar historical records tell us about this one?

Your task is to build a prototype system that helps answer both questions — not as two separate queries, but as an agent that reasons over the record and produces a single, useful recommendation.

**A note on the target variable:** `is_loss_making` is `True` when the actual losses on a record exceeded the premium charged (i.e. `loss_ratio > 1.0`). Your model should predict this outcome before we know the result.

---

## The Task

### Part 1 — Loss Prediction (in the notebook)

Open `notebooks/modelling.ipynb`. Using `records.csv`, build a model that predicts whether a new record is likely to result in a loss. Save your trained model artifact to `app/artifacts/` so the app can load it.

**We care about:**
- How you explore and understand the data before modelling
- Why you chose the model you did
- How you would think about explainability — a reviewer needs to understand why a record is flagged
- What the limitations of your model are

### Part 2 — Assessment Agent (in the app)

Implement the agent logic in `app/tools.py` and `app/model.py`. The retrieval tool (`retrieve_similar_records`) is already wired up — focus on the model prediction tool and the agent's reasoning.

The agent has two tools available: the loss prediction model and the document retrieval system. It should reason over the record and decide how to use them — not call everything blindly.

The agent should produce a recommendation for the reviewer that draws on both tools where appropriate. It should know when it has enough information to make a recommendation, and when it does not.

**We care about:**
- How the agent decides what to do — is it reasoning or just calling everything unconditionally?
- What guardrails you put in place — termination conditions, fallbacks
- How it handles uncertainty — low model confidence, poor retrieval results
- Whether the output is genuinely useful to a non-technical reviewer
- Cost awareness — each LLM call has a cost. We expect the first AI/ML hire to own the AI budget. Where are the cost risks in your agent design, and what did you do about them?

### Part 2b — Agent Evaluation (in evals/)

Implement `score_recommendation()` in `evals/eval.py` and run the harness against `evals/eval_set.json` — 20 records with known outcomes.

**We care about:**
- How you define "correct" for a free-text recommendation — this is not obvious
- Whether your eval catches failure modes the agent might hide behind confident-sounding language
- How you would use this eval to improve the agent over time

### Part 3 — Response Contract (in the app)

The agent loop is already wired to the `/assess` endpoint. Your job is to design `AssessResponse` in `app/schemas.py`.

What fields does a non-technical reviewer actually need to see? What would make this response useful — or useless — in practice? Justify your design in `APPROACH.md`.

**We care about:**
- Whether the response communicates uncertainty in a way a reviewer can act on
- Whether it tells the reviewer when to seek a second opinion
- Whether it would be useful to an auditor six months later

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Add your API key

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
```

Then open `.env` and replace `your-api-key-here` with the key provided separately.

### 3. Verify the data is present

```
data/
├── records.csv       # Historical records with loss outcomes (note: contains data quality issues to handle)
├── documents/        # Plain-English summaries, one per record
└── new_record.json   # The record to assess
```

---

## Running the app

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

- `GET  /health` — liveness check
- `POST /assess` — run the assessment agent on a new record

Interactive docs: `http://localhost:8000/docs`

To test the assess endpoint with the provided example record:

```bash
curl -X POST http://localhost:8000/assess \
  -H 'Content-Type: application/json' \
  -d "{\"record\": $(cat data/new_record.json)}"
```

---

## Running the notebook

```bash
uv run jupyter notebook notebooks/modelling.ipynb
```

Or open it directly in VS Code / Cursor.

---

## Project structure

```
├── data/
│   ├── records.csv            # Historical records with loss outcomes
│   ├── documents/             # Plain-English summaries, one per record
│   └── new_record.json        # New record to run through your system
│
├── notebooks/
│   └── modelling.ipynb        # Start here for Part 1
│
├── app/
│   ├── main.py                # HTTP layer — endpoints only
│   ├── schemas.py             # Request/response contracts — design AssessResponse here (Part 3)
│   ├── agent.py               # Agent loop — pre-wired, set MAX_ITERATIONS and SYSTEM_PROMPT
│   ├── llm.py                 # Anthropic client — pre-configured, ready to use
│   ├── vectorstore.py         # ChromaDB retrieval — pre-built, call retrieve()
│   ├── tools.py               # Tool schemas — retrieval pre-built, implement predict (Part 2)
│   ├── model.py               # Model loader — implement predict() (Part 2)
│   └── artifacts/             # Save your model.pkl here after Part 1
│
├── evals/
│   ├── eval_set.json          # 20 labeled records with ground truth
│   └── eval.py                # Eval harness skeleton — implement score_recommendation() (Part 2b)
│
├── pyproject.toml
└── README.md
```

---

## Data dictionary

### records.csv

| Field | Type | Description |
|---|---|---|
| `record_id` | string | Unique identifier, links to document filename in `data/documents/` |
| `risk_type` | categorical | Category of the record (property, liability, marine, cyber, aviation) |
| `territory` | categorical | Geographic territory |
| `industry` | categorical | Industry sector of the insured |
| `limit` | float | Maximum exposure in USD |
| `premium` | float | Revenue generated from this record in USD |
| `broker` | categorical | Originating broker |
| `prior_claims` | integer | Number of prior claims on this record |
| `years_trading` | integer | Years the counterparty has been trading |
| `loss_ratio` | float | Actual outcome — losses divided by premium |
| `is_loss_making` | boolean | True if `loss_ratio > 1.0` |

### new_record.json

A single record in the same structure as above, without `loss_ratio` or `is_loss_making`.

---

## What to submit

A GitHub repo (public or private — if private, please invite us) containing:

- Your completed notebook (`notebooks/modelling.ipynb`)
- Your completed app logic (`app/tools.py`, `app/model.py`, `app/main.py`)
- Your completed eval harness (`evals/eval.py`) with results
- Your saved model artifact (`app/artifacts/model.pkl`)
- A short `APPROACH.md` (one page maximum) covering:
  - The decisions you made and why
  - What you would do differently with more time
  - Where you used AI copilots and how
  - How you would design a feedback loop so that reviewer decisions improve the system over time
  - What you would need to think about before putting this into production — including cost, monitoring, and when the system should defer to a human

---

## What we are not looking for

- A perfect model or state-of-the-art retrieval
- Changes to the plumbing we have provided — focus on the logic
- A large codebase — simple and clear beats complex and clever
