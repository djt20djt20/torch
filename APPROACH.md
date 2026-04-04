# APPROACH.md

## Decisions made and why

### Part 1 — Loss Prediction

**Model choice: Logistic Regression with calibration**
Logistic regression was chosen over gradient boosted trees for two reasons: it produces well-understood, linear feature attributions that a reviewer can interpret directly, and it degrades gracefully on a dataset of ~500 records where overfitting is a real risk. A `StandardScaler → LogisticRegression` pipeline wrapped in `CalibratedClassifierCV` (cv=5, Platt scaling) gives calibrated probabilities, which matter here — a reviewer acting on "82% confident loss-making" needs that number to mean something.

**Feature engineering**
Two derived features were added: `premium_rate` (premium / limit) and `prior_claim_rate` (prior_claims / years_trading). Both encode domain-relevant ratios that the raw fields don't express — a £5k premium on a £10M limit is qualitatively different from £5k on a £100k limit. The four categorical columns are target-encoded and saved in the artifact so inference applies the same transform the model was trained on.

**Data quality handling**
Obvious transcription errors — `limit <= 50`, `limit >= 999,999,999`, `premium == 0`, `premium >= 14,000,000`, `years_trading < 0` — were replaced with training-set medians rather than dropped, on the grounds that a real incoming record with a bad field shouldn't be rejected entirely. These corrections are mirrored exactly in `model.py` so training and inference behave identically.

**Outlier bounds**
Changed from 1st/99th to 5th/95th percentile during the session. This widens the "warn but don't reject" band — appropriate given the small dataset where extreme percentiles are noisy.

**SHAP**
`LinearExplainer` with a zero-vector background (the mean of scaled space). Using the input row itself as background was the original code path and produced all-zero SHAP values; the fix ensures attributions reflect deviation from the training-set average, which is the interpretable quantity a reviewer cares about.

---

### Part 2 — Assessment Agent

**MAX_ITERATIONS: 4**
The agent needs at most two tool calls (one `predict_loss`, one `retrieve_similar_records`) plus one synthesis turn. Setting the limit to 4 allows for one retry if a tool fails while capping runaway loops. Setting it higher is a cost risk with no upside — the agent has no useful third action to take. Setting it lower risks cutting off the final synthesis after both tools respond.

**SYSTEM_PROMPT design**
The prompt instructs the agent to:
- Call `predict_loss` first, always — it's cheap and provides the quantitative anchor.
- Call `retrieve_similar_records` with a natural-language description of the account, not a field dump. This produces better semantic matches.
- Weight model confidence explicitly: below 0.55 confidence, the recommendation should say so and flag for human review.
- Produce a recommendation in plain English — no JSON, no raw scores dropped in without context. The target reader is an underwriter, not an engineer.
- When retrieval returns high cosine distance (poor match), say so rather than citing distant analogues as if they were relevant.

What the prompt explicitly rules out: calling tools more than once each, citing `record_id`s as if they are meaningful to the reviewer, producing a recommendation without having called `predict_loss`.

---

### Part 3 — Response contract

`AssessResponse` keeps `recommendation` (the core output), `tools_used` (auditability), and the existing `record_id`. Fields worth adding with more time: `confidence` (float, surfaced from the model so the caller can gate on it without parsing text), `requires_review` (bool, set when confidence < threshold or warnings were raised), and `similar_record_ids` (for linking to the retrieved cases). What was deliberately left out: raw SHAP values and feature lists — these belong in a debug endpoint, not in the reviewer-facing response.

---

## Evaluation approach

`score_recommendation()` uses an LLM-as-judge prompt that extracts the implied direction (loss-making / not) and expressed certainty from the recommendation text, then compares to `is_loss_making`. A regex approach was rejected — it will miss hedged language ("likely", "borderline") and give false confidence. The three dimensions scored: outcome alignment (did it point the right way?), calibration (was expressed certainty proportionate to how borderline the case was?), and safe deferral (did it escalate when it should have?). The `case_type` field in the eval set is used to separate clear and borderline cases, since a wrong answer on a borderline record is less serious than a confidently wrong answer on a clear one.

---

## What I would do differently with more time

- Train on a larger, rebalanced dataset — the class imbalance in `records.csv` is ~40/60 and the small size means held-out metrics have wide confidence intervals.
- Replace target encoding with a proper Bayesian average to reduce leakage on rare categories.
- Add a fallback: if `predict_loss` returns a `FileNotFoundError`, the agent should still produce a retrieval-only recommendation rather than failing.
- Instrument token usage per agent call and set a hard budget via `max_tokens` on the synthesis turn.

---

## Where the AI tool made suggestions I changed or rejected

- The initial SHAP code used the input row as the `LinearExplainer` background — this produced all-zero values. I diagnosed and fixed this: the background must be a zero vector representing the training mean in scaled space.
- The original outlier bounds were 1st/99th percentile. I changed these to 5th/95th to reduce false OOD warnings on a small dataset where extreme percentiles are unreliable.
- The AI drafted a `SYSTEM_PROMPT` that called both tools unconditionally in a fixed order. I replaced this with conditional logic: retrieval is skipped if the model produces a clearly high-confidence result and no unusual features are flagged — unnecessary retrieval calls add latency and cost without improving the recommendation.

---

## Production readiness

**Degradation signals:** rising `loss_ratio` on records the model predicted as non-loss-making; model confidence distribution shifting towards 0.5 (the model becoming uncertain about everything); feature distribution drift on `premium_rate` or `prior_claim_rate` (the two most predictive engineered features). These should be monitored as population statistics on a rolling 30-day window.

**Feedback loop:** reviewer accept/reject decisions should be captured as labels. More usefully, disagreements — where a reviewer overrides a confident recommendation — should be flagged for active review. These are the cases most likely to reveal systematic failure modes: a risk type that has shifted, a new territory, or a broker whose book has deteriorated. Storing the full `shap_values` dict alongside each decision creates the audit trail needed to identify which features the model is over-weighting when it's wrong.

**Before production:** (1) latency — the agent makes at minimum 2 LLM calls per record; the p99 should be measured and a timeout set; (2) cost — at Claude Sonnet pricing, 20k assessments/month at ~2k tokens per call is non-trivial; a cheaper model should be evaluated for the synthesis step; (3) the model should have a hard confidence threshold below which it does not return a prediction — it returns a `requires_human_review` flag instead; (4) the vectorstore should be rebuilt whenever new historical records are added, with a checksum to detect stale embeddings.
