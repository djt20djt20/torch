# APPROACH.md

## Decisions made and why

### Part 1 — Loss Prediction

**Model choice: Logistic Regression with calibration**
Logistic regression was chosen over gradient boosted trees for two reasons: it produces well-understood, linear feature attributions that a reviewer can interpret directly, and it degrades gracefully on a dataset of ~500 records where overfitting is a real risk. A `StandardScaler → LogisticRegression` pipeline wrapped in `CalibratedClassifierCV` (cv=5, Platt scaling) gives calibrated probabilities — a reviewer acting on "82% confident loss-making" needs that number to mean something, not just rank-order cases.

**Feature engineering**
Two derived features were added: `premium_rate` (premium / limit) and `prior_claim_rate` (prior_claims / years_trading). Both encode domain-relevant ratios the raw fields don't express — a £5k premium on a £10M limit is qualitatively different from £5k on a £100k limit. The four categorical columns are target-encoded and the encoder is saved in the artifact so inference applies exactly the same transform the model was trained on.

**Explainability**
SHAP `LinearExplainer` is used to produce per-feature attributions for every prediction. These are averaged across the 5 calibration folds for stability. The top features by absolute SHAP value are returned alongside the prediction, giving a reviewer a plain-English explanation of what drove the flag — not just a probability.

**Data quality handling**
Obvious transcription errors — `limit <= 50`, `limit >= 999,999,999`, `premium == 0`, `premium >= 14,000,000`, `years_trading < 0` — are replaced with training-set medians rather than dropped, on the grounds that a real incoming record with a bad field shouldn't be rejected entirely. These corrections are mirrored exactly in `model.py` so training and inference are identical. Missing fields are imputed and a warning is surfaced to the reviewer.

**Outlier / OOD flagging**
The model flags features outside the 5th–95th percentile range of training values and includes these as warnings in the prediction output. The agent passes these warnings through to the recommendation, so a reviewer knows when the model is extrapolating.

---

### Part 2 — Assessment Agent

**Architecture: hard-coded prediction, LLM-controlled retrieval**
The key design decision is that `predict_loss` is called unconditionally in Python *before* the LLM loop starts — it is not an LLM tool call. There is no decision to be made about whether to run it; it always runs. The LLM receives the prediction result in its initial context and then decides whether to call `retrieve_similar_records`.

This separation matters: it avoids paying LLM cost to make a decision that isn't actually a decision, and it means the prediction is always available even if the LLM loop fails. The only genuinely non-trivial LLM decision is whether retrieval adds enough value to be worth calling — and if so, what query to write. A natural-language query generated from the record produces better semantic matches than a structured field dump.

**When retrieval is skipped**
The system prompt instructs the LLM that it may skip retrieval if the model confidence is very high (>0.85) and no data quality warnings were raised — in that case the evidence is already sufficient and retrieval adds latency and cost without improving the recommendation. In borderline or uncertain cases, retrieval is always called to provide context.

**MAX_ITERATIONS: 3**
With prediction hard-coded, the agent only needs to: (1) optionally call `retrieve_similar_records`, and (2) synthesise a final answer. Three iterations allows for a small buffer if the LLM emits a reasoning step before its tool call, without leaving room for runaway loops. Setting this higher is a cost risk with no upside.

**Confidence thresholds**
Below 0.55 confidence, the recommendation explicitly flags uncertainty and recommends human review. Above 0.85, the recommendation can be more directive. The band between 0.55 and 0.85 is the "refer for senior review" zone — the model has a view but not a high-conviction one.

**Retrieval quality guard**
If retrieved records have cosine distance above 0.4, the prompt instructs the LLM to say so rather than citing them as if they were relevant. Citing distant analogues as close comparables would be actively misleading to a reviewer.

**Cost awareness**
Each assessment makes one deterministic Python call (model inference, negligible cost) and one LLM call with up to one tool round-trip. At current Claude Sonnet pricing, this is roughly $0.01–0.02 per record. The main cost risk is a high `max_tokens` on the synthesis turn — set to 2048, which is generous but bounded. The alternative risk is `MAX_ITERATIONS` being too high; at 3, a runaway loop has a hard ceiling.

---

### Part 3 — Response contract

`AssessResponse` surfaces: `recommendation` (the core reviewer-facing output), `tools_used` (auditability — what did the agent actually do?), `record_id` (links the response back to the source record), `confidence` (the raw model probability, surfaced as a float so a downstream system can gate on it without parsing text), `requires_review` (bool, true when confidence is below threshold or warnings were raised — gives a machine-readable escalation signal), and `model_warnings` (any data quality or OOD flags the model raised). Raw SHAP values and feature lists are deliberately excluded from the reviewer-facing response — they belong in a debug or audit endpoint.

The `requires_review` field is the most important addition. It means a downstream workflow can automatically route borderline records to a queue without parsing the recommendation text, while still giving the reviewer the full narrative.

---

## Evaluation approach

`score_recommendation()` uses an LLM-as-judge prompt that extracts the implied direction (loss-making / not) and expressed certainty from the free-text recommendation, then compares to `is_loss_making`. A regex or keyword approach was rejected — it would miss hedged language ("likely", "borderline", "moderate confidence") and give false precision. Three dimensions are scored: outcome alignment (did it point the right way?), calibration (was expressed certainty proportionate to how borderline the case was?), and safe deferral (did it escalate when it should have?). The `case_type` field in the eval set separates clear and borderline cases — a wrong answer on a genuinely borderline record is less serious than a confidently wrong answer on a clear one, and the scorer reflects this.

The eval is designed to catch the failure mode the agent is most likely to hide behind: confident-sounding language on a wrong call. A recommendation that says "this record is unlikely to be loss-making" when `is_loss_making` is true fails on outcome alignment regardless of how well-written it is.

---

## What I would do differently with more time

- Train on a larger, rebalanced dataset — the class imbalance in `records.csv` is ~40/60 and the small size means held-out metrics have wide confidence intervals. Bootstrapped confidence intervals on AUC would make the model's limitations more legible.
- Replace target encoding with a proper Bayesian average to reduce leakage on rare categories (brokers or industries with few training examples). A Bayesian average shrinks the encoded value for a rare category towards the global mean, rather than trusting a noisy estimate from a handful of records — e.g. a broker that appears twice in training shouldn't get a very high or very low encoding just because both their records happened to be losses.
- Add a retrieval-only fallback: if `predict_loss` fails at inference time (e.g. model artifact missing or corrupt), the agent should still produce a retrieval-based recommendation with an explicit caveat, rather than erroring out entirely.
- Instrument token usage per agent call and alert when calls exceed a budget threshold — the current implementation has no visibility into per-call cost.
- Run the eval harness as part of CI on every model or prompt change, not just as a one-off.

---

## Where the AI tool made suggestions I changed or rejected

**SHAP background vector:** The initial AI-generated SHAP code used the input row itself as the `LinearExplainer` background. This produced all-zero SHAP values — every feature appears to contribute nothing. The background must be a zero vector representing the training-mean in scaled space; otherwise the explainer has no reference point to compute deviations from. I diagnosed and fixed this.

**Outlier percentile bounds:** The AI defaulted to 1st/99th percentile for OOD flagging. On a dataset of ~500 records, the 1st and 99th percentiles are determined by a handful of extreme values and are unreliable. I changed this to 5th/95th to reduce false positive OOD warnings that would erode reviewer trust in the system.

**Agent architecture — unconditional tool calling:** The AI's initial `SYSTEM_PROMPT` called both tools unconditionally in a fixed sequence. I restructured this: `predict_loss` is now called directly in Python before the LLM loop, and the LLM only decides whether to call `retrieve_similar_records`. This removes a fake decision from the agent (there is no case where you wouldn't run the model), saves one token round-trip when retrieval isn't needed, and makes the agent's actual reasoning task — deciding whether comparables add value, and constructing a useful query — clearer and more honest.

---

## Production readiness

**How I would know the model is degrading**
The clearest signal is rising loss ratio on records the model predicted as non-loss-making — the model's false negatives are the business-critical failure mode. Secondary signals: model confidence distribution drifting towards 0.5 (the model becoming uncertain about everything, which may indicate feature drift); and distributional shift on `premium_rate` or `prior_claim_rate`, the two most predictive engineered features. These should be tracked as population statistics on a rolling 30-day window and compared against the training distribution. If the training data is no longer representative of incoming records — for example, a new risk type is being submitted that wasn't in training — the model will produce confident predictions based on the wrong features, which is worse than producing uncertain ones.

**Feedback loop**
Reviewer accept/override decisions should be captured as labels and stored alongside the record, the model's prediction, and the full `shap_values` dict. Raw accept/reject labels alone are insufficient — they don't distinguish "model was right, reviewer agreed" from "model was wrong, reviewer caught it." The useful signal is disagreements: cases where a reviewer overrides a high-confidence recommendation. These are the cases most likely to reveal systematic failure modes — a risk type that has shifted, a broker whose book has deteriorated, or a territory with changed regulatory conditions. A regular (monthly) review of high-confidence overrides, segmented by the model's top SHAP features on those records, would identify where the model is systematically wrong and what retraining is needed.

**Before going to production**
Four conditions would need to be in place:
1. **Latency:** The agent makes one model inference call and one LLM call with up to one tool round-trip. The p99 latency should be measured under realistic concurrency and a hard timeout set — a reviewer waiting indefinitely is worse than a "model unavailable" message.
2. **Cost monitoring:** Per-call token usage should be instrumented and alerted on. The main risk is prompt size growing as records get more complex — a cost spike is the first sign something has changed.
3. **Hard deferral threshold:** The model should have a documented confidence threshold below which it does not make a recommendation — it returns `requires_review: true` with no directional call. Surfacing a weakly-supported recommendation as if it were a real signal is actively harmful.
4. **Vectorstore freshness:** The retrieval index should be rebuilt whenever new historical records are added, with a checksum to detect stale embeddings. A checksum here means hashing the source documents (e.g. an MD5 of all document contents) and storing it alongside the index — on startup, if the hash of the current documents doesn't match the stored hash, the index is flagged as stale and rebuilt before serving queries. A retrieval system returning outdated comparables silently degrades without any error.
