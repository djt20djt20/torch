# APPROACH.md

## Decisions made and why

### Part 1 — Loss Prediction

**1. Exploration — how the data was understood before modelling**
Before fitting anything, the notebook walks distributions and sanity checks: class balance for `is_loss_making`, scale and skew of `limit` and `premium`, plausibility of `prior_claims` and `years_trading`, missingness and how it was handled, and relationships between numerics and the target (including correlation-style views). Obvious bad rows (e.g. impossible or sentinel values) are corrected at **training** time so the model is not dominated by transcription errors; that EDA is what motivates the feature choices and the limitations section below.

**Target engineering:** The original target (`loss_ratio > 1.0`) was adjusted to `loss_ratio > 0.9818` to better reflect insurance economics. The core problem is that observed loss ratios are systematically downward-biased for policies with short exposure history: a product with a 1/10 chance of paying out 10x the premium has an expected loss ratio of 1.0 but may show 0 for several consecutive years. Since we cannot observe the true long-run loss ratio, a threshold of 1.0 on observed ratios is too high — it will miss policies that are genuinely loss-making. There is also a cost asymmetry: investigating a potentially loss-making policy is much cheaper than underwriting one that turns out to be a loss. The threshold is computed directly from the data as the exact 80th percentile of observed loss ratios (0.9818), targeting precisely a 20% positive class consistent with industry estimates of loss-making policy rates in commercial lines. This is preferable from an ML perspective too, since it better balances the classes, though that is not the primary motivation.

**2. Why this model was chosen**
Logistic regression was preferred over gradient boosted trees for this ~500-row setting: linear structure gives stable, inspectable behaviour when data are scarce, and coefficients (via SHAP on the linear model) map cleanly to “what pushed this record.” A `StandardScaler → LogisticRegression` pipeline wrapped in `CalibratedClassifierCV` (cv=5, Platt scaling) yields **calibrated** probabilities so “~80% loss-making” is meaningful for reviewers, not only a ranking. Two engineered features were added — `premium_rate` (premium / limit) and `prior_claim_rate` (prior_claims / years_trading) — because the raw fields alone miss economically meaningful exposure and claims frequency. Categoricals are target-encoded; the encoder is saved so inference matches training.

**3. Explainability — why a record is flagged**
Reviewers need attribution, not only a score. Training uses SHAP `LinearExplainer` on the scaled linear model (with plots in the notebook); at **inference**, `app/model.py` returns `top_features` and per-feature SHAP values so the API/agent can describe the main drivers in plain language. Calibration ties the headline probability to behaviour reviewers can trust; SHAP ties it to **which** fields moved the needle for **this** record.

**4. Limitations of the model**
The model only sees fields available at quote time (`risk_type`, `territory`, `industry`, `broker`, `limit`, `premium`, `prior_claims`, `years_trading` plus engineered ratios) — it cannot see policy wording, macro shocks, or management quality. With ~500 rows, hold-out metrics are noisy; rare category combinations are weakly estimated even with target encoding. SHAP reflects **association** in this dataset, not proven causation (e.g. a strong `broker` effect may be portfolio mix). Small *n* also means SHAP and calibration will move as new data arrive — the notebook states this explicitly. Finally, performance will degrade under **covariate shift** ( P(X) changes with new industries, territories, or limit/premium scales not represented in training).

**Training vs inference on data quality**
Training-time cleaning uses medians for bad training rows as described in the notebook. At **inference**, implausible raw inputs are **flagged**, not imputed: prefix `Extreme raw value —` for negatives, values outside training min–max (`raw_numeric_bounds`), or unseen categories (`raw_categorical_values` in the artifact). Encoded features outside the training percentile band are flagged separately as OOD on the **engineered** vector. Re-save `model.pkl` after updating artifact keys in the notebook.

---

### Part 2 — Assessment Agent


**1. How the agent decides what to do — reasoning vs calling everything unconditionally**
The loss model is **not** exposed as an LLM tool: `predict_loss` runs in Python **before** any LLM turn, so the model is never “skipped” or re-run by the agent. That avoids a fake decision and keeps cost more deterministic. The LLM’s only tool is **`retrieve_similar_records`**; the non-trivial choice is whether retrieval adds value and what **natural-language query** to use (better semantic matches than dumping raw fields). 

**2. Guardrails — termination conditions and fallbacks**
**Termination:** `MAX_ITERATIONS` is **3** — enough for one optional tool round plus synthesis, with a small buffer in case the LLM emits a reasoning step before its tool call. No unbounded tool chains are possible. The prompt caps retrieval at **one** call per assessment. **Fallback:** if `load_model()` fails or `predict()` throws, the agent does not return a bare error: it switches to a **retrieval-only** system prompt (no invented ML scores), still bounded by the same iteration limit, and `AssessResponse.model_available` is **false**.

**3. How uncertainty is handled — low confidence and poor retrieval**
**Low model confidence:** the system prompt encodes bands — below **0.55**, push human review; **0.55–0.85** is the “refer for senior review” zone; above **0.85** the model can be more directive. **Poor retrieval:** if cosine distance to matches is above **0.4**, the LLM is instructed to say they are not close comparables, not to cite them as strong precedent. **Data quality:** model `warnings` (OOD features, extreme raw values) and the extra user-message block for `Extreme raw value —` force the narrative to surface weak evidence and bad inputs.

**4. Whether the output is useful to a non-technical reviewer**
Recommendations are prompted in **plain English**: likely loss-making or not, confidence in reviewer terms, 2–3 drivers from **top features** (not raw SHAP dumps), optional historical context from retrieval, a clear **action** (approve / decline / refer), and no JSON or field names in the final text. Extreme inputs and “model unavailable” fallback are spelled out so a reviewer knows what to trust.

**5. Cost awareness — risks and mitigations**
Cost is dominated by **one** LLM thread per record (plus at most **one** retrieval tool round). Rough order of magnitude: ~**$0.01–0.02** per record at current Sonnet pricing. **Mitigations:** prediction is **free** of LLM tokens; **skip retrieval** when confidence is high and warnings are absent; **`max_tokens` 2048** caps synthesis; **`MAX_ITERATIONS` 3** caps multi-turn spend; retrieval is **one** call, not a loop.

---

### Part 3 — Response contract

*The README asks what a reviewer needs in the API response, what would make it useful or useless, and three specific “we care about” points. The contract is `AssessResponse` in `app/schemas.py`, returned by `POST /assess`.*

**What is in the response**
The schema has two layers: fields for the **human reviewer** and fields for **machines and auditors**.

For the reviewer: `recommendation` (the full narrative — this is where uncertainty, drivers, and next steps live in prose) and `requires_review` (a boolean a UI can use to flag the case without parsing text).

For routing and automation: `confidence` (the calibrated model probability, or `null` if the model was unavailable) and `requires_review` (true when confidence is below 0.55, any OOD or data quality warnings were raised, or the model did not run — a workflow can act on this without reading prose).

For audit: `model_warnings` (structured list of OOD / extreme-value flags from the model artifact — an auditor should not have to search the recommendation text for these), `components_used` (what actually ran), `model_available`, `truncated`, and `iteration_limit_reached`.

**Deliberately excluded**: raw SHAP dicts — these belong on a debug/audit endpoint, not the primary response. The `top_features` narrative is already in the recommendation text via the agent prompt.

**1. Communicating uncertainty so a reviewer can act on it**
Two signals carry this: `confidence` (the calibrated probability — a reviewer can see "23%" and know the model is close to a coin-flip) and the `recommendation` narrative (which the agent translates into plain English). `model_available: false` is the hard guardrail — when false, nothing in the narrative should be read as "the ML score said X."

**2. When to seek a second opinion**
`requires_review` is the machine-readable signal — it is true when confidence is below 0.55, any OOD or data quality warnings are present, or the model did not run. A workflow can route on this without parsing prose. The `recommendation` text explains *why* review is needed for the human reading it. `model_warnings` exposes the structured flags so a reviewer does not have to find them buried in the narrative.

**3. Usefulness to an auditor six months later**
An auditor needs: `record_id` (which record), `confidence` (what the model said at the time), `model_warnings` (what data quality issues were present), `components_used` (what actually ran — model only, model + retrieval, or retrieval-only fallback), `model_available` (whether ML was in play), and `truncated` / `iteration_limit_reached` (whether any system limit was silently hit). Together with stored request/response logs these answer "what was recommended, how confident was the model, and under what technical conditions?"\.
---

## Evaluation approach

`score_recommendation()` uses an LLM-as-judge prompt that extracts the implied direction (loss-making / not) and expressed certainty from the free-text recommendation, then compares to `is_loss_making`. Three dimensions are scored: outcome alignment (did it point the right way?), calibration (was expressed certainty proportionate to how borderline the case was?), and safe deferral (did it escalate when it should have?). The `case_type` field in the eval set separates clear and borderline cases — a wrong answer on a genuinely borderline record is less serious than a confidently wrong answer on a clear one, and the scorer reflects this.

The eval is designed to catch the failure mode the agent is most likely to hide behind: confident-sounding language on a wrong call. A recommendation that says "this record is unlikely to be loss-making" when `is_loss_making` is true fails on outcome alignment regardless of how well-written it is.

The eval run also tracks two silent failure modes — `truncated` (LLM hit `max_tokens` mid-response) and `iteration_limit_reached` (loop exhausted `MAX_ITERATIONS` without reaching `end_turn`) — and reports them per-record and in the summary. On the current 20-record eval set, both flags were **0/20**, meaning the agent completed naturally on every record.

**Current results (20-record eval set):** 17/20 correct (85%). Loss-making records: 8/9. Non-loss-making records: 9/11. Average latency: 27.2s/record (dominated by one LLM call + one retrieval round-trip per record).

---

## AI tools and human judgment

I used an AI-assisted editor throughout — primarily for speed on scaffolding, boilerplate, and iterating on `APPROACH.md` prose. The decisions below are the ones I made explicitly, and where I pushed back on suggestions.

**What I changed or rejected:**

- **GBM over logistic regression** — the AI defaulted to `GradientBoostingClassifier` as the primary candidate. I kept it as a comparison candidate but chose calibrated logistic regression as the final model: at ~500 rows, the interpretability and stability advantages outweigh the performance gain from a tree ensemble, and SHAP on a linear model is far more trustworthy. The AI could not weigh that trade-off — it has no view on how a reviewer will actually use the output.
- **More complex agent designs** — suggestions included multi-step planning loops, tool-calling chains longer than one retrieval round, and a separate "clarification" tool. I rejected all of these: the brief is a single-turn underwriting assessment, not an open-ended agent. Bounded complexity is a feature, not a limitation.
- **Imputing extreme raw values** — an early suggestion was to impute implausible inputs (e.g. negative premiums) with training medians at inference time, mirroring the training-time cleaning. I rejected this: imputing silently at inference hides bad data from the reviewer. The right behaviour is to flag it and let a human decide. The model still runs — but the reviewer is told the input was suspect.
- **Target threshold** — the AI suggested using `loss_ratio > 1.0` as the obvious threshold. I overrode this based on domain reasoning: observed loss ratios are downward-biased for short-exposure policies, so a threshold of 1.0 systematically misses genuinely loss-making records. The 80th percentile (0.9818) was a judgment call informed by industry priors on loss-making policy rates, not a data-driven optimisation.

**What the AI could not decide:**
The AI has no view on what an underwriter actually needs, what "useful" means in a claims context, or where the cost asymmetry between false negatives and false positives sits in this business. Those judgments determined the target threshold, the deferral bands, the response contract, and the eval scoring dimensions. The AI helped execute; it did not set the brief.

---

## What I would do differently with more time

- Train on a larger dataset — with 504 rows and an 80/20 class split (by design, targeting the industry prior of ~20% loss-making policies), held-out metrics have wide confidence intervals. Bootstrapped confidence intervals on AUC would make the model's limitations more legible.
- Add more context to the agent, particularly about the categoricals, and the interactions between them. From my own experience in ML, I've found that categorical variables representing different domains tend to both contain a great deal explanatory and predictive power, but also tend to hide a lot of information that could be independent features. For example, a geographic region could amalgamate laws, cultural norms, and economic conditions that are all relevant to the outcome but not captured by a single categorical label. If I could, I'd try and tease out these factors, as they offer additional explanatory power and could help the agent make more informed decisions.
- Replace target encoding with a proper Bayesian average to reduce leakage on rare categories (brokers or industries with few training examples). A Bayesian average shrinks the encoded value for a rare category towards the global mean, rather than trusting a noisy estimate from a handful of records — e.g. a broker that appears twice in training shouldn't get a very high or very low encoding just because both their records happened to be losses. I didn't do it in this one because the categories were all well represented in the training data.
- Instrument token usage per agent call and alert when calls exceed a budget threshold — the current implementation has no visibility into per-call cost.
- Run the eval harness as part of CI on every model or prompt change, not just as a one-off.

## Production readiness

**How I would know the model is degrading**
The challenge is that ground truth (did the policy actually make a loss?) arrives with a long lag — claims can emerge months or years after underwriting, so you cannot wait for outcome labels to detect degradation in real time. Monitoring therefore needs to work at two separate horizons:

**Immediate signals — no ground truth needed:**
- **Input distribution drift:** track the distribution of incoming features (`premium_rate`, `prior_claim_rate`, `limit`, `territory`, `industry`) on a rolling window and compare against the training distribution using a simple population stability index (PSI) or KS test. If the feature distribution shifts, the model is being asked to score records that look nothing like its training data — it will produce confident but meaningless predictions. This is detectable immediately.
- **Prediction distribution drift:** if the share of records scoring above 0.85 or below 0.15 drops sharply — i.e. the model starts pushing most predictions towards 0.5 — that is a sign of covariate shift even without any outcome labels.
- **OOD flag rate:** track the rate at which `model.warnings` fires (unseen categories, out-of-distribution features). A rising flag rate is an early warning that incoming records are moving outside the training manifold.
- **Reviewer override rate:** if reviewers are systematically overriding high-confidence recommendations, that is a behavioural signal of degradation even before outcome labels arrive.

**Lagged signals — require outcome data, but worth capturing when available:**
- Rising loss ratio on records the model predicted as non-loss-making (false negatives) is the business-critical outcome signal, but it may only be measurable quarterly or annually.
- When outcome labels do arrive, segment them by the model's top SHAP features at prediction time — this identifies *which* risk types or territories have drifted, not just that drift occurred.

The practical implication is that input monitoring and override tracking should be in place from day one, and outcome-based evaluation should be scheduled as a periodic review (e.g. quarterly) as claims data accumulates — not treated as a continuous signal.

**Feedback loop**
Reviewer accept/override decisions should be captured as labels and stored alongside the record, the model's prediction, and the full `shap_values` dict. Raw accept/reject labels alone are insufficient — they don't distinguish "model was right, reviewer agreed" from "model was wrong, reviewer caught it." The useful signal is disagreements: cases where a reviewer overrides a high-confidence recommendation. These are the cases most likely to reveal systematic failure modes — a risk type that has shifted, a broker whose book has deteriorated, or a territory with changed regulatory conditions. A regular (monthly) review of high-confidence overrides, segmented by the model's top SHAP features on those records, would identify where the model is systematically wrong and what retraining is needed.

**Before going to production**
Four conditions would need to be in place:
1. **Latency:** The agent makes one model inference call and one LLM call with up to one tool round-trip. The p99 latency should be measured under realistic concurrency and a hard timeout set — a reviewer waiting indefinitely is worse than a "model unavailable" message.
2. **Cost monitoring:** Per-call token usage should be instrumented and alerted on. The main risk is prompt size growing as records get more complex — a cost spike is the first sign something has changed.
3. **Hard deferral threshold:** The model should have a documented confidence threshold below which it does not make a recommendation — it returns `requires_review: true` with no directional call. Surfacing a weakly-supported recommendation as if it were a real signal is actively harmful.
4. **Vectorstore freshness:** The model and the retrieval index should be updated together as a single deployment step — retraining on new records and rebuilding the index are not independent operations. The risk to guard against is a partial update: a new model deployed without a corresponding index rebuild (or vice versa), leaving them out of sync. A retrieval system returning outdated comparables silently degrades without any error.
