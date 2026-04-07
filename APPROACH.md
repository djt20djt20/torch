# APPROACH.md

## Decisions made and why

**Model choice:** Calibrated logistic regression over GBM. At ~500 rows the interpretability and stability advantages outweigh any ensemble performance gain, and SHAP on a linear model is far more trustworthy — coefficients map directly to log-odds, so a reviewer can follow the reasoning. A `StandardScaler → LogisticRegression` pipeline in `CalibratedClassifierCV` (cv=5, Platt scaling) gives calibrated probabilities, not just rankings. Two engineered features — `premium_rate` (premium / limit) and `prior_claim_rate` (prior_claims / years_trading) — capture exposure and claims frequency that the raw fields miss. Categoricals are target-encoded; the encoder is saved so inference matches training exactly.

**Target engineering:** The given target (`loss_ratio > 1.0`) was adjusted to the 80th percentile of observed loss ratios (0.9818). Observed loss ratios are systematically downward-biased for short-exposure policies — a policy with expected loss ratio 1.0 will show 0 for several years before a claim arrives. A hard threshold of 1.0 misses genuinely loss-making records. The 80th percentile targets a 20% positive class consistent with industry estimates; class balance is a secondary benefit, not the motivation.

**Training vs inference on data quality:** Bad rows (impossible values etc) are corrected with medians at training time. At inference, implausible inputs are flagged, not imputed — prefix `Extreme raw value —` for negatives, out-of-range values, or unseen categories. Imputing silently hides bad data from the reviewer; the right behaviour is to flag and let a human decide.

**Agent design:** `predict_loss` runs in Python before any LLM turn — it is never an LLM tool call. This removes any possibility the model gets skipped or called twice, and keeps cost deterministic. The LLM's only tool is `retrieve_similar_records`; the non-trivial choice is whether retrieval adds value and what query to use. `MAX_ITERATIONS = 3` bounds the loop. If the model artifact is missing or fails, the agent switches to a retrieval-only system prompt and sets `model_available: false` — it never invents scores.

**Uncertainty handling:** The system prompt encodes three explicit bands: below 0.55 — weak evidence, push human review; 0.55–0.85 — refer for senior review; above 0.85 — can be more directive if no warnings. `requires_review` in the API response covers only the hard cutoff (< 0.55, warnings present, or model unavailable); the 0.55–0.85 soft referral is enforced via the recommendation narrative, since flagging that band in the field would trigger on the majority of records. Cosine distance > 0.4 on retrieval results is treated as additional uncertainty.

**Response contract (`AssessResponse`):** Two layers — reviewer fields (`recommendation`, `requires_review`) and audit fields (`confidence`, `model_warnings`, `components_used`, `model_available`, `truncated`, `iteration_limit_reached`). Raw SHAP dicts are excluded from the primary response; they belong on a debug endpoint. An auditor six months later needs to reconstruct: what was recommended, what the model said, what data quality issues were present, and whether any system limit was silently hit — all of that is in the schema.

**Eval:** LLM-as-judge scoring three dimensions — outcome alignment, calibration (confident wrong answers on borderline cases score 0), and safe deferral. Designed to catch the failure mode the agent most hides behind: confident-sounding language on a wrong call. **Current results: 18/20 (90%). Loss-making: 9/9. Non-loss-making: 9/11. Truncated: 0/20. Iteration limit: 0/20.**

---

## What I would do differently with more time

- Bootstrapped confidence intervals on AUC — with 504 rows, held-out metrics carry wide uncertainty that the point estimates hide.
- Decompose categorical features further. A `territory` label amalgamates laws, economic conditions, and cultural norms that are independent drivers. Teasing these out would add explanatory power the current encoding discards.
- Instrument per-call token usage and alert on budget breaches — the current implementation has no visibility into cost per record.
- Run the eval as part of CI on every prompt or model change.

---

## AI tools and human judgment

I used Cursor throughout for scaffolding and boilerplate. The decisions I made that the AI could not:

- **LR over GBM** — the AI defaulted to GBM. I chose LR because at this scale interpretability and stable SHAP dominate; the AI has no view on how a reviewer will use the output.
- **Agent simplicity** — suggestions included multi-step planning loops and a clarification tool. I rejected them: the brief is a single-turn assessment. Bounded complexity is a feature.
- **Flag, don't impute at inference** — the AI suggested mirroring training-time imputation. I rejected it: silently fixing bad inputs hides them from the reviewer.
- **Target threshold** — the AI suggested `loss_ratio > 1.0`. I overrode it: observed ratios are downward-biased, and 1.0 systematically misses genuinely loss-making records.

The judgments that determined the target, the deferral bands, the response contract, and the eval dimensions are all domain decisions — the AI helped execute, it did not set the brief.

---

## Production readiness

**Degradation signals (immediate, no ground truth needed):** input distribution drift test on a rolling window; prediction distribution drift (scores clustering near 0.5); rising OOD flag rate from `model.warnings`; reviewer override rate on high-confidence calls.

**Degradation signals (lagged):** rising loss ratio on records predicted non-loss-making; when labels arrive, segment by SHAP top features to identify *which* risk types or territories have drifted.

**Feedback loop:** store reviewer accept/override decisions alongside the record, the model prediction, and the full SHAP dict. Raw labels are insufficient — the signal is *disagreements* on high-confidence calls. Monthly review of high-confidence overrides segmented by top SHAP features identifies where the model is systematically wrong and what retraining is needed.

**Before production:**
1. **Latency:** measure p99 under realistic concurrency and set a hard timeout — a reviewer waiting indefinitely is worse than a "model unavailable" fallback.
2. **Cost monitoring:** instrument per-call token usage; a prompt size spike is the first sign something changed.
3. **Vectorstore freshness:** model and retrieval index must be updated as a single deployment step. A partial update — new model without a rebuilt index — silently degrades retrieval with no error.
4. **Deferral condition:** the system should stop making recommendations and escalate to human-only review when `model_available` is false, when `requires_review` is true and retrieval found no close matches, or when the OOD flag rate on incoming records exceeds a threshold set during initial deployment.
