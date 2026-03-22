# %% [markdown]
# # 04 — Prompt Engineering Portfolio
# 
# Five structured prompts demonstrating prompt engineering skills for 
# quantitative finance and market risk analysis.
# 
# Each prompt is documented with: **Task, Prompt Text, Output Format, 
# Failure Modes, and Iteration Notes.**

# %% [markdown]
# ---
# ## Prompt 1: Extract Tariff Event Dates and Affected Instruments
# 
# **Task:** Given a Federal Register PDF excerpt, extract structured 
# event data — dates, tariff rates, affected instruments.
# 
# **Output Format:** JSON array of events.
# 
# **Failure Modes:**
# - LLM may hallucinate specific tariff rates not in the source
# - Date formats may be inconsistent (July 6 vs 07/06/2018)
# - May miss instruments that are implicitly affected (e.g., FX pairs)
# 
# **Iteration Notes:**
# - v1 used open-ended "extract information" → too vague, got paragraphs
# - v2 added explicit JSON schema → structured output improved dramatically
# - v3 added "only use information present in the text" guard → reduced hallucination

# %%
prompt_1 = """
You are a trade policy analyst. Extract ALL tariff events from the text below.

For each event, return a JSON object with these exact fields:
{
  "date": "YYYY-MM-DD",
  "action": "description of the tariff action",
  "rate": "tariff rate as percentage",
  "value_affected": "dollar value of goods affected",
  "instruments_impacted": ["list of tickers/instruments likely affected"],
  "direction": "escalation | de-escalation | neutral"
}

RULES:
- Only extract information explicitly stated in the text
- If a field cannot be determined from the text, use null
- For instruments_impacted, include both direct (e.g., specific sector ETFs) 
  and indirect (e.g., EUR/USD, VIX) instruments
- Return a JSON array, even if there's only one event

TEXT:
{federal_register_excerpt}

OUTPUT (JSON array only, no commentary):
"""

print("Prompt 1 — Tariff Event Extraction")
print("=" * 50)
print(prompt_1)

# %% [markdown]
# ---
# ## Prompt 2: Risk-On / Risk-Off Classification
# 
# **Task:** Given GARCH volatility and ADTV inputs, classify a market day.
# 
# **Output Format:** JSON with classification and reasoning.
# 
# **Failure Modes:**
# - Model may over-weight one signal (e.g., high vol alone → risk-off)
# - May not account for regime-specific thresholds
# - Classification boundary is inherently subjective
# 
# **Iteration Notes:**
# - v1 just asked "is this risk-on or risk-off?" → inconsistent reasoning
# - v2 provided explicit thresholds → too rigid, missed context
# - v3 (final) provides thresholds as guidelines + asks for confidence score

# %%
prompt_2 = """
You are a quantitative risk analyst at a major investment bank. 
Classify the current market regime as RISK-ON or RISK-OFF.

INPUT DATA:
- Ticker: {ticker}
- GARCH(1,1) Annualized Volatility: {garch_vol}%
- Volatility Persistence (alpha + beta): {persistence}
- 20-day ADTV: {adtv:,} shares
- ADTV vs 90-day average: {adtv_change}%
- VIX: {vix}

CLASSIFICATION GUIDELINES (not hard rules):
- RISK-OFF signals: Vol > 30%, Persistence > 0.98, ADTV surge > +40%, VIX > 25
- RISK-ON signals: Vol < 20%, Persistence < 0.95, stable ADTV, VIX < 18
- MIXED: conflicting signals across metrics

Return a JSON object:
{
  "classification": "RISK-ON | RISK-OFF | MIXED",
  "confidence": 0.0 to 1.0,
  "primary_signal": "the most important factor driving the classification",
  "reasoning": "2-3 sentence explanation citing specific data points",
  "action_bias": "reduce_exposure | maintain | increase_exposure"
}
"""

print("\nPrompt 2 — Risk Regime Classification")
print("=" * 50)
print(prompt_2)

# %% [markdown]
# ---
# ## Prompt 3: FRED Data Release Summarization
# 
# **Task:** Summarise a FRED data release into structured JSON.
# 
# **Output Format:** JSON with indicator, value, direction, implication.
# 
# **Failure Modes:**
# - LLM may add editorializing beyond what the data shows
# - Direction field may be ambiguous for sideways moves
# - Implications may overstate certainty
# 
# **Iteration Notes:**
# - v1 asked for "summary" → got free-form paragraphs
# - v2 specified JSON schema → got structure but implications were generic
# - v3 added "cite the specific numbers" instruction → much more precise

# %%
prompt_3 = """
You are a fixed income research analyst. Parse the FRED data release below 
and return a structured JSON summary.

FRED DATA:
{fred_release_text}

Return a JSON object with this EXACT schema:
{
  "indicator": "official FRED series name",
  "series_id": "FRED series ID (e.g., DGS10, UNRATE)",
  "latest_value": numeric value,
  "previous_value": numeric value or null,
  "unit": "percent | index | thousands | millions | billions",
  "direction": "increasing | decreasing | flat",
  "magnitude": "the specific change (e.g., +0.15%)",
  "implication": "one sentence on what this means for markets, citing specific numbers",
  "affected_sectors": ["list of sectors/asset classes most impacted"],
  "release_date": "YYYY-MM-DD or null if not specified"
}

RULES:
- Use only data present in the input. Do not infer values.
- For "direction", compare to the previous period if available.
- For "implication", be specific — cite the actual numbers, don't generalize.
"""

print("\nPrompt 3 — FRED Release Summarization")
print("=" * 50)
print(prompt_3)

# %% [markdown]
# ---
# ## Prompt 4: Risk Narrative Generation
# 
# **Task:** Generate a one-paragraph risk narrative from VaR/CVaR outputs.
# 
# **Output Format:** Single paragraph, suitable for a risk report.
# 
# **Failure Modes:**
# - May produce overly technical or overly casual language
# - May not properly contextualize the numbers (is 2% VaR high or low?)
# - May inject opinions about market direction
# 
# **Iteration Notes:**
# - v1 was too academic — read like a textbook
# - v2 added "write for a portfolio manager audience" → better tone
# - v3 added "include historical context" → produced actionable narrative

# %%
prompt_4 = """
You are a senior risk analyst writing the daily risk summary for the trading desk.
Generate ONE paragraph (4-6 sentences) summarizing the risk metrics below.

RISK METRICS:
- Ticker: {ticker}
- Historical VaR (95%): {var_95}%
- CVaR (95%): {cvar_95}%
- Monte Carlo VaR (10,000 sims, 1-day): {mc_var}%
- GARCH Annualized Vol: {garch_vol}%
- Volatility Persistence: {persistence}
- Current ADTV: {adtv:,} shares
- Liquidity Flag: {liquidity_flag}

WRITING RULES:
- Audience: portfolio manager who reads 20 of these daily. Be concise.
- Lead with the most actionable finding.
- Compare VaR and MC VaR — if they diverge, explain why (fat tails vs normal assumption).
- If persistence > 0.97, flag that volatility is sticky and likely to persist.
- If liquidity flag is "low_liquidity", this is the lead item.
- Do NOT recommend trades. State risk, not direction.
- Use specific numbers from the input, not vague qualifiers.
"""

print("\nPrompt 4 — Risk Narrative Generation")
print("=" * 50)
print(prompt_4)

# %% [markdown]
# ---
# ## Prompt 5: Chain-of-Thought — Volatility Shock Risk Window
# 
# **Task:** Given volatility persistence and a tariff shock, reason through 
# the expected risk exposure window step by step.
# 
# **Output Format:** Chain-of-thought reasoning followed by a conclusion.
# 
# **Failure Modes:**
# - LLM may skip steps or jump to conclusion
# - May not properly apply the half-life formula
# - May conflate realized vol with implied vol
# 
# **Iteration Notes:**
# - v1 asked directly "what's the exposure window?" → got a number with no reasoning
# - v2 added "think step by step" → improved but steps were too general
# - v3 (final) provides explicit reasoning framework → reliable chain-of-thought

# %%
prompt_5 = """
You are a quantitative risk analyst. Think through this problem step by step.

GIVEN:
- GARCH(1,1) volatility persistence (alpha + beta) = {persistence}
- A Section 301 tariff shock just occurred, causing a {shock_magnitude}% 
  intraday move in {instrument}
- Pre-shock annualized volatility: {pre_vol}%
- Post-shock GARCH annualized volatility: {post_vol}%

QUESTION: What is the expected risk exposure window (in trading days) and why?

THINK STEP BY STEP:

Step 1 — Quantify the volatility shock:
Calculate the ratio of post-shock to pre-shock vol. How many standard deviations 
is this move relative to the historical distribution?

Step 2 — Apply the persistence parameter:
Volatility persistence of {persistence} means that a shock decays as:
  vol(t) = omega + persistence^t * (shock_vol - long_run_vol)
Calculate the half-life: t_half = -ln(2) / ln(persistence)
This tells you how many days until the shock is 50% absorbed.

Step 3 — Define the risk window:
The risk window is the number of days until volatility returns to within 
10% of its pre-shock level. Use the decay formula from Step 2.

Step 4 — Contextualise with market structure:
Consider: Does ADTV typically normalise faster or slower than volatility 
after tariff shocks? What does this mean for execution risk?

Step 5 — Conclusion:
State the expected risk window in trading days, the confidence level 
of this estimate, and the key assumption that could invalidate it.

SHOW ALL CALCULATIONS. Use actual numbers, not placeholders.
"""

print("\nPrompt 5 — Chain-of-Thought: Volatility Shock Risk Window")
print("=" * 50)
print(prompt_5)

# %% [markdown]
# ---
# ## Summary Table
# 
# | # | Task | Technique | Key Learning |
# |---|------|-----------|--------------|
# | 1 | Tariff event extraction | Structured output (JSON schema) | Explicit schemas reduce hallucination |
# | 2 | Risk-on/off classification | Guidelines + confidence scoring | Soft thresholds > hard rules |
# | 3 | FRED release summarization | Schema enforcement + citation | "Cite specific numbers" constraint works |
# | 4 | Risk narrative generation | Audience-aware writing rules | Define the reader, get better prose |
# | 5 | Volatility shock analysis | Chain-of-thought with formulas | Providing the reasoning framework > "think step by step" |