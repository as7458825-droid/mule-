"""explain.py — GenAI explanation layer (plain-English alerts).

Builds a structured prompt from a flagged account's risk score, top
features, and rule hits. The prompt can be passed to any LLM API
(Claude, GPT, Gemini) for a 100-150 word officer-ready explanation.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

LOG = logging.getLogger("muleshield.explain")


def build_prompt(
    account_id: str,
    risk_score: float,
    top_features: list[tuple[str, float]],
    rule_hits: list[str],
    gov_ticket_match: bool = False,
) -> str:
    """Build a structured GenAI prompt for a flagged account.

    Parameters
    ----------
    account_id : str
        The account identifier (e.g., "AC-2847").
    risk_score : float
        0-100 risk score from the ML model.
    top_features : list of (feature, importance)
        Top-5 SHAP-like features driving the risk.
    rule_hits : list of str
        Names of rules that fired (e.g., "R1_dormant_activation").
    gov_ticket_match : bool
        Whether the account appears in CFCFRMS / Sanchar Saathi.

    Returns
    -------
    str
        A prompt ready to send to an LLM API.
    """
    if not 0 <= risk_score <= 100:
        raise ValueError("risk_score must be in [0, 100]")

    level = (
        "CRITICAL" if risk_score >= 76 else
        "HIGH" if risk_score >= 51 else
        "MEDIUM" if risk_score >= 26 else
        "LOW"
    )

    feature_lines = "\n".join(
        f"  - {name}: importance = {imp:.3f}" for name, imp in top_features[:5]
    )
    rule_lines = "\n".join(f"  - {r}" for r in rule_hits) if rule_hits else "  - (none)"

    prompt = f"""You are an explainable-AI assistant for an Indian public-sector bank fraud team.
Generate a 100-150 word plain-English risk explanation for the following flagged account.
The audience is a non-technical bank officer who will decide whether to FREEZE the account.

Account: {account_id}
Risk Score: {risk_score:.0f}/100 ({level})
Matched rule patterns ({len(rule_hits)}):
{rule_lines}

Top contributing features (by SHAP importance):
{feature_lines}

Government fraud-ticket match (CFCFRMS / Sanchar Saathi): {"YES" if gov_ticket_match else "NO"}

Output format (strict):
- "Reasons:" — numbered list, max 5 items, each one short sentence.
- "Action:" — single sentence: FREEZE + KYC re-verify / RESTRICT + manual review / MONITOR / NO ACTION.

Do not invent any facts. Only use the data above. Be concise. No preamble.
"""
    LOG.debug("Built prompt for %s (score=%.0f, %s)", account_id, risk_score, level)
    return prompt


def generate_explanation(
    account_id: str,
    risk_score: float,
    top_features: list[tuple[str, float]],
    rule_hits: list[str],
    gov_ticket_match: bool = False,
    api: str = "claude",
) -> str:
    """Generate an explanation. Calls the LLM if API key is set;
    otherwise returns the prompt itself (for offline demo).

    Set environment variables:
      - ANTHROPIC_API_KEY  (for claude)
      - OPENAI_API_KEY     (for gpt)
      - GOOGLE_API_KEY     (for gemini)
    """
    prompt = build_prompt(account_id, risk_score, top_features, rule_hits, gov_ticket_match)

    if api == "claude" and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            LOG.warning("Claude call failed (%s); returning prompt", e)
            return prompt

    if api == "gpt" and os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            LOG.warning("GPT call failed (%s); returning prompt", e)
            return prompt

    # Offline mode: return prompt so user can paste into their UI
    return prompt


# Sample of the canonical MuleShield example shown in the solution PDF
SAMPLE_ACCOUNT_ID = "AC-2847"
SAMPLE_RISK_SCORE = 87.0
SAMPLE_FEATURES = [
    ("F376_dormancy_gap", 0.31),
    ("F259_rapid_movement", 0.22),
    ("F198_night_ratio", 0.18),
    ("F431_new_beneficiary", 0.12),
    ("F527_ch_UPI", 0.09),
]
SAMPLE_RULES = ["R1_dormant_activation", "R2_smurfing", "R3_odd_hour", "R6_rapid_movement"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = generate_explanation(
        SAMPLE_ACCOUNT_ID, SAMPLE_RISK_SCORE, SAMPLE_FEATURES, SAMPLE_RULES, gov_ticket_match=True
    )
    print("=" * 70)
    print("Account:", SAMPLE_ACCOUNT_ID, "| Risk:", SAMPLE_RISK_SCORE)
    print("=" * 70)
    print(out)
