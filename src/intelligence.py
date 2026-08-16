"""
Loop Closer — Intelligence / Extraction Node
============================================

Ingests a raw chat string and uses the Groq API (free tier) to determine
whether the message contains a developer commitment, extract the task, and
resolve the deadline into an absolute timestamp.

Provider: Groq  (https://console.groq.com — free, no credit card needed)
SDK:      openai Python SDK pointed at Groq's OpenAI-compatible base URL.
          No extra package required — the same `openai` library is reused.

Groq vs OpenAI difference
--------------------------
OpenAI supports `.parse(response_format=PydanticModel)` (Structured Outputs).
Groq does NOT — it supports JSON mode (`response_format={"type":"json_object"}`)
which instructs the model to emit valid JSON, but does not enforce a schema.
We handle this by:
  1. Describing the exact schema in the system prompt (few-shot + field rules).
  2. Calling `.create()` with `response_format={"type": "json_object"}`.
  3. Parsing the JSON string manually and validating it with Pydantic.
  4. Falling back to the safe dict on any parse / validation error.

Public API
----------
    extract_commitment(text: str) -> dict

    Returns:
        {
            "is_commitment":    bool,
            "task_description": str | None,
            "deadline":         str | None,   # "YYYY-MM-DD HH:MM:SS" or None
        }

    NEVER raises — any error yields the safe fallback dict so the main
    event loop cannot crash.

Environment variables (add to .env)
------------------------------------
    GROQ_API_KEY   — from https://console.groq.com/keys  (free, instant)
    GROQ_MODEL     — default: llama-3.3-70b-versatile
                     alternatives: llama-3.1-8b-instant  (faster, lighter)
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Groq's OpenAI-compatible endpoint — the ONLY URL difference vs OpenAI
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Default model: best free Groq model for structured JSON classification.
# llama-3.3-70b-versatile is more reliable at strict JSON than the 8B model.
# Switch to llama-3.1-8b-instant via GROQ_MODEL env var for max speed.
_DEFAULT_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Data Contract — Pydantic schema used for validation after JSON parse
# ---------------------------------------------------------------------------

class CommitmentResult(BaseModel):
    """
    Validated output schema for a single commitment-extraction call.

    Fields
    ------
    is_commitment : bool
    task_description : str | None
    deadline : str | None   — "YYYY-MM-DD HH:MM:SS" or None
    """
    is_commitment: bool
    task_description: Optional[str]
    deadline: Optional[str]


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a Commitment Extraction Engine for a developer-accountability system
called Loop Closer. Analyse the user's chat message and return ONLY a JSON
object — no markdown, no explanation, nothing outside the JSON.

════════════════════════════════════════════════
TEMPORAL REFERENCE FRAME
════════════════════════════════════════════════
Current timestamp : {now}
Default deadline  : {default_deadline}
  ↑ Use this when is_commitment=true but NO deadline is stated.

Resolve relative expressions against the current timestamp above:
  "tonight" / "evening" → 20:00:00 same day (or next day if already past)
  "morning"             → 09:00:00
  "afternoon"           → 15:00:00
  "EOD" / "end of day"  → 17:00:00
  "noon"                → 12:00:00
  "midnight"            → 23:59:59
  "tomorrow"            → next calendar day, 09:00:00
  "next Friday"         → the coming Friday, 09:00:00
  "in 2 hours"          → current time + 2 hours

════════════════════════════════════════════════
CLASSIFICATION RULES
════════════════════════════════════════════════

is_commitment = true ONLY when ALL hold:
  1. First-person speaker (I / I'll / I will / I'm going to / I'll make sure)
  2. Concrete, technical action (not vague or social)
  3. Future-oriented tense — a promise not yet fulfilled
  4. Definitive phrasing — NOT "might", "probably", "maybe", "should"

is_commitment = false for:
  • Questions     — "Will you fix it?" / "Can someone deploy this?"
  • Historical    — "I fixed that yesterday." / "We shipped last night."
  • Speculation   — "I might look into that." / "Someone should fix this."
  • Casual chat   — "Great standup!" / "Sounds good 👍"
  • Bot commands  — "!register github:x telegram:@y"

════════════════════════════════════════════════
FIELD RULES
════════════════════════════════════════════════
task_description
  • true  → imperative summary, strip greetings/filler/@-mentions/channel refs
  • false → null

deadline
  • true + time stated  → resolve to "YYYY-MM-DD HH:MM:SS"
  • true + no time      → use default deadline: {default_deadline}
  • false               → null

════════════════════════════════════════════════
FEW-SHOT EXAMPLES  (input → exact JSON output)
════════════════════════════════════════════════

Input : "I'll push the hotfix for the memory leak tonight."
Output: {{"is_commitment": true, "task_description": "Push the hotfix for the memory leak", "deadline": "{tonight}"}}

Input : "Will someone fix the DB indexing issue?"
Output: {{"is_commitment": false, "task_description": null, "deadline": null}}

Input : "I fixed the login bug yesterday, all tests pass now."
Output: {{"is_commitment": false, "task_description": null, "deadline": null}}

Input : "I'll probably get around to reviewing that PR at some point."
Output: {{"is_commitment": false, "task_description": null, "deadline": null}}

Input : "I'll write the unit tests for the payments module."
Output: {{"is_commitment": true, "task_description": "Write unit tests for the payments module", "deadline": "{default_deadline}"}}

Input : "I will deploy the staging build by Friday 3pm."
Output: {{"is_commitment": true, "task_description": "Deploy the staging build", "deadline": "{next_friday_3pm}"}}

════════════════════════════════════════════════
OUTPUT CONTRACT — MANDATORY
════════════════════════════════════════════════
Return ONLY a JSON object with exactly these three keys:
  "is_commitment"    → boolean
  "task_description" → string or null
  "deadline"         → "YYYY-MM-DD HH:MM:SS" string or null
No extra keys. No markdown. No text outside the JSON object.
"""


def _build_system_prompt() -> str:
    """Render the system prompt with live-computed temporal values."""
    now = datetime.now() + timedelta(hours=5, minutes=30)
    default_deadline = now + timedelta(hours=24)

    tonight = now.replace(hour=20, minute=0, second=0, microsecond=0)
    if tonight <= now:
        tonight += timedelta(days=1)

    days_ahead = (4 - now.weekday()) % 7  # 4 == Friday
    if days_ahead == 0:
        days_ahead = 7
    next_friday_3pm = (now + timedelta(days=days_ahead)).replace(
        hour=15, minute=0, second=0, microsecond=0
    )

    fmt = "%Y-%m-%d %H:%M:%S"
    return _SYSTEM_PROMPT_TEMPLATE.format(
        now=now.strftime(fmt),
        default_deadline=default_deadline.strftime(fmt),
        tonight=tonight.strftime(fmt),
        next_friday_3pm=next_friday_3pm.strftime(fmt),
    )


# ---------------------------------------------------------------------------
# Safe fallback
# ---------------------------------------------------------------------------

_FALLBACK: dict = {
    "is_commitment": False,
    "task_description": None,
    "deadline": None,
}


# ---------------------------------------------------------------------------
# Public extraction function
# ---------------------------------------------------------------------------

def extract_commitment(text: str) -> dict:
    """
    Analyse *text* for developer commitments via Groq (free tier).

    Uses the openai SDK pointed at Groq's base URL — zero extra dependencies.
    JSON mode + Pydantic validation replaces OpenAI's native Structured Outputs.

    Parameters
    ----------
    text : str
        Raw chat message (may include @-mentions, emoji, bot commands, etc.)

    Returns
    -------
    dict  — always safe, never raises.
        {"is_commitment": bool, "task_description": str|None, "deadline": str|None}

    Environment variables
    ---------------------
    GROQ_API_KEY  (required) — https://console.groq.com/keys
    GROQ_MODEL    (optional) — default: llama-3.3-70b-versatile
    """
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[INTELLIGENCE] ERROR: GROQ_API_KEY is not set.", flush=True)
        return _FALLBACK

    model = os.getenv("GROQ_MODEL", _DEFAULT_MODEL)

    try:
        # Same openai SDK — just a different base_url and key
        client = OpenAI(
            api_key=api_key,
            base_url=_GROQ_BASE_URL,
        )

        system_prompt = _build_system_prompt()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": text},
            ],
            # JSON mode: Groq guarantees the response is valid JSON.
            # Schema adherence is enforced by the prompt + Pydantic below.
            response_format={"type": "json_object"},
            temperature=0,    # deterministic — classification, not creativity
            max_tokens=256,   # schema is small; cap spend
            timeout=10.0,     # CRITICAL: Prevent infinite hang if API stalls
        )

        raw_json: str = response.choices[0].message.content

        # Parse the JSON string → dict → Pydantic model (schema validation)
        parsed_dict = json.loads(raw_json)
        result = CommitmentResult.model_validate(parsed_dict)

        # Semantic guard: non-commitments must have null fields
        if not result.is_commitment:
            return _FALLBACK

        return {
            "is_commitment":    result.is_commitment,
            "task_description": result.task_description,
            "deadline":         result.deadline,
        }

    except RateLimitError as exc:
        print(f"[INTELLIGENCE] Groq rate limit hit: {exc}", flush=True)
    except APIConnectionError as exc:
        print(f"[INTELLIGENCE] Groq connection error: {exc}", flush=True)
    except APIError as exc:
        print(
            f"[INTELLIGENCE] Groq API error "
            f"(status={getattr(exc, 'status_code', 'N/A')}): {exc}",
            flush=True,
        )
    except json.JSONDecodeError as exc:
        print(f"[INTELLIGENCE] JSON parse error: {exc}", flush=True)
    except ValidationError as exc:
        print(f"[INTELLIGENCE] Pydantic validation error: {exc}", flush=True)
    except Exception as exc:
        print(f"[INTELLIGENCE] Unexpected error: {exc}", flush=True)

    return _FALLBACK


# ---------------------------------------------------------------------------
# Smoke test — python src/intelligence.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _TEST_CASES = [
        ("commitment + relative deadline",
         "I'll fix the auth token expiry bug tonight."),
        ("commitment, no deadline -> +24h default",
         "I will write integration tests for the billing module."),
        ("commitment with explicit day + time",
         "I'll deploy the staging build by Thursday 5pm."),
        ("question -> False",
         "Will someone fix the DB indexing issue?"),
        ("historical -> False",
         "I deployed the hotfix yesterday, all good."),
        ("speculative -> False",
         "I'll probably look into that at some point."),
        ("casual chat -> False",
         "Hey everyone, great standup today!"),
        ("bot command -> False",
         "!register github:veerandrakumar telegram:@veer123"),
        ("definitive before EOD -> True",
         "I'll get the PR reviewed and merged before EOD."),
    ]

    print("=" * 72)
    print("Loop Closer — Intelligence Smoke Test  (Groq / llama-3.3-70b)")
    print(f"  Reference time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    for label, msg in _TEST_CASES:
        result = extract_commitment(msg)
        marker = "OK" if (
            (result["is_commitment"] and "False" not in label) or
            (not result["is_commitment"] and "False" in label) or
            ("commitment" in label and result["is_commitment"])
        ) else "!!"
        print(f"\n[{marker}] {label}")
        print(f"  Input  : {msg!r}")
        print(f"  Output : {result}")

    print("\n" + "=" * 72)

