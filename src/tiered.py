"""Bonus: tiered regex -> LLM cost optimizer.

The common case in a finance app is a clean "<item> <amount>" message, optionally
repeated ("กาแฟ 45 ขนมปัง 30"). A conservative regex handles those for $0 / ~0ms.
Anything the regex is not confident about (Thai numerals, number-words, times,
injection, leftover unparsed text, no match) is DEFERRED to the LLM so accuracy
never drops below the pure-LLM system.

Public API:
    extract_tiered_verbose(text, model) -> ExtractResult   (route = "regex" | "llm" | "short_circuit")
"""

from __future__ import annotations

import re

from src.ner import ExtractResult, _normalize_amount, extract_verbose

THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
CONNECTORS = ["แล้วก็", "และ", "กับ", "ก็", "แล้ว"]
LEAD_VERBS = ["ซื้อ", "จ่าย", "เติม"]            # stripped from detail; ค่า is preserved

# If any of these appear, the regex is NOT confident -> defer to the LLM.
DEFER_KEYWORDS = [
    "โมง", "นาฬิกา",                              # times look like amounts
    "กิน",                                        # "กิน ... ไป" circumfix slang
    "ignore", "instruction", "previous", "system", "reveal", "prompt",
    "ลืมคำสั่ง", "คำสั่งก่อน",                      # injection (Thai)
]

_PAIR = re.compile(r"([^\d]+?)\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(บาท|บ|฿|k|K)?")


def _clean_detail(d: str) -> str:
    d = d.strip(" ,\n\t")
    # strip leading connectors (possibly several)
    changed = True
    while changed:
        changed = False
        for c in CONNECTORS:
            if d.startswith(c):
                d = d[len(c):].strip()
                changed = True
    # strip a single leading action verb, but never strip "ค่า"
    if not d.startswith("ค่า"):
        for v in LEAD_VERBS:
            if d.startswith(v):
                d = d[len(v):].strip()
                break
    # strip a ฿ that leaked into the detail as an amount prefix (e.g. "เสื้อ ฿")
    d = d.replace("฿", "")
    return d.strip()


def regex_extract(text: str) -> dict | None:
    """Return {"transactions": [...]} if confident, else None (-> defer to LLM)."""
    if not text or not text.strip():
        return {"transactions": []}
    if any(ch in text for ch in THAI_DIGITS):
        return None
    low = text.casefold()
    if any(k.casefold() in low for k in DEFER_KEYWORDS):
        return None

    matches = list(_PAIR.finditer(text))
    if not matches:
        return None  # no "<text> <number>" structure -> let the LLM decide

    covered = [False] * len(text)
    txns = []
    for m in matches:
        for i in range(m.start(), m.end()):
            covered[i] = True
        detail = _clean_detail(m.group(1))
        if not detail:
            return None  # amount with no item -> ambiguous, defer
        raw = m.group(2).replace(",", "")
        amount = float(raw)
        if m.group(3) in ("k", "K"):
            amount *= 1000
        txns.append({"amount": _normalize_amount(amount), "detail": detail})

    # everything not consumed by a pair must be whitespace / connectors / punctuation
    leftover = "".join(c for i, c in enumerate(text) if not covered[i])
    leftover = re.sub(r"[\s,]+", "", leftover)
    for c in CONNECTORS:
        leftover = leftover.replace(c, "")
    if leftover:
        return None  # unexplained text -> defer to the LLM to be safe

    return {"transactions": txns}


def extract_tiered_verbose(text, model: str) -> ExtractResult:
    """Regex fast-path first; fall back to the LLM. Never raises."""
    try:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        if not text.strip():
            return ExtractResult(short_circuited=True, route="short_circuit")
        fast = regex_extract(text)
        if fast is not None:
            return ExtractResult(transactions=fast["transactions"], route="regex")
        res = extract_verbose(text, model)
        res.route = "llm"
        return res
    except Exception as e:
        return ExtractResult(error=f"unexpected:{type(e).__name__}", route="regex")


def extract_tiered(text, model: str) -> dict:
    return extract_tiered_verbose(text, model).output
