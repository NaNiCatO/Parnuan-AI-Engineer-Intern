"""Eval harness for the Text -> Transaction NER system.

One command:
    uv run python src/eval.py                          # all 3 default models
    uv run python src/eval.py --model google/gemini-2.5-flash-lite
    uv run python src/eval.py --dataset data/dataset.jsonl --limit 10

Reports, per model and per bucket:
  - amount  P / R / F1
  - detail  P / R / F1   (a detail is correct only if attached to a correct amount)
  - exact-match rate (full transactions array, order-independent)
  - transaction-count accuracy
  - latency p50 / p95 (over real API calls)
  - cost per 1k messages (live OpenRouter pricing x measured tokens)
  - a failure taxonomy with concrete examples

Writes reports/eval_<model>.json and prints a markdown report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

# Make `src` importable whether run as `python src/eval.py` or `python -m src.eval`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ner import extract_verbose

# Live OpenRouter pricing pulled from GET /api/v1/models (Step 0), $ per token.
PRICING = {
    "google/gemini-2.5-flash-lite": {"in": 0.0000001, "out": 0.0000004},
    "openai/gpt-5-mini":            {"in": 0.00000025, "out": 0.000002},
    "anthropic/claude-sonnet-4.6":  {"in": 0.000003,  "out": 0.000015},
}
DEFAULT_MODELS = list(PRICING)


# --------------------------------------------------------------------------- #
# Normalization & matching primitives
# --------------------------------------------------------------------------- #
def norm_text(s: str) -> str:
    """Normalized detail comparison: trim + casefold + collapse internal whitespace."""
    return re.sub(r"\s+", " ", str(s).strip()).casefold()


def norm_amount(a) -> float:
    return float(a)


@dataclass
class Aligned:
    matched: list[tuple[dict, dict]]   # (gold, pred) paired on equal amount
    unmatched_gold: list[dict]
    unmatched_pred: list[dict]


def align(gold: list[dict], pred: list[dict]) -> Aligned:
    """Greedy one-to-one pairing of gold<->pred on exact amount equality."""
    pred_pool = list(pred)
    matched, unmatched_gold = [], []
    for g in gold:
        hit = next((p for p in pred_pool if norm_amount(p["amount"]) == norm_amount(g["amount"])), None)
        if hit is not None:
            matched.append((g, hit))
            pred_pool.remove(hit)
        else:
            unmatched_gold.append(g)
    return Aligned(matched=matched, unmatched_gold=unmatched_gold, unmatched_pred=pred_pool)


# --------------------------------------------------------------------------- #
# Per-message scoring
# --------------------------------------------------------------------------- #
@dataclass
class RowScore:
    amount_tp: int = 0
    amount_fp: int = 0
    amount_fn: int = 0
    detail_tp: int = 0           # matched amount AND detail equal
    detail_fp: int = 0           # preds that are not detail-correct
    detail_fn: int = 0           # golds not captured detail-correct
    exact: bool = False
    count_ok: bool = False
    errors: list = None          # failure-taxonomy tags for this row

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def score_row(gold: list[dict], pred: list[dict], err: str | None) -> RowScore:
    a = align(gold, pred)
    rs = RowScore()

    # ---- amount field metrics ----
    rs.amount_tp = len(a.matched)
    rs.amount_fp = len(a.unmatched_pred)
    rs.amount_fn = len(a.unmatched_gold)

    # ---- detail field metrics (correct only if amount matched AND detail equal) ----
    rs.detail_tp = sum(1 for g, p in a.matched if norm_text(g["detail"]) == norm_text(p["detail"]))
    rs.detail_fp = len(pred) - rs.detail_tp
    rs.detail_fn = len(gold) - rs.detail_tp

    # ---- exact match (order-independent multiset of (amount, norm detail)) ----
    g_set = Counter((norm_amount(t["amount"]), norm_text(t["detail"])) for t in gold)
    p_set = Counter((norm_amount(t["amount"]), norm_text(t["detail"])) for t in pred)
    rs.exact = g_set == p_set
    rs.count_ok = len(gold) == len(pred)

    # ---- failure taxonomy ----
    rs.errors = classify_errors(gold, pred, a, err)
    return rs


def classify_errors(gold, pred, a: Aligned, err: str | None) -> list[str]:
    tags = []
    if err:
        tags.append(f"degraded:{err}")

    # non-transaction message that leaked transactions
    if not gold and pred:
        tags.append("false_positive")
        return tags  # nothing else meaningful to say

    # wrong detail on an otherwise-correct transaction
    for g, p in a.matched:
        if norm_text(g["detail"]) != norm_text(p["detail"]):
            tags.append("wrong_detail")

    ug = list(a.unmatched_gold)
    up = list(a.unmatched_pred)

    # merged: a predicted amount equals the sum of >=2 unmatched golds
    for p in list(up):
        combo = [g for g in ug]
        if len(combo) >= 2 and abs(sum(norm_amount(g["amount"]) for g in combo) - norm_amount(p["amount"])) < 1e-9:
            tags.append("merged_transactions")
            up.remove(p)
            ug = []
            break

    # split: a gold amount equals the sum of >=2 unmatched preds
    for g in list(ug):
        combo = [p for p in up]
        if len(combo) >= 2 and abs(sum(norm_amount(p["amount"]) for p in combo) - norm_amount(g["amount"])) < 1e-9:
            tags.append("split_transaction")
            ug.remove(g)
            up = []
            break

    # wrong_amount: a leftover gold and pred share the same detail (right item, wrong number)
    for g in list(ug):
        match = next((p for p in up if norm_text(p["detail"]) == norm_text(g["detail"])), None)
        if match is not None:
            tags.append("wrong_amount")
            ug.remove(g)
            up.remove(match)

    # whatever remains
    if ug:
        tags.append("missed_transaction")
    if up:
        tags.append("extra_transaction")
    return tags


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def aggregate(scores: list[RowScore]) -> dict:
    s = lambda attr: sum(getattr(x, attr) for x in scores)
    ap, ar, af = prf(s("amount_tp"), s("amount_fp"), s("amount_fn"))
    dp, dr, df = prf(s("detail_tp"), s("detail_fp"), s("detail_fn"))
    n = len(scores) or 1
    return {
        "n": len(scores),
        "amount": {"p": ap, "r": ar, "f1": af},
        "detail": {"p": dp, "r": dr, "f1": df},
        "exact_match_rate": sum(x.exact for x in scores) / n,
        "count_accuracy": sum(x.count_ok for x in scores) / n,
    }


def pctile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = (len(vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


# --------------------------------------------------------------------------- #
# Run one model over the dataset
# --------------------------------------------------------------------------- #
def run_model(model: str, rows: list[dict], extractor=extract_verbose, label: str | None = None) -> dict:
    pricing = PRICING.get(model, {"in": 0.0, "out": 0.0})
    results, scores = [], []
    latencies, total_cost = [], 0.0
    routes = Counter()

    for row in rows:
        res = extractor(row["input"], model)
        routes[res.route] += 1
        rs = score_row(row["transactions"], res.transactions, res.error)
        scores.append(rs)
        if not res.short_circuited and res.latency_ms > 0:
            latencies.append(res.latency_ms)
        total_cost += res.prompt_tokens * pricing["in"] + res.completion_tokens * pricing["out"]
        results.append({
            "input": row["input"][:120],
            "bucket": row["bucket"],
            "expected": row["transactions"],
            "got": res.transactions,
            "errors": rs.errors,
            "latency_ms": round(res.latency_ms, 1),
            "api_error": res.error,
        })

    overall = aggregate(scores)
    # per-bucket
    by_bucket = {}
    bucket_scores = defaultdict(list)
    for row, rs in zip(rows, scores):
        bucket_scores[row["bucket"]].append(rs)
    for b, sc in bucket_scores.items():
        by_bucket[b] = aggregate(sc)

    # failure taxonomy with examples
    taxonomy = Counter()
    examples = defaultdict(list)
    for row, rs in zip(rows, scores):
        for tag in rs.errors:
            key = tag.split(":")[0] if tag.startswith("degraded") else tag
            taxonomy[key] += 1
            if len(examples[key]) < 3:
                examples[key].append({"input": row["input"][:80], "expected": row["transactions"],
                                      "got": next(r["got"] for r in results if r["input"][:80] == row["input"][:80])})

    n = len(rows)
    return {
        "model": label or model,
        "base_model": model,
        "routes": dict(routes),
        "pricing_per_token": pricing,
        "overall": overall,
        "by_bucket": by_bucket,
        "latency_p50_ms": round(pctile(latencies, 0.50), 1),
        "latency_p95_ms": round(pctile(latencies, 0.95), 1),
        "n_api_calls": len(latencies),
        "cost_per_1k_usd": round(total_cost / n * 1000, 5) if n else 0.0,
        "failure_taxonomy": dict(taxonomy.most_common()),
        "failure_examples": {k: v for k, v in examples.items()},
        "rows": results,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}"


def print_report(reports: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("MODEL COMPARISON")
    print("=" * 78)
    print(f"\n| {'Model':<32} | {'Amt F1':>6} | {'Det F1':>6} | {'Exact':>6} | "
          f"{'Cnt':>5} | {'p50ms':>6} | {'p95ms':>6} | {'$/1k':>8} |")
    print(f"|{'-'*34}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*7}|{'-'*8}|{'-'*8}|{'-'*10}|")
    for r in reports:
        o = r["overall"]
        print(f"| {r['model']:<32} | {fmt_pct(o['amount']['f1']):>6} | {fmt_pct(o['detail']['f1']):>6} | "
              f"{fmt_pct(o['exact_match_rate']):>6} | {fmt_pct(o['count_accuracy']):>5} | "
              f"{r['latency_p50_ms']:>6} | {r['latency_p95_ms']:>6} | {r['cost_per_1k_usd']:>8} |")

    for r in reports:
        print("\n" + "-" * 78)
        print(f"MODEL: {r['model']}")
        print("-" * 78)
        o = r["overall"]
        print(f"  amount  P/R/F1 : {fmt_pct(o['amount']['p'])} / {fmt_pct(o['amount']['r'])} / {fmt_pct(o['amount']['f1'])}")
        print(f"  detail  P/R/F1 : {fmt_pct(o['detail']['p'])} / {fmt_pct(o['detail']['r'])} / {fmt_pct(o['detail']['f1'])}")
        print(f"  exact-match    : {fmt_pct(o['exact_match_rate'])}%   count-acc: {fmt_pct(o['count_accuracy'])}%")
        print(f"  latency p50/p95: {r['latency_p50_ms']} / {r['latency_p95_ms']} ms   ({r['n_api_calls']} API calls)")
        print(f"  cost / 1k msgs : ${r['cost_per_1k_usd']}")
        print("  per-bucket (amount F1 / detail F1 / exact):")
        for b, m in r["by_bucket"].items():
            print(f"    {b:<16}: {fmt_pct(m['amount']['f1']):>5} / {fmt_pct(m['detail']['f1']):>5} / {fmt_pct(m['exact_match_rate']):>5}  (n={m['n']})")
        print("  failure taxonomy:")
        if not r["failure_taxonomy"]:
            print("    (none)")
        for tag, c in r["failure_taxonomy"].items():
            print(f"    {tag:<22}: {c}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Thai output on Windows consoles
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset.jsonl")
    ap.add_argument("--model", action="append", help="repeatable; defaults to the 3 candidates")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N rows (debug)")
    ap.add_argument("--tiered", action="store_true",
                    help="also run the regex->LLM tiered optimizer on the ship model and show the delta")
    args = ap.parse_args()

    models = args.model or DEFAULT_MODELS
    rows = [json.loads(l) for l in open(args.dataset, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    if not os.getenv("OPENROUTER_API_KEY"):
        print("WARNING: OPENROUTER_API_KEY not set -> every call degrades to []. "
              "Set it in .env to get real numbers.\n")

    os.makedirs("reports", exist_ok=True)
    reports = []
    for model in models:
        print(f"Running {model} over {len(rows)} messages ...")
        rep = run_model(model, rows)
        safe = model.replace("/", "_")
        with open(f"reports/eval_{safe}.json", "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        reports.append(rep)

    print_report(reports)

    if args.tiered:
        from src.tiered import extract_tiered_verbose
        ship = "google/gemini-2.5-flash-lite"
        print(f"\nRunning TIERED (regex -> {ship}) ...")
        tier = run_model(ship, rows, extractor=extract_tiered_verbose, label=f"tiered+{ship}")
        with open("reports/eval_tiered.json", "w", encoding="utf-8") as f:
            json.dump(tier, f, ensure_ascii=False, indent=2)

        base = next((r for r in reports if r["base_model"] == ship), None)
        print_report([tier])
        if base:
            d_f1 = (tier["overall"]["detail"]["f1"] - base["overall"]["detail"]["f1"]) * 100
            c_base, c_tier = base["cost_per_1k_usd"], tier["cost_per_1k_usd"]
            saved = (1 - c_tier / c_base) * 100 if c_base else 0.0
            print("\n" + "=" * 78)
            print("COST OPTIMIZATION DELTA (tiered vs pure Gemini Flash-Lite)")
            print("=" * 78)
            print(f"  routes (tiered)   : {tier['routes']}")
            print(f"  detail F1 delta   : {d_f1:+.1f} pts  ({fmt_pct(base['overall']['detail']['f1'])} -> {fmt_pct(tier['overall']['detail']['f1'])})")
            print(f"  cost / 1k delta   : ${c_base} -> ${c_tier}   ({saved:.1f}% cheaper)")

    print(f"\nSaved per-model JSON to reports/. Ran {len(rows)} messages x {len(models)} models.")


if __name__ == "__main__":
    main()
