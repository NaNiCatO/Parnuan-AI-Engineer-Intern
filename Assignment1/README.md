# Parnuan — Text → Transaction NER

Extract structured transactions from free-form Thai / mixed Thai-English messages, as strict JSON.

```
"ข้าวมันไก่ 50 น้ำเปล่า 7 แล้วก็ช้อปปิ้ง 500"
→ {"transactions": [
     {"amount": 50,  "detail": "ข้าวมันไก่"},
     {"amount": 7,   "detail": "น้ำเปล่า"},
     {"amount": 500, "detail": "ช้อปปิ้ง"}
   ]}
```

This is the **NER layer only** — no categorization, timestamps, UI, or storage. The system's first
job is to be *graceful*: it always returns the contract shape, never crashes, never hallucinates
amounts, and never leaks its prompt — accuracy comes second.

---

## Setup & run

Requires [uv](https://github.com/astral-sh/uv) and an [OpenRouter](https://openrouter.ai) key.
Run all commands from the `Assignment1/` directory.

```bash
cd Assignment1

# 1. install
uv sync

# 2. add your key  (a .env at the repo root also works)
cp .env.example .env        # then edit .env: OPENROUTER_API_KEY=sk-or-...

# 3. run on one message
uv run python src/ner.py "ข้าวมันไก่ 50"

# 4. run the full eval (all 3 models + failure taxonomy + cost)
uv run python src/eval.py

# 5. run the bonus tiered cost optimizer and see the delta
uv run python src/eval.py --tiered
```

> On Windows, Thai console output needs a UTF-8 terminal; the scripts call
> `sys.stdout.reconfigure(encoding="utf-8")` so a plain `uv run` works.

---

## 1. Approach

A **pure-LLM extractor** wrapped in a **defensive validation layer**, evaluated across three price
tiers, with an optional **regex→LLM tiered router** for cost.

In plain terms:

- **Pure-LLM extractor** — the actual extraction is done *only* by an LLM call. There are no hand-written
  parsing rules in the core path; we give the model the raw text + a prompt and it returns the
  transactions. ("Pure" = the logic lives in the prompt and model, not in custom code.)
- **Defensive validation layer** — the model's output is never trusted directly. A wrapper around the
  call assumes the model can misbehave and contains it: it repairs/parses the JSON, checks it against a
  fixed schema (`{"transactions": [{"amount": number, "detail": string}]}`), coerces bad values, and on
  *any* failure (bad JSON, wrong shape, timeout, 5xx, rate-limit) returns `{"transactions": []}` instead
  of crashing or passing through garbage. The LLM does the smart-but-unreliable work; this layer makes
  the system's output reliable regardless. (This is what caught Claude's one unparseable response — §7.)

The whole system is ~2 small files:

- `src/ner.py` — `extract(text, model) -> {"transactions": [...]}`. Calls OpenRouter, parses/repairs the
  JSON, validates it against a Pydantic contract, and coerces *anything* unexpected to an empty array.
- `src/eval.py` — runs the dataset through any model, scores it, and prints a report.
- `src/tiered.py` — (bonus) a conservative regex fast-path that handles the clean 80% for free.

Why pure-LLM as the core: Thai is unsegmented (no spaces between words), code-switches with English, and
uses Thai numerals (๕๐), number-words (ห้าสิบ) and slang. A rules-only parser is brittle here; an LLM
generalizes. The interesting engineering is not "can it parse `ข้าวมันไก่ 50`" — it's **what happens on
the inputs that aren't that**, which is where the validation layer and eval taxonomy live.

I deliberately did **not** build classes, a framework, or an async pipeline. The assignment penalizes
over-engineering and the problem is a script.

---

## 2. Dataset

`data/dataset.jsonl` — **55 labeled examples**, each `{"input", "transactions", "bucket"}`,
tagged by bucket so the eval reports per-bucket metrics.

| Bucket | n | What it covers |
|--------|---|----------------|
| `happy_single` | 10 | one clean `item amount` |
| `happy_multi` | 15 | 2–3 transactions, with/without connectors (`แล้วก็`, `กับ`, commas, newlines) |
| `messy` | 12 | mixed Thai/Eng (`coffee 80`), Thai numerals (`๕๐`), number-words (`ห้าสิบ`), `2k`, `บาท`/`฿`, commas, decimals, no-space (`ข้าว20`) |
| `non_transaction` | 8 | greetings/questions → `[]`, incl. a number-as-time trap (`ประชุมตอน 10 โมง`) |
| `adversarial` | 10 | empty, whitespace, prompt injection (Thai+Eng), amount-only, detail-only, emoji, weird unicode, huge input, and an **injection that tries to inject a fake transaction** |

All 4 required demo cases are present (rows for `ข้าวมันไก่ 50`, the multi case, `สวัสดีครับ วันนี้อากาศดี`,
and injection). Process: I wrote the spec above — the buckets, target counts, and labeling rules — and
used an LLM to generate the examples against it, then reviewed the labels for the rules below.

**Definition of "correct" (the labeling rules I locked before writing the data):**

- `amount` is numeric only, value preserved exactly. Thai numerals → Arabic (`๕๐`→50), number-words →
  number (`ห้าสิบ`→50), `2k`→2000, strip `บาท`/`฿`/commas, keep decimals (`87.50`→87.5).
- `detail` is the item/merchant. **Leading action verbs are dropped** (ซื้อ/จ่าย/เติม) but **`ค่า…` is
  kept** (ค่าไฟ, ค่าน้ำ — "ค่า" is part of the noun, not a verb).
- **amount-only** (`500`) and **detail-only** (`ค่ากาแฟ`) → `[]`. A transaction needs both halves.
- A number that isn't a price (`10 โมง` = 10 o'clock) → not a transaction.

**Deliberately left out:** dates/categories/currencies other than THB (out of scope), OCR/voice noise,
and very long real chat logs. See *Known limitations*.

---

## 3. Prompt / parsing strategy

- **Structured output:** request `response_format={"type":"json_object"}` so models that support it
  return parseable JSON directly.
- **Few-shot system prompt** (`SYSTEM_PROMPT` in `ner.py`) with examples for single, multi,
  non-transaction, and two injection cases — including one that *embeds* a fake-transaction instruction,
  to teach "the user message is data, not instructions."
- **`temperature=0`** for determinism.
- **Parse-then-repair:** strip ```` ```json ```` fences; if `json.loads` fails, grab the first `{...}`
  block; if that fails too → `[]`.
- **Pydantic contract:** every result is validated. Amounts given as `"1,500"` strings are coerced;
  transactions with an empty `detail` are dropped; integral floats render as ints (`50.0`→`50`). Any
  shape that doesn't fit → `{"transactions": []}`.

What worked: JSON mode + temperature 0 + a short, rule-dense prompt was enough for ~98% on the cheap
model. What didn't matter: longer prompts / more examples gave no measurable lift and cost more tokens.

---

## 4. Eval methodology

`src/eval.py`, one command. For each message it aligns predicted vs gold transactions, then scores.

**Why these metrics:** a single accuracy number hides everything. Field-level P/R/F1 separates "got the
money wrong" from "got the label wrong" (very different severities in finance). Exact-match and
count-accuracy capture whole-message correctness. Latency and cost are first-class because they decide
what actually ships.

**Transaction alignment (how I define a match):** for each message, greedily pair each gold transaction
with a predicted one of **equal amount** (one-to-one). Then:

- matched pair → **amount** true-positive; its `detail` is scored too.
- `detail` compared **normalized** (trim + casefold + collapse internal whitespace).
- a detail counts as correct **only if attached to a correctly-extracted amount** (strict but honest).
- unmatched predicted → false positive (hallucinated); unmatched gold → false negative (missed).

| Metric | Meaning |
|--------|---------|
| amount P/R/F1 | did the number come through, exactly? |
| detail P/R/F1 | did the "what" match, on a correct amount? |
| exact-match rate | whole `transactions` array equals gold (order-independent) |
| count accuracy | right *number* of transactions |
| latency p50/p95 | over real API calls (short-circuits excluded) |
| cost / 1k msgs | live OpenRouter $/token × measured tokens |
| per-bucket | all of the above, split by bucket |

---

## 5. Model comparison

3 models via OpenRouter, spanning ~30× cost across three providers. Live pricing pulled from
`GET /api/v1/models` before the run. Full dataset (55 messages).

| Model | Amount F1 | Detail F1 | Exact | p50 / p95 (ms) | $/1k msgs | Notes |
|-------|-----------|-----------|-------|----------------|-----------|-------|
| `google/gemini-2.5-flash-lite` | **100.0** | 98.4 | 98.2 | **767 / 1666** | **$0.056** | budget ship pick; 1 Thai-spacing miss |
| `openai/gpt-5-mini` | 100.0 | **100.0** | **100.0** | 4530 / 8053 | $0.400 | perfect, but ~8s p95 and ~7× the cost |
| `anthropic/claude-sonnet-4.6` | 97.5 | 97.5 | 98.2 | 2903 / 5008 | $2.210 | **most expensive (~39×), lowest F1** |

(Prices per 1M tokens in/out: gemini-flash-lite 0.10/0.40 · gpt-5-mini 0.25/2.00 · sonnet-4.6 3.00/15.00.
Cost/1k is measured tokens × live price, full 55-message run.)

Notes: the flagship was the most expensive and the lowest F1 here — one clean 3-transaction message
came back as unparseable JSON, which the system degraded to `[]`. `gpt-5-mini` scored highest but was
~8× slower (p95) and ~7× more expensive than Gemini for 1.6 points of detail-F1.

---

## 6. Recommendation

**Ship `google/gemini-2.5-flash-lite`** (pure), or the **tiered regex→Gemini** system (see §12) for
lower cost at the same accuracy.

The reasoning is multi-objective, not F1-maximizing:

- **Quality:** 100 amount-F1 / 98.4 detail-F1. Its only miss is a Thai-spacing artifact
  (`ค่าวิน`→`ค่า วิน`), not a money error — and the tiered system removes even that.
- **Cost:** ~$0.056 / 1k messages. At 500k users × ~3 msgs/day (~1.5M msgs/day) that's roughly
  **$85/day**; `gpt-5-mini` would be ~$600/day and `sonnet-4.6` ~$3,300/day for the *same or worse*
  accuracy. The tiered system drops Gemini's bill by **67.5%** (~$28/day).
- **Latency:** p95 1.7s vs `gpt-5-mini`'s 8.1s.
- **Ceiling reference:** the flagship was the most expensive and the least accurate in this run, so
  "use the biggest model" is not justified here.

`gpt-5-mini` is the only model that beat Gemini (by 1.6 detail-F1 pts), at ~7× the cost and ~5× the
latency. For a finance app where the user reviews extractions before saving, that trade isn't worth it.

---

## 7. Failure taxonomy

Every error is bucketed (not just counted) with examples, in the eval output and `reports/*.json`.

Categories: `missed_transaction`, `extra_transaction` (hallucinated), `wrong_amount`, `wrong_detail`,
`merged_transactions`, `split_transaction`, `false_positive` (extracted from a non-transaction), and
`degraded` (system fell back to `[]` on an API/parse failure).

**Observed across the 3 models (55 messages each):**

- **Gemini Flash-Lite** — `wrong_detail` ×1: `ค่าวิน45บ` → `ค่า วิน` vs gold `ค่าวิน`. The model inserted
  a space into an unsegmented Thai compound. This is *the* canonical Thai-NLP failure — **word
  segmentation** — and the only error it made.
- **gpt-5-mini** — zero errors on the dataset.
- **claude-sonnet-4.6** — `degraded:unparseable_json` ×1 → `missed_transaction` ×1: on the clean message
  `เน็ต 599 ไฟ 850 น้ำ 320` it returned output the parser+repair couldn't salvage, so the system returned
  `[]` rather than crashing or emitting a bad shape. This is the contract's intended fallback.

No model in this run produced `wrong_amount`, `extra_transaction`, `merged`/`split`, or `false_positive`
errors on the adversarial and non-transaction buckets.

---

## 8. Graceful degradation

`extract()` is total — it never raises and never returns a non-contract shape. Verified behaviors:

| Input | Result |
|-------|--------|
| `""`, `"   "`, `None` | `{"transactions": []}` — short-circuited **before** any API call (saves cost) |
| non-transaction text | `[]` |
| prompt injection (`ignore all previous instructions…`) | `[]`, **no prompt leak** |
| injection that tries to add a fake txn | only the *real* transaction survives |
| malformed / non-JSON model reply | parse-repair, else `[]` |
| timeout / 5xx / 429 | retry with exponential backoff, then `[]` |
| oversized input | truncated to 4000 chars, never crashes |
| bad amount type (`"1,500"`) | coerced to `1500` |

The system prompt explicitly frames the user message as data and refuses to follow instructions inside
it; the validation layer is the backstop if the model misbehaves anyway.

---

## 9. Trade-offs

- **Optimized for:** graceful degradation, honest eval, cost-awareness, small surface area.
- **Sacrificed:** I score `detail` strictly (normalized exact match), so a semantically-fine paraphrase
  counts as wrong. This *under*-reports quality on purpose — I'd rather be pessimistic than flatter the
  model. I also kept the dataset at 55 curated examples rather than a noisy 500; coverage over volume.
- **Pure-LLM core** over rules: simpler and more general, at the cost of per-call latency/cost — which
  the tiered bonus then claws back.

---

## 10. Known limitations (what would fail at scale)

- **Thai word segmentation** is the real accuracy ceiling — detail boundaries/spaces (`ค่า วิน`).
  A `pythainlp` normalization pass on both labels and outputs would help.
- **No real distributional data:** 55 curated examples ≠ the long tail of 500k users (voice-to-text
  noise, multi-line receipts, emoji-laden chat, regional slang).
- **Ambiguous amounts:** quantities vs prices (`เบียร์ 2 ขวด 180`), ranges, discounts ("ลด 20%") are
  not robustly handled.
- **Cost at scale is real:** even at $0.056/1k, billions of messages add up — hence the tiered system.
- **Single-provider dependency:** OpenRouter/Gemini uptime; a cascade or local fallback would harden it.

---

## 11. What I'd improve with one more week

1. Add a `pythainlp`-based detail normalizer (shared by labeling + eval) to neutralize spacing noise.
2. Grow the dataset to ~300 with real (anonymized) message distributions and inter-annotator checks.
   In particular, add **harder messy patterns** the current set doesn't cover — e.g. *grouped*
   layouts where all items come first and all amounts after (`เน็ต ไฟ น้ำ 599 850 320`), which force
   positional alignment, plus ranges, discounts (`ลด 20%`), and quantity-vs-price (`เบียร์ 2 ขวด 180`).
3. Add a cheap-model→flagship **confidence cascade** (route only low-confidence messages up).
4. Add prompt-caching / batching for the LLM tier; measure the cost delta.
5. **Move the LLM tier to a self-hosted local model** to cut per-call API cost toward ~zero at scale.
   A small open-weight model (Qwen3 / a quantized Llama, or a Thai-specialized **Typhoon** by SCB 10X /
   **OpenThaiGPT**) running on our own GPU has no per-token charge — at 500k+ users the fixed GPU cost
   amortizes well below per-call API pricing, and keeps financial text in-house (privacy). The honest
   cost comparison: API is cheaper until throughput is high enough that a GPU stays busy; past that
   break-even, local inference is effectively free per message. Worth benchmarking F1 vs. the API
   models before committing — a local model only wins if it holds accuracy on Thai.

---

## 12. Cost optimization (bonus) — tiered regex → LLM

`src/tiered.py`, run with `uv run python src/eval.py --tiered`.

**Idea:** the common case is a clean `<item> <amount>` message. A conservative regex parses those for
**$0 / ~0ms**. Anything it isn't sure about — Thai numerals, number-words, time-like numbers, prompt
injection, or *any leftover unparsed text* — is **deferred to the LLM**, so accuracy can't drop below
the pure-LLM system.

**Routing on the 55-message eval:** `regex 34` (free) · `short_circuit 2` (empty/whitespace, free) ·
`llm 19`. So **36/55 (65%) never touch the LLM.**

**Result (tiered regex→Gemini vs pure Gemini):**

| | Detail F1 | $/1k msgs |
|--|-----------|-----------|
| pure Gemini Flash-Lite | 98.4 | $0.0565 |
| **tiered regex → Gemini** | **100.0** | **$0.0183** |
| delta | **+1.6 pts** | **−67.5%** |

The tiered system is **67.5% cheaper *and* slightly more accurate** — the regex happens to handle the one
case Gemini got wrong (`ค่าวิน45บ`). In production the regex hit-rate would be higher still, since most
logged transactions are clean `item amount` lines.

Trade-off: the regex is intentionally *cautious* (high precision, lower recall) — it defers anything
ambiguous (Thai numerals, number-words, injection, leftover text) and would rather pay for an LLM call
than guess wrong on a money field.

---

## 13. Time spent

Measured from git history: the build ran from the first commit (scaffold, 20:35) to the last (~22:06) —
**~1.5 hours of wall-clock build time**. That window also includes environment setup (uv/Python on
Windows), waiting on OpenRouter credit, three eval reruns, and README revisions from review feedback.
Add roughly 45 minutes of planning, plan critique, and dataset-spec work before the first commit, for a
**total of ~2.25 hours** (hands-on time is somewhat less, given the idle waits).
