# Parnuan AI Engineer Intern — Take-Home

Text → Transaction NER for Thai / mixed-language finance messages.

- **[Assignment 1](Assignment1/README.md)** — NER system (OpenRouter), honest eval + failure taxonomy,
  3-model comparison + recommendation, and a bonus regex→LLM tiered cost optimizer.
- **[Assignment 2](Assignment2/README.md)** — (bonus) LoRA fine-tune of a small open model
  (Qwen2.5-7B on free Colab; Typhoon intended but too large for a free T4) to match the A1 commercial
  baseline on the **same held-out eval set**, enabling self-hosted inference. Finding: the open model
  matches the baseline **zero-shot**. A2 reuses A1's eval harness + held-out eval set as the single
  source of truth.

Each folder is self-contained with its own README, setup, and run commands. Secrets go in a `.env`
(gitignored); never commit keys.
