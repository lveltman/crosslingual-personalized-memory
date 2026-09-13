# Cross-Lingual Personalized Memory for Frozen LLMs

Research code (MSc). A compact **latent memory** of a user's dialogue history is compressed into
soft-prompt vectors and prepended to a **frozen** multilingual LLM for personalization in a
**low-resource** language (Uzbek, dual-script Latin/Cyrillic). Research question: what transfers
across languages when personalization memory is reused, and what is lost — in particular
**critical facts / negation** — and how to preserve it.

## Structure

```
data_prep/               # data preparation & analysis
  anonymize.py           # raw dialogues -> anonymized
  detect_lang_script.py  # language / script detection (ru, uz-latin, uz-cyrillic, mixed)
  build_sessions.py      # group by user, clean, chronological history -> target
  make_splits.py         # author-disjoint train/dev/test, stratified
  extract_user_facts.py  # LLM extraction of user facts (incl. polarity) for the fidelity set
  make_negation_pairs.py # controlled negation minimal pairs (fidelity)
  annotate_gold_llm.py   # LLM annotation helper
  analyze_bootstrap.py   # bootstrap CIs + figures
src/
  config.py  collator.py  metrics.py  train.py  eval.py
  dataset.py             # personalization dataset (history -> target, cross-lingual conditions)
  data.py                # utils + Latin<->Cyrillic transliteration (dual-script)
  morph_tags.py  alignment.py
  architecture/          # the model
    behavioral_twin.py   # Q-Former / prefix-fusion personalization memory
    bt_dataset.py
    model.py             # cross-lingual personalized memory (main method)
    losses.py            # negation-preserving objectives (marker-weighted / contrastive / fact-slot)
    token_level.py       # ICAE-style token-level compressor (second architecture / ablation)
    mac_module.py        # naive morphology channel (ablation)
benchmark/               # baselines (no-personalization / full-context / RAG / PPlug / xRAG)
Dockerfile  requirements.txt  LICENSE
```

## Method (short)

Frozen multilingual encoder + frozen decoder LLM; only a small **memory module** (Q-Former /
learnable slots + projection) is trained. History is compressed into `N` slots and injected as a
prefix; no morphology tags are needed at inference. Two novelty axes: **cross-lingual alignment**
(dual-script transliteration as an alignment signal) and **negation-preserving training**
(marker-weighted loss, contrastive minimal pairs, dedicated fact slots).

## Setup

- Python env with `torch`, `transformers`, `peft`, `matplotlib` (see `requirements.txt`).
- A frozen multilingual decoder LLM and a multilingual sentence encoder (paths configured via env;
  `src/config.py`). Training only updates the memory module, so a single 24–48 GB GPU is enough.

## Pipeline

1. Prepare data: `anonymize -> detect_lang_script -> build_sessions -> make_splits`.
2. Build fidelity sets: `extract_user_facts`, `make_negation_pairs`.
3. Train the memory module (`src/train.py`), evaluate (`src/eval.py`), aggregate
   (`data_prep/analyze_bootstrap.py`).
4. Compare against baselines (`benchmark/`) across language conditions (en / ru / uz-latin /
   uz-cyrillic / code-switch).
