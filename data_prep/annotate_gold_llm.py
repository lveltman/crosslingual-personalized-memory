"""Gold-разметка морфологии через Sonnet (OpenRouter).

Для каждого примера (латиница) просим Sonnet разметить КАЖДОЕ слово по трём
категориям: CASE / NUM / NEG (или NA, если категория не применима — напр. падеж
у глагола). Это «silver+» разметка: чище наивного rule-based (разрешает омонимию
аффиксов -ma-, отличает падеж от корня). Пишем tag_source="gold".

Кириллические копии (--dual_script) получают ТЕ ЖЕ теги (транслит посимвольный —
границы слов и морфология сохраняются).

Использование:
    python data_prep/annotate_gold_sonnet.py \
        --in_jsonl data/uz_eval.jsonl --out_jsonl data/uz_eval_gold.jsonl \
        --limit 40 --dual_script
"""
import argparse
import json
import os
import re
import sys

from crewai import LLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from morph_tags import TAG_VOCAB, CATEGORIES, FLAT_TAGS, FLAT_TAG2ID, CAT_TAG2ID  # noqa: E402
from data import transliterate_lat2cyr  # noqa: E402

KEY = os.environ.get("OPENROUTER_API_KEY")
if not KEY:
    raise RuntimeError("OPENROUTER_API_KEY не задан.")

llm = LLM(model="openrouter/google/gemini-3.5-flash",
          base_url="https://openrouter.ai/api/v1", api_key=KEY, max_tokens=12000)

SYS = (
    "You are an expert in Uzbek (Latin script) morphology. You are given the word list "
    "of one dialogue turn. Tag EVERY word for three morphological categories and return "
    "STRICTLY a JSON array — one object per word, in the same order, no prose.\n\n"
    "Categories and allowed values:\n"
    "  case: NOM | GEN | ACC | DAT | LOC | ABL | NA\n"
    "        (grammatical case; ONLY for nouns/pronouns. Verbs, particles, interjections, "
    "punctuation -> NA)\n"
    "  num:  SG | PL | NA\n"
    "        (number; ONLY for nominals. Non-nominals -> NA)\n"
    "  neg:  POS | NEG | NA\n"
    "        (polarity; ONLY for verbs/predicates. NEG for negation -ma-/-mas-/emas/yo‘q; "
    "POS for an affirmative verb; non-predicates -> NA)\n\n"
    "CRITICAL RULES:\n"
    "  - Resolve affix homonymy: a substring 'ma' inside a ROOT is NOT negation "
    "(e.g. 'malaka' = skill -> neg=NA, not NEG).\n"
    "  - Case suffixes are ambiguous: judge by the whole word, not the ending alone.\n"
    "  - Placeholders like <PERSON_NAME>, <PHONE>, <...> are proper nouns: "
    "case=NOM, num=SG, neg=NA.\n"
    "  - The array length MUST equal the number of words.\n\n"
    "EXAMPLES:\n"
    "Words:\n0. kitoblarni\n1. o‘qimadim\n"
    "Output:\n[{\"case\":\"ACC\",\"num\":\"PL\",\"neg\":\"NA\"},"
    "{\"case\":\"NA\",\"num\":\"NA\",\"neg\":\"NEG\"}]\n\n"
    "Words:\n0. Malika\n1. uyga\n2. keldi\n"
    "Output:\n[{\"case\":\"NOM\",\"num\":\"SG\",\"neg\":\"NA\"},"
    "{\"case\":\"DAT\",\"num\":\"SG\",\"neg\":\"NA\"},"
    "{\"case\":\"NA\",\"num\":\"NA\",\"neg\":\"POS\"}]\n\n"
    "Words:\n0. <PERSON_NAME>\n1. malaka\n2. bormadi\n"
    "Output:\n[{\"case\":\"NOM\",\"num\":\"SG\",\"neg\":\"NA\"},"
    "{\"case\":\"NOM\",\"num\":\"SG\",\"neg\":\"NA\"},"
    "{\"case\":\"NA\",\"num\":\"NA\",\"neg\":\"NEG\"}]"
)


def parse_json_array(text):
    text = text.strip()
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def annotate(words):
    numbered = "\n".join(f"{i}. {w}" for i, w in enumerate(words))
    for attempt in range(2):
        msg = [{"role": "system", "content": SYS},
               {"role": "user", "content": f"Words ({len(words)}):\n{numbered}"}]
        if attempt == 1:
            msg[1]["content"] += (f"\n\nReturn EXACTLY {len(words)} objects in the JSON array.")
        arr = parse_json_array(llm.call(msg) or "")
        if arr is not None and len(arr) == len(words):
            return arr
    return None


def to_tags(arr):
    """arr[i] = {case,num,neg} -> (tag_ids{CASE,NEG,NUM}, y_multi_hot)."""
    case_ids, neg_ids, num_ids = [], [], []
    y = [0] * len(FLAT_TAGS)

    def put(cat, val, id_list):
        val = (val or "NA").upper()
        if val in TAG_VOCAB[cat]:
            id_list.append(CAT_TAG2ID[cat][val])
            y[FLAT_TAG2ID[f"{cat}:{val}"]] = 1
        else:
            id_list.append(0)  # NA / неприменимо

    for o in arr:
        put("CASE", o.get("case"), case_ids)
        put("NEG", o.get("neg"), neg_ids)
        put("NUM", o.get("num"), num_ids)
    return {"CASE": case_ids, "NEG": neg_ids, "NUM": num_ids}, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--dual_script", action="store_true")
    args = ap.parse_args()

    src = [json.loads(l) for l in open(args.in_jsonl, encoding="utf-8") if l.strip()]
    src = [e for e in src if e.get("script", "latin") == "latin"][:args.limit]

    out, failed = [], 0
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    for i, ex in enumerate(src):
        words = ex["text"].split()
        arr = annotate(words)
        if arr is None:
            failed += 1
            print(f"  [{i}] пропуск (не распарсилось/длина)")
            continue
        tag_ids, y = to_tags(arr)
        out.append({"text": ex["text"], "script": "latin",
                    "tag_ids": tag_ids, "y_multi_hot": y, "tag_source": "gold"})
        if args.dual_script:
            out.append({"text": transliterate_lat2cyr(ex["text"]), "script": "cyrillic",
                        "tag_ids": tag_ids, "y_multi_hot": y, "tag_source": "gold"})
        print(f"  [{i}] ok ({len(words)} слов)")

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nГотово: {len(out)} примеров -> {args.out_jsonl} (пропущено {failed})")


if __name__ == "__main__":
    main()
