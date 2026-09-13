"""E2: генерация МИНИМАЛЬНЫХ ПАР отрицания для причинного теста.

Пара = два примера, различающиеся ТОЛЬКО полярностью предиката (утвердительный vs
отрицательный), в остальном идентичный банковский контекст. Это контролирует всё,
кроме полярности: если сжатая память теряет отрицание, пара становится неразличимой.

К каждому примеру прилагается yes/no вопрос и эталонный ответ (ha/yo'q) для downstream:
подаём сжатую память как soft-prefix, спрашиваем «одобрено?» и проверяем, отличается ли
ответ между утвердительным и отрицательным членом пары.

Выход: jsonl, совместимый с MorphExample (text/script/tag_ids/y_multi_hot/tag_source),
плюс поля pair_id / polarity / question / answer. Двускриптно (--dual_script).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import transliterate_lat2cyr  # noqa: E402
from morph_tags import TAG_VOCAB, FLAT_TAGS, FLAT_TAG2ID, CAT_TAG2ID  # noqa: E402

# Пары предикатов: (утвердительный, отрицательный, англ-смысл для вопроса).
VERB_PAIRS = [
    ("tasdiqlandi", "tasdiqlanmadi", "confirmed"),
    ("keldi", "kelmadi", "came"),
    ("to‘landi", "to‘lanmadi", "paid"),
    ("bajarildi", "bajarilmadi", "completed"),
    ("tugadi", "tugamadi", "finished"),
    ("bog‘landi", "bog‘lanmadi", "connected"),
    ("ochildi", "ochilmadi", "opened"),
    ("yakunlandi", "yakunlanmadi", "closed"),
    ("o‘tkazildi", "o‘tkazilmadi", "transferred"),
    ("ro‘yxatdan o‘tdi", "ro‘yxatdan o‘tmadi", "registered"),
]

# Рамки: (SUBJ, текст со слотом {V}). SUBJ — подлежащее, к которому привязан вопрос,
# чтобы yes/no было прямым считыванием полярности предиката (потолок обязан быть ~1.0).
FRAMES = [
    ("To‘lovingiz", "Assalomu alaykum. Sizning to‘lovingiz {V}. Rahmat, xayr."),
    ("Arizangiz", "Alo, <PERSON_NAME>? Sizning arizangiz {V}. Boshqa savolingiz bormi?"),
    ("Kartangiz", "Hurmatli mijoz, kartangiz {V}. Iltimos, ilovani tekshiring."),
    ("O‘tkazmangiz", "Salom. Pul o‘tkazmangiz {V}. Yordam kerakmi?"),
    ("Hisobingiz", "Xayrli kun. Hisobingiz {V}. Batafsil uchun qo‘ng‘iroq qiling."),
    ("Sug‘urtangiz", "Assalomu alaykum. Sug‘urtangiz {V}. Kechirasiz, yana tekshiramiz."),
]

# Вопрос строится из подлежащего + утвердительного предиката + -mi? (узб. yes/no).
# Эталон: ha (утв.) / yo‘q (отр.). Полярность ответа = полярность предиката в тексте.
def make_question(subj, pos_verb):
    return f"{subj} {pos_verb}mi?"


def build_tags(words, verb_word_idx, polarity):
    """Минимальные per-word теги: у предиката NEG=POS/NEG, у остальных дефолт.

    Дефолты (NOM/SG/POS) — как немаркированный фон; на предикате ставим полярность.
    Для E2 важна только ось NEG, падеж/число оставляем дефолтными (не оцениваем).
    """
    n = len(words)
    neg_id = CAT_TAG2ID["NEG"]["NEG" if polarity == "neg" else "POS"]
    tag_ids = {
        "CASE": [CAT_TAG2ID["CASE"]["NOM"]] * n,
        "NEG":  [CAT_TAG2ID["NEG"]["POS"]] * n,
        "NUM":  [CAT_TAG2ID["NUM"]["SG"]] * n,
    }
    tag_ids["NEG"][verb_word_idx] = neg_id
    # sentence-level multi-hot
    y = [0] * len(FLAT_TAGS)
    y[FLAT_TAG2ID["CASE:NOM"]] = 1
    y[FLAT_TAG2ID["NUM:SG"]] = 1
    y[FLAT_TAG2ID["NEG:POS"]] = 1  # фон всегда содержит утвердительные предикаты
    if polarity == "neg":
        y[FLAT_TAG2ID["NEG:NEG"]] = 1
    return tag_ids, y


def make_example(text, polarity, pair_id, verb_word_idx, question, script="latin"):
    words = text.split()
    tag_ids, y = build_tags(words, verb_word_idx, polarity)
    return {
        "text": text, "script": script,
        "tag_ids": tag_ids, "y_multi_hot": y, "tag_source": "gold",
        "pair_id": pair_id, "polarity": polarity,
        "question": question,
        "answer": "yo‘q" if polarity == "neg" else "ha",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", default="data/neg_pairs.jsonl")
    ap.add_argument("--dual_script", action="store_true")
    args = ap.parse_args()

    rows = []
    pair_id = 0
    for subj, frame in FRAMES:
        for pos_v, neg_v, _ in VERB_PAIRS:
            question = make_question(subj, pos_v)
            for polarity, verb in (("pos", pos_v), ("neg", neg_v)):
                text = frame.replace("{V}", verb)
                # индекс слова предиката = первое слово внутри вставленного verb
                prefix = frame.split("{V}")[0]
                verb_word_idx = len(prefix.split())
                rows.append(make_example(text, polarity, pair_id, verb_word_idx, question))
                if args.dual_script:
                    cyr = transliterate_lat2cyr(text)
                    rows.append(make_example(cyr, polarity, pair_id, verb_word_idx,
                                             question, script="cyrillic"))
            pair_id += 1

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_pairs = pair_id
    print(f"[E2] {len(rows)} примеров, {n_pairs} пар × "
          f"{'2 скрипта' if args.dual_script else '1 скрипт'} -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
