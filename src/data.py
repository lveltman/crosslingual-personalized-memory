"""Датасет для baseline-компрессии контекста + двускриптный контроль.

Baseline-задача (ICAE-стиль): подаём контекст -> сжимаем в N memory-слотов ->
frozen decoder восстанавливает тот же контекст (autoencoding).

Для метрики Morphological Fidelity к каждому примеру прилагается разметка:
  tag_ids: dict{CASE,NEG,NUM} -> список id тегов на уровне word-токенов,
  y_multi_hot: multi-hot по FLAT_TAGS (кла́узный/пример-уровень),
  script: 'latin' | 'cyrillic',
  tag_source: 'gold' | 'auto'   (для честного разделения train/test).

ВАЖНО (правки ревью №9, №10): синтетический генератор — ТОЛЬКО smoke-test.
Он использует rule-based правила с известной омонимией аффиксов, поэтому НЕ
пригоден для отчётных метрик. Реальный корпус подключается через from_jsonl()
с человеческой (gold) разметкой; train.py требует --eval_jsonl для отчёта.
"""
import json
import random
import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from morph_tags import TAG_VOCAB, CATEGORIES, FLAT_TAGS, FLAT_TAG2ID, CAT_TAG2ID


# --- Простейшая морфемно-консистентная транслитерация Φ (кир<->лат) ---
# Замечание рецензента №3 / правка ревью №13: маппинг апострофных графем
# неоднозначен. Здесь фиксируем ЯВНЫЙ маппинг и применяем Unicode-нормализацию.
# Все апострофные варианты приводятся к единой графеме U+2018 (‘) — типографский
# ЛЕВЫЙ полукруг, стандартно используемый в узбекской латинице для o‘/g‘.
CYR2LAT = {
    "ў": "o‘", "қ": "q", "ғ": "g‘", "ҳ": "h",
    "ш": "sh", "ч": "ch", "нг": "ng", "я": "ya", "ю": "yu", "ё": "yo",
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "j",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
    "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "x", "ц": "ts", "ъ": "’", "ь": "", "э": "e",
}

# Единая целевая апострофная графема и множество её вариантов для нормализации.
APOSTROPHE_CANON = "‘"  # U+2018 LEFT SINGLE QUOTATION MARK
APOSTROPHE_VARIANTS = {"'", "’", "‘", "`", "ʻ", "ʼ", "´"}


def normalize_unicode(text: str) -> str:
    """Unicode-нормализация (NFC) + унификация апострофных графем.

    Правка ревью №13: устраняем silent script-mismatch из-за разного кодирования
    апострофов (U+2019 vs U+2018 vs U+0027 и т.п.). Регистр НЕ трогаем здесь —
    это делает вызывающая сторона осознанно.
    """
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        out.append(APOSTROPHE_CANON if ch in APOSTROPHE_VARIANTS else ch)
    return "".join(out)


def _restore_case(src_ch: str, mapped: str) -> str:
    """Восстанавливаем регистр: если исходный символ был заглавным — капитализируем
    результат транслитерации (первую букву). Правка ревью №13 — сохранение регистра."""
    if src_ch.isupper() and mapped:
        return mapped[0].upper() + mapped[1:]
    return mapped


# Плейсхолдеры анонимизатора (<PERSON_NAME>, <PHONE_NUMBER>, ...) — спец-токены,
# их НЕЛЬЗЯ транслитерировать: иначе dual-script контроль ломается (плейсхолдер
# токенизируется по-разному в лат/кир) и появляется ложная скрипт-разница.
# Держим их дословно в обоих скриптах.
_PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]*>")


def _translit_protecting_placeholders(text: str, core) -> str:
    """Применяет core-транслитерацию только к не-плейсхолдерным сегментам."""
    out, last = [], 0
    for m in _PLACEHOLDER_RE.finditer(text):
        out.append(core(text[last:m.start()]))
        out.append(m.group(0))            # плейсхолдер — дословно
        last = m.end()
    out.append(core(text[last:]))
    return "".join(out)


def transliterate_cyr2lat(text: str) -> str:
    """кир->лат транслитерация; плейсхолдеры <...> сохраняются дословно."""
    return _translit_protecting_placeholders(text, _cyr2lat_core)


def _cyr2lat_core(text: str) -> str:
    """Наивная кир->лат транслитерация с сохранением регистра и Unicode-нормализацией.

    Правка ревью №13:
      - применяем NFC + унификацию апострофов ДО и ПОСЛЕ транслитерации;
      - не теряем регистр (обрабатываем посимвольно с восстановлением заглавных).
    Multi-char ключи ('нг') обрабатываем до однобуквенных.
    """
    text = normalize_unicode(text)
    out = []
    i = 0
    n = len(text)
    while i < n:
        # Пытаемся сматчить двухсимвольный ключ (регистр-независимо).
        two = text[i:i + 2]
        two_low = two.lower()
        if two_low in CYR2LAT:
            mapped = CYR2LAT[two_low]
            # Регистр по первому символу пары.
            out.append(_restore_case(two[0], mapped))
            i += 2
            continue
        ch = text[i]
        ch_low = ch.lower()
        mapped = CYR2LAT.get(ch_low, ch)
        out.append(_restore_case(ch, mapped))
        i += 1
    return normalize_unicode("".join(out))


# --- Обратная транслитерация Φ⁻¹ (лат->кир) для двускриптного контроля ---
# После normalize_unicode все апострофные графемы -> ‘ (U+2018). Значит oʻ/gʻ
# матчатся как "o‘"/"g‘", а одиночный ‘ трактуем как tutuq belgisi -> ъ.
# ВНИМАНИЕ: лат->кир неоднозначна (e->э/е, диграфы); это документированный шум
# «контроля» (см. THESIS.md, замечание рецензента №3). Доля неоднозначных случаев
# считается отдельно.
LAT2CYR_DIGRAPHS = [
    ("o‘", "ў"), ("g‘", "ғ"),
    ("sh", "ш"), ("ch", "ч"), ("ng", "нг"),
    ("yo", "ё"), ("yu", "ю"), ("ya", "я"), ("ye", "е"), ("ts", "ц"),
]
LAT2CYR_SINGLE = {
    "a": "а", "b": "б", "d": "д", "f": "ф", "g": "г", "h": "ҳ", "i": "и",
    "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "x": "х",
    "y": "й", "z": "з", "c": "с", "‘": "ъ",
}


def transliterate_lat2cyr(text: str) -> str:
    """лат->кир транслитерация; плейсхолдеры <...> сохраняются дословно."""
    return _translit_protecting_placeholders(text, _lat2cyr_core)


def _lat2cyr_core(text: str) -> str:
    """Наивная лат->кир транслитерация (Φ⁻¹) с сохранением регистра и границ слов.

    Longest-match: сначала диграфы, потом одиночные буквы. 'e' -> 'э' в начале
    слова, иначе 'е' (детерминированное правило; орфографически не идеально —
    задокументированный шум контроля).
    """
    text = normalize_unicode(text)
    out = []
    i, n = 0, len(text)
    while i < n:
        two = text[i:i + 2]
        two_low = two.lower()
        matched = None
        # Неоднозначность yoʻ: если после "yo" идёт апостроф, то это y + oʻ (й+ў),
        # а не диграф yo (ё). Напр. yoʻq -> йўқ, а не ёъқ. Тогда 'y' обрабатываем
        # как одиночную, а "oʻ" сматчится на следующей итерации.
        if two_low == "yo" and i + 2 < n and text[i + 2] == APOSTROPHE_CANON:
            two_low = "__skip__"
        for lat, cyr in LAT2CYR_DIGRAPHS:
            if two_low == lat:
                matched = cyr
                break
        if matched is not None:
            out.append(_restore_case(two[0], matched))
            i += 2
            continue
        ch = text[i]
        ch_low = ch.lower()
        if ch_low == "e":
            at_word_start = (i == 0) or (not text[i - 1].isalpha())
            mapped = "э" if at_word_start else "е"
        else:
            mapped = LAT2CYR_SINGLE.get(ch_low, ch)
        out.append(_restore_case(ch, mapped))
        i += 1
    return normalize_unicode("".join(out))


@dataclass
class MorphExample:
    text: str
    script: str                       # 'latin' | 'cyrillic'
    tag_ids: Dict[str, List[int]]     # категория -> id тега на word-токен (0=PAD)
    y_multi_hot: List[int]            # multi-hot по FLAT_TAGS
    tag_source: str = "auto"          # 'gold' | 'auto'


def _synthetic_examples(n: int, seed: int = 0) -> List[MorphExample]:
    """Генерируем синтетические узбекоподобные примеры для smoke-теста харнесса.

    ВНИМАНИЕ (правка ревью №10): rule-based правила ниже воспроизводят омонимию
    аффиксов (например, подстрока 'ma' внутри корня vs суффикс -ma-) и НЕ являются
    истинной морфологией. Использовать ТОЛЬКО как dsummy для проверки пайплайна.
    """
    rng = random.Random(seed)
    latin_words = ["kitob", "uy", "tasdiqlandi", "tasdiqlanmadi", "bola", "kelmadi", "keldi"]
    examples = []
    for _ in range(n):
        k = rng.randint(3, 8)
        words = [rng.choice(latin_words) for _ in range(k)]
        text = " ".join(words)
        # Простейший rule-based "анализатор" A∘Φ поверх токенов (заведомо шумный):
        case_ids, neg_ids, num_ids = [], [], []
        y = [0] * len(FLAT_TAGS)
        for w in words:
            # NEG: наличие суффикса отрицания -ma-/-mad- (ОМОНИМИЯ не разрешается!)
            neg = 1 if ("mad" in w or "ma" in w) else 0
            neg_tag = "NEG" if neg else "POS"
            neg_ids.append(CAT_TAG2ID["NEG"][neg_tag])
            y[FLAT_TAG2ID[f"NEG:{neg_tag}"]] = 1
            # NUM: суффикс -lar -> PL
            num_tag = "PL" if w.endswith("lar") else "SG"
            num_ids.append(CAT_TAG2ID["NUM"][num_tag])
            y[FLAT_TAG2ID[f"NUM:{num_tag}"]] = 1
            # CASE: грубо по окончанию
            if w.endswith("ni"): case_tag = "ACC"
            elif w.endswith("ning"): case_tag = "GEN"
            elif w.endswith("ga"): case_tag = "DAT"
            elif w.endswith("da"): case_tag = "LOC"
            elif w.endswith("dan"): case_tag = "ABL"
            else: case_tag = "NOM"
            case_ids.append(CAT_TAG2ID["CASE"][case_tag])
            y[FLAT_TAG2ID[f"CASE:{case_tag}"]] = 1

        script = rng.choice(["latin", "cyrillic"])
        examples.append(MorphExample(
            text=text, script=script,
            tag_ids={"CASE": case_ids, "NEG": neg_ids, "NUM": num_ids},
            y_multi_hot=y, tag_source="auto",   # синтетика ВСЕГДА auto
        ))
    return examples


class CompressionDataset(Dataset):
    """Датасет для baseline-компрессии + морфо-разметка.

    Возвращает сырой пример; токенизацию и alignment тегов делает collator,
    т.к. именно там (замечание рецензента №3) происходит word->subword выравнивание.
    """
    def __init__(self, examples: List[MorphExample], is_synthetic: bool = False):
        self.examples = examples
        # Флаг синтетики нужен train.py, чтобы не выдавать synthetic-числа как отчётные.
        self.is_synthetic = is_synthetic

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

    @classmethod
    def synthetic(cls, n=64, seed=0):
        return cls(_synthetic_examples(n, seed), is_synthetic=True)

    @classmethod
    def from_jsonl(cls, path: str):
        exs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                exs.append(MorphExample(**d))
        return cls(exs, is_synthetic=False)
