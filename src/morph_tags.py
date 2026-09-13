"""Определения морфо-категорий и словарей тегов (общие для baseline-метрики и MAC).

ВАЖНО (по замечанию рецензента, уязвимость №1): здесь мы НЕ утверждаем
корректность анализатора A∘Φ. Теги трактуются как выход шумного учителя.
Метрика Morphological Fidelity считает СОГЛАСИЕ с эталонной разметкой,
которая должна поступать отдельно (gold или тот же A — это фиксируется
в поле `source` датасета).
"""
from typing import Dict, List

# Целевые категории C = {CASE, NEG, NUM}
TAG_VOCAB: Dict[str, List[str]] = {
    "CASE": ["NOM", "GEN", "ACC", "DAT", "LOC", "ABL"],
    "NEG":  ["POS", "NEG"],
    "NUM":  ["SG", "PL"],
}
CATEGORIES: List[str] = list(TAG_VOCAB.keys())

# Плоский словарь тегов для multi-label головы g_phi
FLAT_TAGS: List[str] = [f"{c}:{t}" for c in TAG_VOCAB for t in TAG_VOCAB[c]]
NUM_FLAT_TAGS: int = len(FLAT_TAGS)
FLAT_TAG2ID: Dict[str, int] = {t: i for i, t in enumerate(FLAT_TAGS)}

# PAD_ID=0 внутри каждой категории; реальные теги начинаются с 1.
def cat_tag2id(category: str) -> Dict[str, int]:
    """id тега внутри категории (0 зарезервирован под PAD)."""
    return {t: i + 1 for i, t in enumerate(TAG_VOCAB[category])}

CAT_TAG2ID: Dict[str, Dict[str, int]] = {c: cat_tag2id(c) for c in CATEGORIES}

# Веса категорий: приоритет отрицания (см. §5.2/6.1)
CATEGORY_WEIGHT: Dict[str, float] = {"NEG": 4.0, "CASE": 1.5, "NUM": 1.0}
