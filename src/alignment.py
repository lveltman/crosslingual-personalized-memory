"""Token-to-morpheme alignment (ответ на уязвимость рецензента №3).

Word-level морфо-теги A(Φ(w)) нужно выровнять по SUBWORD-токенам backbone.

ПРАВКИ РЕВЬЮ №4/№5: раньше мы токенизировали КАЖДОЕ слово отдельно, что не
гарантирует совпадения с токенизацией целого текста (мерджи на границах слов,
особое поведение первого токена в SentencePiece/BPE Gemma). Теперь используем
ЕДИНУЮ токенизацию всего текста через fast-tokenizer и `word_ids()`
(offsets-based alignment). Если fast-tokenizer недоступен — явный fallback с
ассертом/логом на несовпадение длин.
"""
from typing import Dict, List, Optional
import warnings
import torch


def align_word_tags_to_subwords_fast(
    text: str,
    words: List[str],
    word_tag_ids: Dict[str, List[int]],   # категория -> id тега на word
    tokenizer,
    encoding=None,                        # BatchEncoding уже готового вызова (опц.)
    input_ids: Optional[List[int]] = None,
    broadcast: str = "first",             # 'first' | 'all'
) -> Dict[str, List[int]]:
    """Alignment через word_ids() ЕДИНОЙ токенизации текста.

    Требует fast-tokenizer (tokenizer.is_fast == True). Возвращает subword-теги,
    согласованные ровно с input_ids этого же текста.

    ВАЖНО: `words` должны соответствовать `text.split()` — тому же разбиению,
    что использовалось для генерации `word_tag_ids`.
    """
    categories = list(word_tag_ids.keys())
    out: Dict[str, List[int]] = {c: [] for c in categories}

    # Получаем word_ids: для каждого subword — индекс исходного слова или None.
    if encoding is not None and hasattr(encoding, "word_ids"):
        word_ids = encoding.word_ids(batch_index=0)
    else:
        enc = tokenizer(text, add_special_tokens=False)
        word_ids = enc.word_ids()

    prev_word = None
    for wid in word_ids:
        if wid is None:
            # спец-токен / паддинг — тег PAD
            for c in categories:
                out[c].append(0)
            continue
        is_first_subword = (wid != prev_word)
        prev_word = wid
        for c in categories:
            # Защита от рассинхрона длины words и тегов.
            if wid >= len(word_tag_ids[c]):
                out[c].append(0)
                continue
            tag = word_tag_ids[c][wid]
            if broadcast == "all":
                out[c].append(tag)
            else:  # 'first'
                out[c].append(tag if is_first_subword else 0)
    return out


def align_word_tags_to_subwords_slow(
    words: List[str],
    word_tag_ids: Dict[str, List[int]],
    tokenizer,
    broadcast: str = "first",
    add_prefix_space: bool = True,
) -> Dict[str, List[int]]:
    """Fallback: пословная токенизация (может не совпасть с целым текстом!).

    Оставлен только для случая отсутствия fast-tokenizer. Вызов сопровождается
    предупреждением, а вызывающая сторона обязана проверить длину относительно
    реальных input_ids (см. collator).
    """
    categories = list(word_tag_ids.keys())
    out: Dict[str, List[int]] = {c: [] for c in categories}
    for wi, w in enumerate(words):
        prefix = " " if (add_prefix_space and wi > 0) else ""
        sub = tokenizer.encode(prefix + w, add_special_tokens=False)
        n_sub = max(1, len(sub))
        for c in categories:
            tag = word_tag_ids[c][wi] if wi < len(word_tag_ids[c]) else 0
            if broadcast == "all":
                out[c].extend([tag] * n_sub)
            else:
                out[c].extend([tag] + [0] * (n_sub - 1))
    return out


def pad_subword_tags(
    batch_tag_ids: List[Dict[str, List[int]]],
    categories: List[str],
    max_len: int,
    pad_id: int = 0,
) -> Dict[str, torch.LongTensor]:
    """Паддинг subword-тегов до общей длины. -> dict[cat] = [B, T]."""
    B = len(batch_tag_ids)
    out = {c: torch.full((B, max_len), pad_id, dtype=torch.long) for c in categories}
    for bi, tags in enumerate(batch_tag_ids):
        for c in categories:
            seq = tags[c][:max_len]
            out[c][bi, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out
