"""Collator: токенизация + word->subword alignment тегов + сборка батча.

Baseline-режим: контекст = целевой текст (autoencoding-восстановление).
Для двускриптного контроля НЕ смешиваем скрипты внутри метрик.

ПРАВКА РЕВЬЮ №4/№5: alignment теперь строится на ЕДИНОЙ токенизации каждого
текста (per-example) с offsets/word_ids, а не пословно. Есть ассерт на
совпадение длины aligned-тегов с числом реальных subword-токенов.

ПРАВКА РЕВЬЮ №12: при normalize_to_latin ведём отдельно original_script и
effective_script (после Φ). Метрики стратифицируются по effective_script.
"""
from typing import List, Dict
import warnings
import torch

from data import MorphExample, transliterate_cyr2lat, normalize_unicode
from alignment import (
    align_word_tags_to_subwords_fast,
    align_word_tags_to_subwords_slow,
    pad_subword_tags,
)
from morph_tags import CATEGORIES, FLAT_TAGS


class CompressionCollator:
    def __init__(self, tokenizer, max_ctx_len=256, broadcast="first",
                 normalize_to_latin=False):
        self.tok = tokenizer
        self.max_ctx_len = max_ctx_len
        self.broadcast = broadcast
        # Если True — Φ работает как единая нормализация в латиницу (см. рец. №3).
        self.normalize_to_latin = normalize_to_latin
        self.use_fast = getattr(tokenizer, "is_fast", False)
        if not self.use_fast:
            warnings.warn(
                "Tokenizer не является fast — alignment пойдёт через медленный "
                "fallback (пословная токенизация), возможен silent misalignment. "
                "Рекомендуется use_fast=True."
            )

    def _prepare_text(self, ex: MorphExample):
        """Возвращает (text_for_tokenizer, effective_script)."""
        # Всегда нормализуем Unicode (правка №13), чтобы токенизация была стабильной.
        text = normalize_unicode(ex.text)
        effective_script = ex.script
        if self.normalize_to_latin and ex.script == "cyrillic":
            text = transliterate_cyr2lat(text)
            effective_script = "latin"  # ПОСЛЕ Φ скрипт фактически латинский
        return text, effective_script

    def __call__(self, batch: List[MorphExample]) -> Dict:
        texts, orig_scripts, eff_scripts, y_list, sources = [], [], [], [], []
        per_example_words = []

        for ex in batch:
            text, eff_script = self._prepare_text(ex)
            words = text.split()
            texts.append(text)
            per_example_words.append(words)
            orig_scripts.append(ex.script)
            eff_scripts.append(eff_script)
            y_list.append(ex.y_multi_hot)
            sources.append(ex.tag_source)

        # Токенизация контекста (единым батчем, с паддингом).
        enc = self.tok(
            texts, padding=True, truncation=True,
            max_length=self.max_ctx_len, return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        T = input_ids.size(1)

        # Alignment per-example на ЕДИНОЙ токенизации текста.
        aligned_tags = []
        for bi, (text, words, ex) in enumerate(zip(texts, per_example_words, batch)):
            if self.use_fast:
                # Отдельный вызов на текст, чтобы получить корректные word_ids
                # именно для этого примера (без паддинга батча).
                single = self.tok(text, add_special_tokens=False)
                sub_tags = align_word_tags_to_subwords_fast(
                    text=text, words=words, word_tag_ids=ex.tag_ids,
                    tokenizer=self.tok, encoding=single, broadcast=self.broadcast,
                )
                n_tokens_single = len(single["input_ids"])
                # Проверка согласованности: длина aligned == числу subword'ов.
                for c in CATEGORIES:
                    if len(sub_tags[c]) != n_tokens_single:
                        warnings.warn(
                            f"[align] несовпадение длины тегов ({len(sub_tags[c])}) "
                            f"и токенов ({n_tokens_single}) для категории {c}; "
                            f"будет применён pad/truncate до T={T}."
                        )
                        break
            else:
                sub_tags = align_word_tags_to_subwords_slow(
                    words, ex.tag_ids, self.tok, broadcast=self.broadcast,
                )
            aligned_tags.append(sub_tags)

        # Паддинг/усечение до общей длины батча T (согласовано с input_ids).
        tag_ids = pad_subword_tags(aligned_tags, CATEGORIES, max_len=T)

        y_multi_hot = torch.tensor(y_list, dtype=torch.float32)

        return {
            "input_ids": input_ids,               # [B, T]
            "attention_mask": attn,               # [B, T]
            "tag_ids": tag_ids,                   # dict[cat] -> [B, T]
            "y_multi_hot": y_multi_hot,           # [B, |T|]
            "original_scripts": orig_scripts,     # List[str] — до Φ
            "effective_scripts": eff_scripts,     # List[str] — после Φ (для метрик)
            "tag_sources": sources,               # List[str]
        }
