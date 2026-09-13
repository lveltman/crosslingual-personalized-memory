"""MAC-РАСШИРЕНИЕ (частичная реализация, не в ущерб baseline).

Dual-channel Q-Former + morph-embedder + головы L_morph/L_suffix.
Подключается ПОВЕРХ baseline-памяти.

ПРАВКИ РЕВЬЮ:
  №6  Добавлен L_suffix (§6.2). Класс переименован в PartialMACLoss и в docstring
      явно указано, что это ЧАСТИЧНАЯ реализация §6.3 (L_LM + α·L_morph + β·L_suffix),
      причём L_suffix требует suffix-разметки; при её отсутствии β·L_suffix=0.
  №7  Восстановлен behavioral channel: семантический канал cross-attends к
      behavioral_embs (отдельный источник), морфо-канал — к morph_embs и
      input_embs. Разделение каналов на уровне параметров сохранено; если
      behavioral_embs не передан — используется input_embs с явным предупреждением
      (документированное отступление).
"""
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from morph_tags import (
    TAG_VOCAB, CATEGORIES, FLAT_TAGS, NUM_FLAT_TAGS, CATEGORY_WEIGHT,
)


def _make_safe_key_padding_mask(mask):
    """Открываем нулевую позицию полностью замаскированным строкам (защита от NaN)."""
    if mask is None:
        return None
    all_masked = mask.all(dim=1)
    if all_masked.any():
        mask = mask.clone()
        mask[all_masked, 0] = False
    return mask


class MorphTagEmbedder(nn.Module):
    """Rule-based теги A(Φ(w)) -> эмбеддинги для cross-attention."""
    def __init__(self, hidden_size, tag_vocab=TAG_VOCAB, dropout=0.1):
        super().__init__()
        self.categories = list(tag_vocab.keys())
        self.embeds = nn.ModuleDict({
            c: nn.Embedding(len(tag_vocab[c]) + 1, hidden_size, padding_idx=0)
            for c in self.categories
        })
        self.cat_marker = nn.Parameter(torch.empty(len(self.categories), hidden_size))
        nn.init.normal_(self.cat_marker, std=0.02)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tag_ids: dict):
        summed, any_tag = None, None
        for ci, c in enumerate(self.categories):
            ids = tag_ids[c]
            e = self.embeds[c](ids) + self.cat_marker[ci]
            summed = e if summed is None else summed + e
            has = (ids != 0)
            any_tag = has if any_tag is None else (any_tag | has)
        morph_embs = self.dropout(self.norm(summed))
        morph_mask = ~any_tag   # True = игнорировать
        return morph_embs, morph_mask


class DualChannelQFormer(nn.Module):
    """Двухканальный bottleneck: semantic + morphology слоты.

    ПРАВКА №7: семантический канал cross-attends к BEHAVIORAL источнику
    (behavioral_embs), морфо-канал — к morph_embs и input_embs. Так разделение
    каналов сохраняется на уровне параметров (разные cross-attn модули и разные
    источники ключей/значений).
    """
    def __init__(self, hidden_size, num_sem_queries=4, num_morph_queries=3,
                 num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_sem_queries = num_sem_queries
        self.num_morph_queries = num_morph_queries
        self.sem_queries = nn.Parameter(torch.empty(1, num_sem_queries, hidden_size))
        self.morph_queries = nn.Parameter(torch.empty(1, num_morph_queries, hidden_size))
        nn.init.xavier_uniform_(self.sem_queries)
        nn.init.xavier_uniform_(self.morph_queries)

        mha = lambda: nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        # Behavioral channel восстановлен (правка №7)
        self.cross_attn_beh = mha()          # semantic -> behavioral_embs
        self.cross_attn_morph = mha()        # morph -> morph_embs
        self.cross_attn_morph_inp = mha()    # morph -> input_embs

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm_sem = nn.LayerNorm(hidden_size)
        self.norm_morph1 = nn.LayerNorm(hidden_size)
        self.norm_morph2 = nn.LayerNorm(hidden_size)

    def forward(self, input_embs, morph_embs, behavioral_embs=None,
                input_mask=None, morph_mask=None, behavioral_mask=None):
        B = input_embs.size(0)
        dtype = self.sem_queries.dtype
        input_embs = input_embs.to(dtype)
        morph_embs = morph_embs.to(dtype)

        if behavioral_embs is None:
            # Документированное отступление: за неимением отдельного поведенческого
            # источника используем input_embs (с предупреждением).
            warnings.warn(
                "behavioral_embs не передан — semantic channel использует input_embs "
                "как источник (fallback). Разделение источников каналов ослаблено."
            )
            behavioral_embs = input_embs
            behavioral_mask = input_mask
        else:
            behavioral_embs = behavioral_embs.to(dtype)

        sem_q = self.sem_queries.expand(B, -1, -1)
        morph_q = self.morph_queries.expand(B, -1, -1)

        # SEMANTIC/BEHAVIORAL CHANNEL: cross-attn к behavioral источнику
        safe_beh = _make_safe_key_padding_mask(behavioral_mask)
        q_beh, _ = self.cross_attn_beh(sem_q, behavioral_embs, behavioral_embs,
                                       key_padding_mask=safe_beh)
        sem_q = self.norm_sem(sem_q + q_beh)

        # MORPHOLOGY CHANNEL
        safe_m = _make_safe_key_padding_mask(morph_mask)
        q_m, _ = self.cross_attn_morph(morph_q, morph_embs, morph_embs,
                                       key_padding_mask=safe_m)
        morph_q = self.norm_morph1(morph_q + q_m)
        safe_inp = _make_safe_key_padding_mask(input_mask)
        q_mi, _ = self.cross_attn_morph_inp(morph_q, input_embs, input_embs,
                                            key_padding_mask=safe_inp)
        morph_q = self.norm_morph2(morph_q + q_mi)

        prefix = self.transformer(torch.cat([sem_q, morph_q], dim=1))
        sem_slots = prefix[:, :self.num_sem_queries]
        morph_slots = prefix[:, self.num_sem_queries:]
        return prefix, sem_slots, morph_slots


class PartialMACLoss(nn.Module):
    """ЧАСТИЧНАЯ реализация лосса §6.3: L = L_LM + α·L_morph + β·L_suffix.

    ПРАВКА №6:
      - L_morph (§6.1): взвешенный multi-label BCE поверх пулинга памяти.
      - L_suffix (§6.2): классификация суффиксной категории на позициях
        суффиксов (нужны suffix_mask [B,T] и suffix_cat_id [B,T]). Если
        suffix-разметка не передана — β·L_suffix = 0 (безопасный no-op).
    Класс сознательно назван Partial, чтобы не создавать ложного впечатления
    полноты (замечание ревью №6).
    """
    def __init__(self, alpha=1.0, beta=1.0, num_suffix_classes: int = None):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        tag_weights = [CATEGORY_WEIGHT[t.split(":")[0]] for t in FLAT_TAGS]
        self.register_buffer("tag_weights", torch.tensor(tag_weights, dtype=torch.float32))
        self.gamma = {"NEG": 3.0, "CASE": 2.0, "NUM": 1.0}
        # Голова для L_suffix создаётся лениво снаружи (см. train.build_mac_heads),
        # чтобы знать hidden_size; здесь храним число классов для ассерта.
        self.num_suffix_classes = num_suffix_classes

    def morph_probe_loss(self, memory_logits, y_multi_hot):
        bce = F.binary_cross_entropy_with_logits(
            memory_logits, y_multi_hot.to(memory_logits.dtype), reduction="none")
        # tag_weights приводим к device/dtype логитов (страховка помимо .to(device))
        w = self.tag_weights.to(device=memory_logits.device, dtype=memory_logits.dtype)
        return (bce * w.view(1, -1)).mean()

    def suffix_loss(self, token_hidden, suffix_head, suffix_mask, suffix_cat_id):
        """L_suffix (§6.2): CE по суффиксным позициям.

        token_hidden: [B, T, H] — скрытые представления токенов (например,
            input_embs или hidden из энкодера);
        suffix_head:  nn.Linear(H, num_suffix_classes);
        suffix_mask:  [B, T] bool — True на позициях суффиксов;
        suffix_cat_id:[B, T] long — целевой класс суффикса (валиден там, где mask).
        Возвращает скаляр (0, если суффиксных позиций нет).
        """
        if suffix_mask is None or suffix_cat_id is None:
            return token_hidden.new_zeros(())
        logits = suffix_head(token_hidden)                 # [B, T, C]
        C = logits.size(-1)
        flat_logits = logits.reshape(-1, C)
        flat_targets = suffix_cat_id.reshape(-1)
        flat_mask = suffix_mask.reshape(-1)
        if flat_mask.sum() == 0:
            return token_hidden.new_zeros(())
        sel_logits = flat_logits[flat_mask]
        sel_targets = flat_targets[flat_mask]
        return F.cross_entropy(sel_logits, sel_targets)

    def forward(self, memory, y_multi_hot, morph_head, lm_loss=None,
                token_hidden=None, suffix_head=None,
                suffix_mask=None, suffix_cat_id=None):
        pooled = memory.mean(dim=1)
        logits = morph_head(pooled)
        l_morph = self.morph_probe_loss(logits, y_multi_hot)

        # L_suffix (опционально)
        l_suffix = memory.new_zeros(())
        if suffix_head is not None and token_hidden is not None:
            l_suffix = self.suffix_loss(token_hidden, suffix_head,
                                        suffix_mask, suffix_cat_id)

        total = self.alpha * l_morph + self.beta * l_suffix
        if lm_loss is not None:
            total = lm_loss + total
        return total, {
            "L_LM": lm_loss.detach() if lm_loss is not None else None,
            "L_morph": l_morph.detach(),
            "L_suffix": l_suffix.detach(),
        }
