"""Харнесс метрики Morphological Fidelity + Critical Flip Rate.

Morphological Fidelity: качество морфо-тегов эталона, предсказанных
пробинг-головой g_phi поверх памяти M. Считается ПО КАТЕГОРИЯМ (CASE/NEG/NUM)
и с разбивкой по (effective_script, tag_source).

ПРАВКА РЕВЬЮ №11: помимо recall теперь считаем TP/FP/FN/TN по каждому тегу,
что позволяет строить precision/recall/F1 (требование рецензента).

Critical Flip Rate: доля примеров, где предсказание перевернуло полярность
NEG относительно эталона (POS<->NEG) — primary metric для инверсии смысла.

ВАЖНО (уязвимость №1): если tag_source == 'auto', метрика измеряет согласие
с шумным учителем, а не gold. Мы это логируем отдельно и НЕ смешиваем
gold/auto при агрегации.
"""
from collections import defaultdict
from typing import Dict, List

import torch

from morph_tags import CATEGORIES, FLAT_TAGS, TAG_VOCAB, FLAT_TAG2ID


def _category_slice(category: str):
    """Индексы FLAT_TAGS, относящиеся к категории."""
    return [FLAT_TAG2ID[f"{category}:{t}"] for t in TAG_VOCAB[category]]


class MorphFidelityMeter:
    """Аккумулятор метрик с разбивкой по (script, category, tag_source).

    Для каждого (script, source, tag) храним confusion-счётчики:
      tp, fp, fn, tn — для полноценных precision/recall/F1 (правка №11).
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        # (script, source, flat_tag) -> dict(tp,fp,fn,tn)
        self.conf = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        # Critical Flip Rate: (script, source) -> [flips, total]
        self.flip = defaultdict(lambda: [0, 0])

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y_multi_hot: torch.Tensor,
               scripts: List[str], sources: List[str]):
        """logits/y: [B, |T|]. Confusion по каждому бинарному тегу."""
        preds = (torch.sigmoid(logits) > self.threshold).long().cpu()
        y = y_multi_hot.long().cpu()
        B = preds.size(0)

        neg_idx = _category_slice("NEG")  # [POS_id, NEG_id]
        for b in range(B):
            sc, src = scripts[b], sources[b]
            for cat in CATEGORIES:
                for i in _category_slice(cat):
                    flat_tag = FLAT_TAGS[i]
                    key = (sc, src, flat_tag)
                    yt = int(y[b, i]); yp = int(preds[b, i])
                    if yt == 1 and yp == 1:
                        self.conf[key]["tp"] += 1
                    elif yt == 0 and yp == 1:
                        self.conf[key]["fp"] += 1
                    elif yt == 1 and yp == 0:
                        self.conf[key]["fn"] += 1
                    else:
                        self.conf[key]["tn"] += 1
            # Critical Flip Rate по NEG
            neg_true = y[b, neg_idx].tolist()      # [pos, neg]
            neg_pred = preds[b, neg_idx].tolist()
            self.flip[(sc, src)][1] += 1
            true_is_neg = neg_true[1] == 1
            pred_is_neg = neg_pred[1] == 1
            if true_is_neg != pred_is_neg:
                self.flip[(sc, src)][0] += 1

    @staticmethod
    def _prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        if prec != prec or rec != rec or (prec + rec) == 0:
            f1 = float("nan")
        else:
            f1 = 2 * prec * rec / (prec + rec)
        return prec, rec, f1

    def report(self) -> Dict:
        out = {"per_tag": {}, "per_category": {}, "critical_flip_rate": {}}

        # Агрегация по категориям (микро-усреднение confusion внутри категории).
        cat_agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

        for (sc, src, flat_tag), c in sorted(self.conf.items()):
            prec, rec, f1 = self._prf(c["tp"], c["fp"], c["fn"])
            out["per_tag"][f"{sc}/{src}/{flat_tag}"] = {
                "precision": prec, "recall": rec, "f1": f1, **c,
            }
            cat = flat_tag.split(":")[0]
            k = (sc, src, cat)
            for m in ("tp", "fp", "fn", "tn"):
                cat_agg[k][m] += c[m]

        for (sc, src, cat), c in sorted(cat_agg.items()):
            prec, rec, f1 = self._prf(c["tp"], c["fp"], c["fn"])
            out["per_category"][f"{sc}/{src}/{cat}"] = {
                "precision": prec, "recall": rec, "f1": f1, **c,
            }

        for (sc, src), (f, t) in sorted(self.flip.items()):
            out["critical_flip_rate"][f"{sc}/{src}"] = f / t if t else float("nan")
        return out


def pretty_print_report(report: Dict):
    print("\n=== Morphological Fidelity — per category (script/source) ===")
    for k, v in report["per_category"].items():
        print(f"  {k:28s}: P={v['precision']:.4f} R={v['recall']:.4f} "
              f"F1={v['f1']:.4f} (tp={v['tp']}, fp={v['fp']}, fn={v['fn']})")
    print("=== Critical Flip Rate (NEG, script/source) ===")
    for k, v in report["critical_flip_rate"].items():
        print(f"  {k:28s}: {v:.4f}")
