"""Negation-preserving обучение (train-only):
1) marker-weighted CE (вес на субтокенах отрицания/числа);
2) contrastive на минимальных парах: память(pos) != память(neg);
3) fact-slot aux-loss (выделенные слоты держат полярность).
Частично из train.py (E3) и mac_module.py (ablation)."""
