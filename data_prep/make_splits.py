"""Шаг 5: author-disjoint train/dev/test, стратификация бакет × язык×письмо.
- test ~500 юзеров (+ отдельный cold-start блок), dev ~1-2k, train = остальной ≥3 cohort;
- train: rolling пары (история 1..k-1 → сессия k), cap ≤5 таргетов/юзер;
Выход: data/splits/{train,dev,test}.jsonl."""
