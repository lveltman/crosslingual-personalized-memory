"""Cross-Lingual Personalized Memory (главный метод).
Скелет: frozen bge-m3 (encoder.py/BT BehavioralEncoder) → Q-Former/слоты (+ fact-слоты)
→ проекция → префикс во frozen мультиязычную Gemma. Переиспользует behavioral_twin.py.
Novelty-компоненты в losses.py. На inference теги НЕ нужны."""
