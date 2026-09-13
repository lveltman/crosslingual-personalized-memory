"""PersonalizationDataset: (история юзера, текущий запрос) → таргет.
Условия: en (LaMP, контроль) | ru | uz-latin | uz-cyrillic | code-switch.
Cross-lingual: natural (реальный микс) + controlled (dual-script транслит, контент фиксирован).
Заменяет старый src/data.py (транслитерацию оттуда переиспользуем)."""
