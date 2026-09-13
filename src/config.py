"""Конфигурация проекта: baseline ICAE + MAC-скелет."""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ModelConfig:
    # Базовая модель (decoder-only). Переопределяется env MAC_MODEL_NAME
    # (HF id или путь к локальным весам). Для теста — маленькая модель.
    model_name: str = field(
        default_factory=lambda: os.environ.get("MAC_MODEL_NAME", "google/gemma-2-2b")
    )
    # Число memory-слотов компрессора (ICAE-стиль)
    num_mem_slots: int = 16
    # LoRA-параметры энкодера
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # На какие модули вешаем LoRA (q/v проекции внимания)
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    # dtype в виде строки; фактический torch.dtype вычисляется в train.py с учётом device
    # (правка ревью №14: единая логика fallback dtype прокидывается сюда явно).
    dtype: str = "bfloat16"
    # Если задан явный runtime_dtype (например, "float32" на CPU), он имеет приоритет.
    runtime_dtype: Optional[str] = None


@dataclass
class DataConfig:
    max_ctx_len: int = 256      # длина контекста для сжатия
    max_tgt_len: int = 128      # длина восстанавливаемого/целевого текста
    # Двускриптный контроль: возможные скрипты входа
    scripts: List[str] = field(default_factory=lambda: ["latin", "cyrillic"])


@dataclass
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 2
    num_epochs: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 10
    seed: int = 42
    device: str = field(default_factory=lambda: os.environ.get("MAC_DEVICE", "cuda"))


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
