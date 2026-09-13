# MAC — обучение baseline-компрессора (ICAE) на замороженной Gemma.
# GPU-образ: torch уже в базовом образе, доставляем HF-стек.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache \
    MAC_MODEL_NAME=/models/gemma \
    MAC_DEVICE=cuda

WORKDIR /app

# HF-стек (torch не трогаем — он из базового образа)
RUN pip install --no-cache-dir "transformers>=4.44" "peft>=0.11" "tokenizers>=0.19"

COPY src/ /app/

# По умолчанию — smoke-прогон на синтетике (baseline). Модель монтируется в /models/gemma.
CMD ["python", "train.py", "--allow_synthetic_report", "--n_train", "16", "--n_eval", "8"]
