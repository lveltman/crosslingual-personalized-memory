"""BASELINE: ICAE-подобный компрессор контекста на замороженной Gemma.

Схема:
  1. LoRA-энкодер = сама Gemma (frozen) + обучаемые LoRA-адаптеры;
     к входу приписываем N обучаемых memory-эмбеддингов -> их hidden-состояния
     на выходе энкодера = сжатая память M (N слотов).
  2. Frozen decoder (та же Gemma БЕЗ LoRA) получает M как soft-prefix и
     восстанавливает контекст (LM-loss с labels=-100 на префиксе).

ПРАВКИ РЕВЬЮ:
  №1  decode_loss оборачивается в `with self.model.disable_adapter()`, чтобы
      декодер работал БЕЗ LoRA (соответствие docstring «frozen decoder без LoRA»).
  №2  hidden_size берём из base.config ДО оборачивания в LoRA (не полагаемся
      на делегирование атрибутов PeftModel между версиями peft).
  №3  memory-слоты в encode агрегируются с bidirectional self-attention внутри
      отдельного модуля (документированное соответствие Q-Former self-attention),
      а не только причинно; отступление явно описано ниже.
  №14 dtype прокидывается в __init__ явно через ModelConfig.runtime_dtype.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from config import ModelConfig


def _resolve_dtype(cfg: ModelConfig) -> torch.dtype:
    """Единая логика выбора dtype: runtime_dtype имеет приоритет над dtype."""
    name = cfg.runtime_dtype or cfg.dtype
    return getattr(torch, name)


class ICAECompressor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        dtype = _resolve_dtype(cfg)

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=dtype,
        )
        # ПРАВКА №2: фиксируем hidden_size из base.config ДО обёртки в LoRA.
        self.hidden = base.config.hidden_size

        # Замораживаем базовые веса; LoRA добавит обучаемые адаптеры.
        for p in base.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
        )
        self.model = get_peft_model(base, lora_cfg)  # LoRA = энкодер-адаптер

        self.num_mem = cfg.num_mem_slots

        # Обучаемые memory-эмбеддинги (спец-слоты компрессии, ICAE-стиль)
        self.mem_embeds = nn.Parameter(
            torch.randn(self.num_mem, self.hidden, dtype=dtype) * 0.02
        )

        # ПРАВКА №3: bidirectional self-attention для агрегации memory-слотов.
        # После causal-forward энкодера слоты видят контекст причинно; чтобы
        # приблизить bottleneck к Q-Former self-attention (§2 MAC), пропускаем
        # слоты через небольшой bidirectional TransformerEncoder, где слоты
        # взаимно видят друг друга симметрично.
        # ОТСТУПЛЕНИЕ ОТ MAC-СПЕКИ: сам сбор состояний идёт из causal-LM (baseline),
        # а не из полноценного Q-Former; это документированный компромисс baseline.
        slot_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden, nhead=8,
            dim_feedforward=self.hidden * 4, dropout=0.0, batch_first=True,
        )
        self.slot_aggregator = nn.TransformerEncoder(slot_layer, num_layers=1).to(dtype)

    def get_input_embeddings(self):
        """Единая точка доступа к таблице эмбеддингов (устойчиво к PEFT-обёртке)."""
        return self.model.get_input_embeddings()

    # ---------- ЭНКОДЕР: контекст -> N memory-слотов ----------
    def encode(self, input_ids, attention_mask):
        """Возвращает (M, ctx_hidden).

        M: [B, num_mem, hidden] — сжатая память (memory-слоты).
        ctx_hidden: [B, T, hidden] — КОНТЕКСТУАЛИЗИРОВАННЫЕ представления
            контекстных токенов (behavioral источник для семантического канала
            MAC; отличается от сырых input_embs = таблица эмбеддингов).

        Используем штатный forward causal-LM с output_hidden_states=True и
        берём последний скрытый слой на позициях memory-слотов, затем
        симметрично агрегируем слоты bidirectional self-attention (правка №3).
        """
        B, T = input_ids.size(0), input_ids.size(1)
        tok_emb = self.get_input_embeddings()(input_ids)         # [B, T, H]
        mem = self.mem_embeds.unsqueeze(0).expand(B, -1, -1).to(tok_emb.dtype)
        # Приписываем memory-слоты В КОНЕЦ контекста (они "читают" контекст)
        inputs_embeds = torch.cat([tok_emb, mem], dim=1)         # [B, T+N, H]
        mem_attn = torch.ones(B, self.num_mem, dtype=attention_mask.dtype,
                              device=attention_mask.device)
        attn = torch.cat([attention_mask, mem_attn], dim=1)

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = out.hidden_states[-1]                    # [B, T+N, H]
        M = hidden_states[:, -self.num_mem:, :]                  # память = слоты
        ctx_hidden = hidden_states[:, :T, :]                     # контекстные токены

        # Bidirectional агрегация слотов (симметричное self-attention).
        M = self.slot_aggregator(M.to(self.mem_embeds.dtype))    # [B, N, H]
        return M, ctx_hidden

    # ---------- ДЕКОДЕР: M -> восстановление контекста ----------
    def decode_loss(self, memory, target_ids, target_attn, token_weights=None):
        """LM-loss восстановления контекста из памяти (soft-prefix).

        ПРАВКА №1: декодер работает БЕЗ LoRA — оборачиваем в disable_adapter(),
        чтобы соответствовать заявленному «frozen decoder без LoRA».

        E3 (H3): token_weights [B, L] (по целевым токенам) — перевзвешивает CE на
        субтокенах-маркерах (морфемы отрицания/числа), чтобы reconstruction-objective
        ценил морфологию. Если None — обычная равновесная CE (baseline).
        """
        B, N, _ = memory.shape
        tok_emb = self.get_input_embeddings()(target_ids)        # [B, L, H]
        inputs_embeds = torch.cat([memory.to(tok_emb.dtype), tok_emb], dim=1)

        prefix_attn = torch.ones(B, N, dtype=target_attn.dtype,
                                 device=target_attn.device)
        attn = torch.cat([prefix_attn, target_attn], dim=1)

        # labels: префикс не штрафуется
        prefix_labels = torch.full((B, N), -100, dtype=torch.long,
                                   device=target_ids.device)
        labels = target_ids.clone()
        labels[target_attn == 0] = -100
        labels = torch.cat([prefix_labels, labels], dim=1)

        # Отключаем LoRA-адаптеры на декод-пути (штатный PEFT API).
        with self.model.disable_adapter():
            out = self.model(inputs_embeds=inputs_embeds,
                             attention_mask=attn, use_cache=False)
        logits = out.logits
        if token_weights is None:
            # равновесная CE (как раньше)
            shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            return loss, logits

        # E3: взвешенная CE. Вес по целевым токенам -> в полную сетку [B, N+L].
        w_prefix = torch.ones(B, N, dtype=token_weights.dtype, device=token_weights.device)
        weights = torch.cat([w_prefix, token_weights], dim=1)     # [B, N+L]
        sh_logits = logits[:, :-1]
        sh_labels = labels[:, 1:]
        sh_w = weights[:, 1:]
        tok_loss = F.cross_entropy(
            sh_logits.reshape(-1, sh_logits.size(-1)), sh_labels.reshape(-1),
            ignore_index=-100, reduction="none").view(B, -1)
        valid = (sh_labels != -100).to(tok_loss.dtype)
        ww = sh_w * valid
        loss = (tok_loss * ww).sum() / ww.sum().clamp(min=1.0)
        return loss, logits

    def forward(self, input_ids, attention_mask, token_weights=None):
        """Baseline autoencoding: контекст = цель восстановления.

        token_weights [B, L] (E3) пробрасывается в decode_loss для morpheme-weighted CE.
        """
        M, ctx_hidden = self.encode(input_ids, attention_mask)
        loss, logits = self.decode_loss(M, input_ids, attention_mask,
                                        token_weights=token_weights)
        return {"loss": loss, "logits": logits, "memory": M, "ctx_hidden": ctx_hidden}
