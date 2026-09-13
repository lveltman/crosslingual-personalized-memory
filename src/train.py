"""Тренировочный цикл BASELINE + опциональный MAC-морфо-канал.

Baseline (обязателен, запускаем всегда): ICAE-компрессия (autoencoding).
MAC (--use_mac): добавляем dual-channel Q-Former, morph-head, L_morph и
опциональный L_suffix.

ПРАВКИ РЕВЬЮ:
  №8  перед evaluate все MAC-подмодули переводятся в .eval(); после — .train().
  №9  отчётные метрики требуют --eval_jsonl (gold). Синтетика — только smoke.
  №14 единая логика dtype: вычисляем dtype с учётом device и прокидываем в
      ModelConfig.runtime_dtype (модель Gemma и MAC-модули согласованы).
"""
import argparse
import os
import warnings
import torch

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from config import Config
from data import CompressionDataset
from collator import CompressionCollator
from architecture.token_level import ICAECompressor
from architecture.mac_module import MorphTagEmbedder, DualChannelQFormer, PartialMACLoss
from morph_tags import NUM_FLAT_TAGS, CAT_TAG2ID
from metrics import MorphFidelityMeter, pretty_print_report

import torch.nn as nn
import torch.nn.functional as F


def build_mac_heads(hidden, device, dtype, num_suffix_classes=8):
    """MAC-скелет поверх baseline-памяти. Все модули на нужном device/dtype."""
    embedder = MorphTagEmbedder(hidden).to(device=device, dtype=dtype)
    qformer = DualChannelQFormer(hidden).to(device=device, dtype=dtype)
    morph_head = nn.Linear(hidden, NUM_FLAT_TAGS).to(device=device, dtype=dtype)
    # Голова для L_suffix (правка №6): классификация суффиксной категории.
    suffix_head = nn.Linear(hidden, num_suffix_classes).to(device=device, dtype=dtype)
    return embedder, qformer, morph_head, suffix_head


def _set_mac_mode(mac, train: bool):
    """Переключает все MAC-подмодули между train/eval (правка №8)."""
    if mac is None:
        return
    for key in ("embedder", "qformer", "morph_head", "suffix_head", "loss"):
        m = mac.get(key)
        if isinstance(m, nn.Module):
            m.train(train)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_mac", action="store_true")
    ap.add_argument("--normalize_to_latin", action="store_true",
                    help="Φ как единая нормализация в латиницу (см. рец. №3)")
    ap.add_argument("--n_train", type=int, default=64)
    ap.add_argument("--n_eval", type=int, default=32)
    ap.add_argument("--train_jsonl", type=str, default=None,
                    help="Путь к train-корпусу (jsonl). Если не задан — синтетика (smoke).")
    ap.add_argument("--eval_jsonl", type=str, default=None,
                    help="Путь к eval-корпусу с GOLD-разметкой. ОБЯЗАТЕЛЕН для отчётных метрик.")
    ap.add_argument("--allow_synthetic_report", action="store_true",
                    help="Разрешить отчёт по синтетике (НЕ для публикации, только отладка).")
    ap.add_argument("--seed", type=int, default=None, help="переопределить сид")
    ap.add_argument("--epochs", type=int, default=None, help="переопределить число эпох")
    ap.add_argument("--probe_train_jsonl", type=str, default=None,
                    help="Отдельный GOLD-сет для обучения probe (если не задан — берётся train_dl).")
    ap.add_argument("--dump_preds", type=str, default=None,
                    help="Путь для дампа по-примерных предсказаний (для bootstrap CI).")
    ap.add_argument("--no_compress", action="store_true",
                    help="КОНТРОЛЬ (upper bound): probe читает НЕсжатые hidden states "
                         "Gemma (masked mean ctx_hidden). Компрессор не обучается.")
    ap.add_argument("--neg_qa_jsonl", type=str, default=None,
                    help="E2: минимальные пары отрицания + yes/no QA. Вместо probe_eval "
                         "меряем downstream: отличает ли модель полярность из сжатой памяти.")
    ap.add_argument("--num_mem_slots", type=int, default=None,
                    help="E1: переопределить число memory-слотов (ось коэффициента сжатия).")
    ap.add_argument("--morph_weight", type=float, default=None,
                    help="E3 (H3): вес CE на субтокенах-маркерах (NEG/PL) при реконструкции. "
                         "None -> равновесная CE (baseline).")
    args = ap.parse_args()

    cfg = Config()
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.epochs is not None:
        cfg.train.num_epochs = args.epochs
    if args.num_mem_slots is not None:
        cfg.model.num_mem_slots = args.num_mem_slots
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    # ПРАВКА №14: единая логика dtype. На CPU принудительно float32.
    dtype_name = cfg.model.dtype if device == "cuda" else "float32"
    cfg.model.runtime_dtype = dtype_name           # прокидываем в модель явно
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(cfg.train.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Данные ---
    if args.train_jsonl:
        train_ds = CompressionDataset.from_jsonl(args.train_jsonl)
    else:
        warnings.warn("train_jsonl не задан — используется СИНТЕТИКА (smoke-test).")
        train_ds = CompressionDataset.synthetic(args.n_train, seed=1)

    if args.eval_jsonl:
        eval_ds = CompressionDataset.from_jsonl(args.eval_jsonl)
    else:
        warnings.warn("eval_jsonl не задан — используется СИНТЕТИКА для eval (smoke).")
        eval_ds = CompressionDataset.synthetic(args.n_eval, seed=2)

    collator = CompressionCollator(
        tokenizer, max_ctx_len=cfg.data.max_ctx_len,
        normalize_to_latin=args.normalize_to_latin)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                          shuffle=True, collate_fn=collator)
    eval_dl = DataLoader(eval_ds, batch_size=cfg.train.batch_size,
                         shuffle=False, collate_fn=collator)

    # Отдельный GOLD-сет для обучения probe (компрессор может учиться на auto,
    # а измеряем морфологию probe'ом, обученным на gold -> без циркулярности).
    probe_train_dl = train_dl
    if args.probe_train_jsonl:
        probe_train_ds = CompressionDataset.from_jsonl(args.probe_train_jsonl)
        probe_train_dl = DataLoader(probe_train_ds, batch_size=cfg.train.batch_size,
                                    shuffle=True, collate_fn=collator)

    # --- Модель baseline ---
    model = ICAECompressor(cfg.model).to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    # --- MAC-надстройка ---
    mac = None
    if args.use_mac:
        hidden = model.hidden
        mac_dtype = params[0].dtype if params else dtype
        embedder, qformer, morph_head, suffix_head = build_mac_heads(
            hidden, device, mac_dtype)
        # ПРАВКА №1 (device для лосса): PartialMACLoss (nn.Module с буфером) -> на device.
        loss_module = PartialMACLoss(alpha=1.0, beta=1.0, num_suffix_classes=8).to(device)
        mac = {"embedder": embedder, "qformer": qformer, "morph_head": morph_head,
               "suffix_head": suffix_head, "loss": loss_module}
        for m in (embedder, qformer, morph_head, suffix_head, loss_module):
            params += [p for p in m.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    total_steps = max(1, len(train_dl) * cfg.train.num_epochs)
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * cfg.train.warmup_ratio), total_steps)

    # --- Тренировка ---
    model.train()
    _set_mac_mode(mac, train=True)
    step = 0
    for epoch in range(0 if args.no_compress else cfg.train.num_epochs):
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            # E3: веса CE на субтокенах-маркерах (отрицание/мн.число) из тегов батча.
            token_weights = None
            if args.morph_weight is not None:
                neg = batch["tag_ids"]["NEG"].to(device) == CAT_TAG2ID["NEG"]["NEG"]
                pl = batch["tag_ids"]["NUM"].to(device) == CAT_TAG2ID["NUM"]["PL"]
                marker = (neg | pl).to(input_ids.dtype).float()
                token_weights = 1.0 + (args.morph_weight - 1.0) * marker

            out = model(input_ids, attn, token_weights=token_weights)  # autoencoding loss
            loss = out["loss"]
            logs = {"L_LM": loss.item()}

            if mac is not None:
                tag_ids = {k: v.to(device) for k, v in batch["tag_ids"].items()}
                y = batch["y_multi_hot"].to(device)
                input_embs = model.get_input_embeddings()(input_ids)
                morph_embs, morph_mask = mac["embedder"](tag_ids)
                input_mask = (attn == 0)
                # Двухпоточность: семантический канал получает КОНТЕКСТУАЛИЗИРОВАННЫЙ
                # источник (ctx_hidden из энкодера), морфо-канал — сырые input_embs
                # + морфо-теги. behavioral_mask совпадает с input_mask (те же токены).
                behavioral_embs = out["ctx_hidden"]
                prefix, _, _ = mac["qformer"](
                    input_embs, morph_embs, behavioral_embs=behavioral_embs,
                    input_mask=input_mask, morph_mask=morph_mask,
                    behavioral_mask=input_mask)
                # L_suffix: suffix-разметки в синтетике нет -> None (β·L_suffix=0).
                total, mlogs = mac["loss"](
                    prefix, y, mac["morph_head"], lm_loss=loss,
                    token_hidden=input_embs, suffix_head=mac["suffix_head"],
                    suffix_mask=None, suffix_cat_id=None)
                loss = total
                logs.update({k: (v.item() if v is not None else None)
                             for k, v in mlogs.items()})

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step(); sched.step()

            if step % cfg.train.log_every == 0:
                print(f"[step {step}] loss={loss.item():.4f} {logs}")
            step += 1

    # --- Оценка: единый post-hoc probe для baseline И MAC (сравнимые абляции) ---
    # ПРАВКА №9: отчёт только на GOLD (eval_jsonl). Синтетика -> отказ, если не разрешено.
    if getattr(eval_ds, "is_synthetic", False) and not args.allow_synthetic_report:
        print("\n[eval] Пропуск отчётных метрик: eval-датасет СИНТЕТИЧЕСКИЙ.\n"
              "       Укажите --eval_jsonl с gold-разметкой либо --allow_synthetic_report\n"
              "       (последнее — только для отладки, НЕ для публикации).")
        return

    if args.neg_qa_jsonl:
        neg_qa_eval(model, tokenizer, args.neg_qa_jsonl, device,
                    no_compress=args.no_compress, dump_preds=args.dump_preds)
        return

    probe_eval(model, mac, probe_train_dl, eval_dl, model.hidden, device,
               dump_preds=args.dump_preds, no_compress=args.no_compress)


@torch.no_grad()
def _recon_logprob(model, prefix_embeds, text_ids, device):
    """Средний logprob токенов text_ids при реконструкции из prefix_embeds.

    prefix_embeds [1,P,H] — сжатая память (baseline) ИЛИ эмбеддинги несжатого
    контекста (потолок). Декодер БЕЗ LoRA (disable_adapter), как в decode_loss.
    Нормируем на длину (тексты пары равной длины, нормировка убирает случайный дрейф).
    """
    emb_layer = model.get_input_embeddings()
    txt = emb_layer(text_ids.unsqueeze(0))
    inp = torch.cat([prefix_embeds, txt], dim=1).to(prefix_embeds.dtype)
    attn = torch.ones(inp.shape[:2], device=device, dtype=torch.long)
    with model.model.disable_adapter():
        out = model.model(inputs_embeds=inp, attention_mask=attn, use_cache=False)
    logits = out.logits[0].float()
    P = prefix_embeds.shape[1]
    ids = text_ids.tolist()
    total = 0.0
    for j, tid in enumerate(ids):
        logp = torch.log_softmax(logits[P + j - 1], dim=-1)
        total += logp[tid].item()
    return total / max(1, len(ids))


def neg_qa_eval(model, tokenizer, neg_qa_jsonl, device, no_compress=False,
                dump_preds=None):
    """E2 (reconstruction forced-choice): сохраняет ли сжатая память ПОЛЯРНОСТЬ.

    Для каждого текста (утв./отр.) сжимаем его -> память M. Сравниваем logprob
    РЕКОНСТРУКЦИИ своего текста vs перевёрнутого по полярности (минимальная пара) из
    той же M. Память сохранила отрицание => своя полярность вероятнее. Это то, для
    чего декодер обучен (autoencoding), без zero-shot промптинга. Метрика:
    polarity-acc = доля, где своя полярность выиграла. no_compress = потолок
    (префикс = эмбеддинги полного контекста источника).
    """
    import json
    model.eval()
    rows = [json.loads(l) for l in open(neg_qa_jsonl, encoding="utf-8")]
    pairs = {}
    for r in rows:
        tid = tokenizer(r["text"], add_special_tokens=False, truncation=True,
                        max_length=256, return_tensors="pt")["input_ids"][0]
        pairs.setdefault(r["pair_id"], {}).setdefault(r["script"], {})[r["polarity"]] = tid

    records = []
    for pid, byscript in pairs.items():
        for script, pol in byscript.items():
            if "pos" not in pol or "neg" not in pol:
                continue
            for src in ("pos", "neg"):
                src_ids = pol[src].to(device)
                if no_compress:
                    prefix = model.get_input_embeddings()(src_ids.unsqueeze(0))
                else:
                    M, _ = model.encode(src_ids.unsqueeze(0),
                                        torch.ones_like(src_ids).unsqueeze(0))
                    prefix = M
                ll_pos = _recon_logprob(model, prefix, pol["pos"].to(device), device)
                ll_neg = _recon_logprob(model, prefix, pol["neg"].to(device), device)
                pred = "neg" if ll_neg > ll_pos else "pos"
                records.append({"pair_id": pid, "script": script, "source": src,
                                "pred": pred, "correct": int(pred == src),
                                "margin_own": (ll_neg - ll_pos) if src == "neg"
                                              else (ll_pos - ll_neg)})

    _report_neg_qa(records, no_compress)
    if dump_preds:
        with open(dump_preds, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        print(f"[dump] neg-recon -> {dump_preds} ({len(records)} строк)")


def _report_neg_qa(records, no_compress):
    tag = "UNCOMPRESSED (потолок)" if no_compress else "COMPRESSED (baseline)"
    print(f"\n### E2 Negation reconstruction forced-choice — {tag} ###")
    for script in ["latin", "cyrillic"]:
        rs = [r for r in records if r["script"] == script]
        if not rs:
            continue
        acc = sum(r["correct"] for r in rs) / len(rs)
        ap = [r["correct"] for r in rs if r["source"] == "pos"]
        an = [r["correct"] for r in rs if r["source"] == "neg"]
        ap = sum(ap) / len(ap) if ap else float("nan")
        an = sum(an) / len(an) if an else float("nan")
        print(f"  {script:9}: polarity-acc={acc:.3f}  (pos={ap:.3f} neg={an:.3f}, n={len(rs)})")


def pooled_memory(model, mac, input_ids, attn, batch, device, no_compress=False):
    """Пулинг СЖАТОЙ памяти [B,H] для текущей конфигурации.

    baseline -> mean(M); MAC -> mean(prefix). Представление отсоединяется:
    probe МЕРЯЕТ, что уже лежит в памяти, и не меняет компрессор.

    no_compress (КОНТРОЛЬ): пул несжатых hidden states Gemma (masked mean ctx_hidden)
    — верхняя граница того, что вообще линейно восстановимо probe'ом.
    """
    M, ctx_hidden = model.encode(input_ids, attn)
    if no_compress:
        m = attn.unsqueeze(-1).to(ctx_hidden.dtype)
        rep = (ctx_hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return rep.detach().float()
    if mac is None:
        rep = M.mean(dim=1)
    else:
        tag_ids = {k: v.to(device) for k, v in batch["tag_ids"].items()}
        input_embs = model.get_input_embeddings()(input_ids)
        morph_embs, morph_mask = mac["embedder"](tag_ids)
        input_mask = (attn == 0)
        prefix, _, _ = mac["qformer"](
            input_embs, morph_embs, behavioral_embs=ctx_hidden,
            input_mask=input_mask, morph_mask=morph_mask,
            behavioral_mask=input_mask)
        rep = prefix.mean(dim=1)
    return rep.detach().float()


def seq_memory(model, mac, input_ids, attn, batch, device, no_compress=False):
    """Возвращает (seq [B,S,H], mask [B,S]) — НЕпулированное представление
    для attention-pool readout. Отсоединено, float.

    no_compress -> контекстные токены ctx_hidden [B,T,H], mask=attn (потолок);
    baseline    -> memory-слоты M [B,N,H], mask=единицы;
    MAC         -> prefix-слоты [B,N,H], mask=единицы.
    """
    M, ctx_hidden = model.encode(input_ids, attn)
    if no_compress:
        seq, mask = ctx_hidden, attn
    elif mac is None:
        seq = M
        mask = torch.ones(seq.shape[:2], device=device, dtype=attn.dtype)
    else:
        tag_ids = {k: v.to(device) for k, v in batch["tag_ids"].items()}
        input_embs = model.get_input_embeddings()(input_ids)
        morph_embs, morph_mask = mac["embedder"](tag_ids)
        input_mask = (attn == 0)
        prefix, _, _ = mac["qformer"](
            input_embs, morph_embs, behavioral_embs=ctx_hidden,
            input_mask=input_mask, morph_mask=morph_mask,
            behavioral_mask=input_mask)
        seq = prefix
        mask = torch.ones(seq.shape[:2], device=device, dtype=attn.dtype)
    return seq.detach().float(), mask.detach().float()


class AttnMLPProbe(nn.Module):
    """Attention-pool (обучаемый query) над слотами/токенами -> 2-слойный MLP.

    Заменяет mean+linear: даёт readout'у выбрать позицию маркера (падеж/число/
    отрицание) вместо усреднения, и нелинейную извлекаемость. Обучается на
    ЗАМОРОЖЕННОЙ памяти — компрессор не меняется, протокол одинаков для всех
    конфигураций, абляции сравнимы.
    """

    def __init__(self, hidden, n_out, p_drop=0.1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(hidden) * (hidden ** -0.5))
        self.scale = hidden ** -0.5
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, n_out))

    def forward(self, seq, mask):
        scores = (seq @ self.q) * self.scale                 # [B,S]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)       # [B,S,1]
        pooled = (w * seq).sum(dim=1)                         # [B,H]
        return self.mlp(pooled)


def probe_eval(model, mac, train_dl, eval_dl, hidden, device, probe_epochs=3,
               dump_preds=None, no_compress=False):
    """Честная post-hoc проба (для абляций).

    Замораживаем компрессор, обучаем СВЕЖУЮ голову g_phi предсказывать морфо-теги
    из ЗАМОРОЖЕННОЙ памяти, затем меряем Morphological Fidelity. Протокол одинаков
    для baseline (ICAE) и MAC, поэтому MF напрямую сравнима между абляциями:
    измеряем, сколько морфологии линейно восстановимо из сжатого представления.
    """
    model.eval()
    _set_mac_mode(mac, train=False)

    probe = AttnMLPProbe(hidden, NUM_FLAT_TAGS).to(device).float()
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)

    # --- обучение пробы на ЗАМОРОЖЕННОЙ памяти ---
    probe.train()
    for _ in range(probe_epochs):
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["y_multi_hot"].to(device).float()
            with torch.no_grad():
                seq, mask = seq_memory(model, mac, input_ids, attn, batch, device,
                                       no_compress=no_compress)
            logits = probe(seq, mask)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

    # --- оценка MF ---
    probe.eval()
    meter = MorphFidelityMeter()
    records = []
    with torch.no_grad():
        for batch in eval_dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["y_multi_hot"].to(device)
            seq, mask = seq_memory(model, mac, input_ids, attn, batch, device,
                                   no_compress=no_compress)
            logits = probe(seq, mask)
            meter.update(logits, y, batch["effective_scripts"], batch["tag_sources"])
            # по-примерный дамп для bootstrap (предсказания + эталон + скрипт)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().tolist()
            ytrue = y.long().cpu().tolist()
            for sc, src, yp, yt in zip(batch["effective_scripts"],
                                       batch["tag_sources"], preds, ytrue):
                records.append({"script": sc, "source": src,
                                "y_pred": yp, "y_true": yt})

    tag = "MAC (dual-channel)" if mac is not None else "ICAE-baseline (single-channel)"
    print(f"\n### Probe Morphological Fidelity — конфигурация: {tag} ###")
    pretty_print_report(meter.report())

    if dump_preds:
        import json
        with open(dump_preds, "w", encoding="utf-8") as f:
            json.dump(records, f)
        print(f"[dump] по-примерные предсказания -> {dump_preds} ({len(records)} строк)")

    model.train()
    _set_mac_mode(mac, train=True)


if __name__ == "__main__":
    main()
