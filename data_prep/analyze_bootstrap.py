"""Агрегация мощности: по-примерные дампы (preds_{config}_s{seed}.json) ->
per-script F1 по категориям + Critical Flip Rate, среднее±sd по сидам и
bootstrap-CI на разрыв latin-cyrillic.

Использование:
    python data_prep/analyze_bootstrap.py results/preds_*.json
"""
import glob
import json
import os
import re
import sys
import statistics as st

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from morph_tags import TAG_VOCAB, CATEGORIES, FLAT_TAGS, FLAT_TAG2ID  # noqa: E402

CAT_IDX = {c: [FLAT_TAG2ID[f"{c}:{t}"] for t in TAG_VOCAB[c]] for c in CATEGORIES}
NEG_NEG = FLAT_TAG2ID["NEG:NEG"]


def cat_f1(recs, script, cat):
    idx = CAT_IDX[cat]
    tp = fp = fn = 0
    for r in recs:
        if r["script"] != script:
            continue
        for i in idx:
            yt, yp = r["y_true"][i], r["y_pred"][i]
            tp += (yt == 1 and yp == 1)
            fp += (yt == 0 and yp == 1)
            fn += (yt == 1 and yp == 0)
    if tp == 0:
        return 0.0
    p, r_ = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r_ / (p + r_) if (p + r_) else 0.0


def flip_rate(recs, script):
    tot = flips = 0
    for r in recs:
        if r["script"] != script:
            continue
        tot += 1
        flips += (r["y_true"][NEG_NEG] != r["y_pred"][NEG_NEG])
    return flips / tot if tot else float("nan")


def avg_f1(recs, script):
    return sum(cat_f1(recs, script, c) for c in CATEGORIES) / len(CATEGORIES)


def bootstrap_gap(recs, B=2000, seed=0):
    """CI на разрыв (latin avg-F1 − cyrillic avg-F1) ресэмплом примеров."""
    rng = np.random.default_rng(seed)
    n = len(recs)
    gaps = []
    for _ in range(B):
        samp = [recs[i] for i in rng.integers(0, n, n)]
        gaps.append(avg_f1(samp, "latin") - avg_f1(samp, "cyrillic"))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(np.mean(gaps)), float(lo), float(hi)


# CVD-безопасная пара (синий/оранжевый) для скриптов
SCRIPT_COLOR = {"latin": "#4C78A8", "cyrillic": "#F58518"}


def make_figures(summary, out_dir="results/figures"):
    """Публикационные фигуры из посидовых метрик (mean±sd по сидам)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    cfgs = list(summary.keys())
    scripts = ["latin", "cyrillic"]

    # Фигура 1: MF по категориям, панель на конфиг, серии = скрипт, error=sd по сидам
    fig, axes = plt.subplots(1, len(cfgs), figsize=(5 * len(cfgs), 4), sharey=True)
    if len(cfgs) == 1:
        axes = [axes]
    x = np.arange(len(CATEGORIES)); w = 0.38
    for ax, cfg in zip(axes, cfgs):
        for k, sc in enumerate(scripts):
            means = [np.mean(summary[cfg][f"{c}_{sc[:3]}"]) for c in CATEGORIES]
            sds = [np.std(summary[cfg][f"{c}_{sc[:3]}"]) for c in CATEGORIES]
            ax.bar(x + (k - 0.5) * w, means, w, yerr=sds, capsize=3,
                   color=SCRIPT_COLOR[sc], label=sc, edgecolor="white", linewidth=0.5)
        ax.set_title(cfg); ax.set_xticks(x); ax.set_xticklabels(CATEGORIES)
        ax.set_ylim(0, 1); ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Morphological Fidelity (F1)")
    axes[-1].legend(title="script", frameon=False)
    fig.suptitle("Morphological Fidelity by script (mean±sd across seeds)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/mf_by_script.png", dpi=150, bbox_inches="tight")

    # Фигура 2: Critical Flip Rate (отрицание) по конфигам и скриптам
    fig2, ax = plt.subplots(figsize=(5, 4))
    xc = np.arange(len(cfgs))
    for k, sc in enumerate(scripts):
        means = [np.mean(summary[cfg][f"flip_{sc[:3]}"]) for cfg in cfgs]
        sds = [np.std(summary[cfg][f"flip_{sc[:3]}"]) for cfg in cfgs]
        ax.bar(xc + (k - 0.5) * w, means, w, yerr=sds, capsize=3,
               color=SCRIPT_COLOR[sc], label=sc, edgecolor="white", linewidth=0.5)
    ax.set_xticks(xc); ax.set_xticklabels(cfgs); ax.set_ylim(0, 1)
    ax.set_ylabel("Critical Flip Rate (NEG) ↓")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.3)
    ax.legend(title="script", frameon=False)
    fig2.suptitle("Negation polarity flips (lower is better)")
    fig2.tight_layout()
    fig2.savefig(f"{out_dir}/critical_flip_rate.png", dpi=150, bbox_inches="tight")
    print(f"\n[figures] -> {out_dir}/mf_by_script.png, {out_dir}/critical_flip_rate.png")


def main():
    files = []
    for pat in sys.argv[1:]:
        files += glob.glob(pat)
    if not files:
        print("нет файлов"); return

    by_cfg = {}
    for f in files:
        m = re.search(r"preds_(\w+?)_s(\d+)\.json", os.path.basename(f))
        cfg = m.group(1) if m else os.path.basename(f)
        by_cfg.setdefault(cfg, []).append(f)

    summary = {}
    for cfg, fs in sorted(by_cfg.items()):
        print(f"\n{'='*60}\nКОНФИГ: {cfg}  ({len(fs)} сидов)\n{'='*60}")
        seed_rows = {"lat": [], "cyr": [], "gap": [], "flip_lat": [], "flip_cyr": [],
                     **{f"{c}_lat": [] for c in CATEGORIES},
                     **{f"{c}_cyr": [] for c in CATEGORIES}}
        pooled = []
        for f in sorted(fs):
            recs = json.load(open(f))
            pooled += recs
            seed_rows["lat"].append(avg_f1(recs, "latin"))
            seed_rows["cyr"].append(avg_f1(recs, "cyrillic"))
            seed_rows["gap"].append(avg_f1(recs, "latin") - avg_f1(recs, "cyrillic"))
            seed_rows["flip_lat"].append(flip_rate(recs, "latin"))
            seed_rows["flip_cyr"].append(flip_rate(recs, "cyrillic"))
            for c in CATEGORIES:
                seed_rows[f"{c}_lat"].append(cat_f1(recs, "latin", c))
                seed_rows[f"{c}_cyr"].append(cat_f1(recs, "cyrillic", c))

        def ms(k):
            xs = seed_rows[k]
            sd = st.stdev(xs) if len(xs) > 1 else 0.0
            return f"{st.mean(xs):.3f}±{sd:.3f}"

        print("avg-F1  latin:", ms("lat"), " cyrillic:", ms("cyr"))
        for c in CATEGORIES:
            print(f"  {c:5} latin: {ms(c+'_lat')}   cyrillic: {ms(c+'_cyr')}")
        print("Critical Flip Rate  latin:", ms("flip_lat"), " cyrillic:", ms("flip_cyr"))
        print("РАЗРЫВ (latin−cyrillic) по сидам:", ms("gap"))
        gmean, lo, hi = bootstrap_gap(pooled)
        sig = "ЗНАЧИМ (CI не содержит 0)" if (lo > 0 or hi < 0) else "НЕ значим (CI содержит 0)"
        print(f"bootstrap-CI разрыва (pooled): {gmean:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {sig}")
        summary[cfg] = seed_rows

    try:
        make_figures(summary)
    except Exception as e:
        print(f"[figures] пропущено: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
