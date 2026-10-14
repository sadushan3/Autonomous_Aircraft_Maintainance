"""Read-only EDA over the C-MAPSS raw files. Writes PNGs + a stats JSON.

Touches nothing in the project: reads Backend/data/raw/*.txt and writes only
into this scratchpad directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

RAW = Path("D:/Anomaly_Health_Monitering_System/Backend/data/raw")
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)

COLS = ["unit", "cycle"] + [f"op_setting_{i}" for i in (1, 2, 3)] + [f"sensor_{i}" for i in range(1, 22)]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
OPS = [f"op_setting_{i}" for i in (1, 2, 3)]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# ---- validated palettes (see dataviz validator runs) ----------------------
THEMES = {
    "light": dict(
        surface="#ffffff", ink="#111a1e", body="#2c383e", muted="#61737b",
        grid="#d5dde0", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        seq="#2a78d6", div_lo="#2a78d6", div_mid="#f0efec", div_hi="#e34948",
        accent="#0c6d77", copper="#96501f",
    ),
    "dark": dict(
        surface="#141d21", ink="#e9eef0", body="#bfcbd0", muted="#7e8f97",
        grid="#2b3a40", series=["#3987e5", "#d95926", "#199e70", "#c98500"],
        seq="#3987e5", div_lo="#3987e5", div_mid="#383835", div_hi="#e66767",
        accent="#4bbcc6", copper="#d4915b",
    ),
}


def read(split: str, sub: str) -> pd.DataFrame:
    d = pd.read_csv(RAW / f"{split}_{sub}.txt", sep=r"\s+", header=None)
    d = d.dropna(axis=1, how="all")
    d.columns = COLS
    return d


def add_train_rul(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["RUL"] = d.groupby("unit")["cycle"].transform("max") - d["cycle"]
    return d


def style(ax, t, xlabel="", ylabel="", title="", grid_axis="y"):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["grid"])
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=t["muted"], labelsize=9, length=3, width=0.9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.7, alpha=0.85)
        ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=t["muted"], fontsize=9.5, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["muted"], fontsize=9.5, labelpad=8)
    if title:
        ax.set_title(title, color=t["ink"], fontsize=12.5, fontweight="600",
                     loc="left", pad=12)


def save(fig, name, t, mode):
    fig.savefig(OUT / f"{name}_{mode}.png", dpi=155, facecolor=t["surface"],
                bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)


# =========================================================================
print("loading...")
train = {s: add_train_rul(read("train", s)) for s in SUBSETS}
test = {s: read("test", s) for s in SUBSETS}
rul_files = {s: pd.read_csv(RAW / f"RUL_{s}.txt", sep=r"\s+", header=None).iloc[:, 0].to_numpy()
             for s in SUBSETS}

stats = {"subsets": {}, "quality": {}, "sensors": {}}
for s in SUBSETS:
    tr, te = train[s], test[s]
    lives = tr.groupby("unit")["cycle"].max()
    stats["subsets"][s] = dict(
        train_rows=int(len(tr)), train_engines=int(tr.unit.nunique()),
        test_rows=int(len(te)), test_engines=int(te.unit.nunique()),
        life_min=int(lives.min()), life_med=float(lives.median()),
        life_max=int(lives.max()), life_std=float(lives.std()),
        constant_sensors=int(sum(tr[c].std() < 1e-6 for c in SENSORS)),
        regimes=int(len(tr[OPS].round({"op_setting_1": 0, "op_setting_2": 2,
                                       "op_setting_3": 0}).drop_duplicates())),
        rul_at_end_mean=float(rul_files[s].mean()),
        rul_at_end_min=int(rul_files[s].min()),
        rul_at_end_max=int(rul_files[s].max()),
    )

allrows = sum(v["train_rows"] + v["test_rows"] for v in stats["subsets"].values())
stats["quality"] = dict(
    total_rows=allrows,
    total_engines=sum(v["train_engines"] + v["test_engines"] for v in stats["subsets"].values()),
    missing_cells=int(sum(train[s][COLS].isna().sum().sum() + test[s][COLS].isna().sum().sum()
                          for s in SUBSETS)),
    duplicate_keys=int(sum(train[s].duplicated(["unit", "cycle"]).sum() +
                           test[s].duplicated(["unit", "cycle"]).sum() for s in SUBSETS)),
    non_finite=int(sum((~np.isfinite(train[s][SENSORS].to_numpy())).sum() for s in SUBSETS)),
)

pool = pd.concat([train[s].assign(subset=s) for s in SUBSETS], ignore_index=True)
for c in SENSORS:
    stats["sensors"][c] = dict(
        std_fd001=float(train["FD001"][c].std()),
        std_fd002=float(train["FD002"][c].std()),
        std_pooled=float(pool[c].std()),
        corr_rul_fd001=float(train["FD001"][c].corr(train["FD001"]["RUL"])),
        n_unique_fd001=int(train["FD001"][c].nunique()),
    )
print("stats done")

# =========================================================================
# FIG 1 - sensor dispersion, FD001 vs FD002
# =========================================================================
for mode, t in THEMES.items():
    fig, ax = plt.subplots(figsize=(9.4, 8.2))
    fig.patch.set_facecolor(t["surface"])
    y = np.arange(21)
    a = np.array([max(stats["sensors"][c]["std_fd001"], 1e-5) for c in SENSORS])
    b = np.array([stats["sensors"][c]["std_fd002"] for c in SENSORS])
    ax.axhspan(5.5, 6.5, color=t["copper"], alpha=0.10, zorder=0)
    ax.barh(y - 0.20, a, height=0.34, color=t["series"][0], label="FD001 - one flight condition", zorder=3)
    ax.barh(y + 0.20, b, height=0.34, color=t["series"][1], label="FD002 - six flight conditions", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(8e-6, 1400)
    ax.set_yticks(y, [f"s{i}" for i in range(1, 22)])
    ax.set_ylim(20.7, -0.7)
    style(ax, t, xlabel="standard deviation across the whole training fleet  (log scale)", grid_axis="x")
    for i, c in enumerate(SENSORS):
        if stats["sensors"][c]["std_fd001"] < 1e-6:
            ax.text(1.15e-5, i - 0.20, "  flat", va="center", ha="left",
                    fontsize=8.5, color=t["muted"], style="italic", zorder=4)
    ax.text(1350, 6, "164x", va="center", ha="right", fontsize=10.5,
            fontweight="600", color=t["copper"], zorder=4)
    leg = ax.legend(frameon=False, fontsize=10.5, loc="lower left",
                    bbox_to_anchor=(0, 1.005), ncol=2, handlelength=1.3,
                    columnspacing=2.2, borderaxespad=0)
    for txt in leg.get_texts():
        txt.set_color(t["body"])
    fig.suptitle("Sensor dispersion collapses when the engine holds one flight condition",
                 color=t["ink"], fontsize=12.5, fontweight="600", x=0.125, y=1.045, ha="left")
    save(fig, "f1_dispersion", t, mode)

# =========================================================================
# FIG 2 — engine lifetime distribution per subset
# =========================================================================
for mode, t in THEMES.items():
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    fig.patch.set_facecolor(t["surface"])
    rng = np.random.default_rng(42)
    for i, s in enumerate(SUBSETS):
        lives = train[s].groupby("unit")["cycle"].max().to_numpy()
        col = t["series"][i]
        ax.scatter(lives, np.full_like(lives, i, dtype=float) + rng.uniform(-0.16, 0.16, lives.size),
                   s=13, color=col, alpha=0.42, linewidths=0)
        q1, med, q3 = np.percentile(lives, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=col, lw=5.5, solid_capstyle="butt", alpha=0.95)
        ax.plot([med, med], [i - 0.26, i + 0.26], color=t["surface"], lw=2.6)
        ax.text(586, i, f"median {med:.0f}", va="center", ha="right",
                fontsize=9.5, color=t["body"], fontweight="600")
    ax.set_yticks(range(4), [f"{s}\n{stats['subsets'][s]['train_engines']} engines" for s in SUBSETS])
    ax.invert_yaxis()
    ax.set_xlim(90, 600)
    style(ax, t, xlabel="engine lifetime  (cycles to failure)", grid_axis="x")
    ax.set_title("Every training engine runs to failure — but lifetime varies 4× across the fleet",
                 color=t["ink"], fontsize=12.5, fontweight="600", loc="left", pad=14)
    ax.text(0.995, -0.20, "one dot = one engine   ·   bar = interquartile range",
            transform=ax.transAxes, ha="right", fontsize=9, color=t["muted"])
    save(fig, "f2_lifetime", t, mode)

# =========================================================================
# FIG 3 - the six operating regimes (position carries identity; single hue)
# =========================================================================
for mode, t in THEMES.items():
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    fig.patch.set_facecolor(t["surface"])
    d2 = train["FD002"]
    ax = axes[0]
    ax.scatter(d2.op_setting_1, d2.op_setting_2, s=95, color=t["seq"], alpha=0.9, linewidths=0)
    style(ax, t, xlabel="op_setting_1  -  altitude (kft)",
          ylabel="op_setting_2  -  Mach number", grid_axis="both")
    ax.set_title("53,759 rows collapse into six points", color=t["ink"],
                 fontsize=11.5, fontweight="600", loc="left", pad=10)
    cent = d2[OPS].round({"op_setting_1": 0, "op_setting_2": 2, "op_setting_3": 0}).drop_duplicates()
    cent = cent.sort_values("op_setting_1").reset_index(drop=True)
    for i, r in cent.iterrows():
        ax.annotate(str(i + 1), (r.op_setting_1, r.op_setting_2), xytext=(11, 8),
                    textcoords="offset points", fontsize=11, fontweight="600", color=t["copper"])
    ax.set_xlim(-5, 50)
    ax.set_ylim(-0.09, 0.99)

    ax = axes[1]
    sel = d2[(d2.op_setting_1 > 41) & (d2.op_setting_1 < 43)]
    ax.scatter(sel.op_setting_1, sel.op_setting_2, s=9, color=t["seq"], alpha=0.30, linewidths=0)
    style(ax, t, xlabel="op_setting_1  -  altitude (kft)", ylabel="", grid_axis="both")
    ax.set_title("cluster 6, zoomed in", color=t["ink"],
                 fontsize=11.5, fontweight="600", loc="left", pad=34)
    note = ("within-cluster spread\n"
            "+/-%.4f kft   +/-%.4f Mach\n"
            "pure measurement noise" % (sel.op_setting_1.std(), sel.op_setting_2.std()))
    ax.text(0.5, 1.10, note.replace(chr(10), "   "), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9.5, color=t["copper"])
    fig.suptitle("Six tight, well-separated clusters - which is why KMeans is the right tool",
                 color=t["ink"], fontsize=12.5, fontweight="600", x=0.055, y=1.03, ha="left")
    fig.tight_layout()
    save(fig, "f3_regimes", t, mode)

# =========================================================================
# FIG 4 - degradation signal: rising vs falling channels
# =========================================================================
RISE = ["sensor_11", "sensor_4", "sensor_15", "sensor_2", "sensor_17",
        "sensor_3", "sensor_8", "sensor_13", "sensor_9", "sensor_14", "sensor_6"]
FALL = ["sensor_12", "sensor_7", "sensor_21", "sensor_20"]
for mode, t in THEMES.items():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    fig.patch.set_facecolor(t["surface"])
    d = train["FD001"]
    d = d[d.RUL <= 200].copy()
    bins = np.arange(0, 205, 5)
    d["bin"] = pd.cut(d.RUL, bins, labels=bins[:-1] + 2.5)
    panels = [
        (axes[0], RISE, "sensor_11", "s11  HPC static pressure", 0,
         "%d channels RISE as the engine degrades" % len(RISE)),
        (axes[1], FALL, "sensor_12", "s12  fuel flow / Ps30", 1,
         "%d channels FALL" % len(FALL)),
    ]
    for ax, group, star, lbl, hue, head in panels:
        for c in group:
            g = d.groupby("bin", observed=True)[c].mean()
            v = (g - g.mean()) / g.std()
            ax.plot(g.index.astype(float), v, lw=1.1, color=t["series"][hue], alpha=0.28, zorder=2)
        g = d.groupby("bin", observed=True)[star].mean()
        v = (g - g.mean()) / g.std()
        x = g.index.astype(float)
        ax.plot(x, v, lw=2.4, color=t["series"][hue], zorder=4, solid_capstyle="round")
        ax.scatter([x[-1]], [v.iloc[-1]], s=40, color=t["series"][hue], zorder=5,
                   edgecolors=t["surface"], linewidths=1.8)
        ax.annotate(lbl, (x[-1], v.iloc[-1]), xytext=(12, 15), textcoords="offset points",
                    ha="left", va="bottom", fontsize=9.5, color=t["body"], fontweight="600")
        ax.invert_xaxis()
        ax.set_xlim(232, -12)
        ax.axvline(130, color=t["accent"], lw=1.2, ls=(0, (4, 3)), zorder=1)
        style(ax, t, xlabel="remaining useful life (cycles)  -  failure at the right")
        ax.set_title(head, color=t["ink"], fontsize=11.5, fontweight="600", loc="left", pad=10)
    axes[0].set_ylabel("fleet-mean value (standardised)", color=t["muted"], fontsize=9.5, labelpad=8)
    axes[0].text(127, 2.45, " RUL = 130\n healthy cutoff", fontsize=9, color=t["accent"], va="top")
    fig.suptitle("The degradation signal is real, monotone, and only wakes up below RUL = 130",
                 color=t["ink"], fontsize=12.5, fontweight="600", x=0.055, y=1.04, ha="left")
    fig.tight_layout()
    save(fig, "f4_degradation", t, mode)

# =========================================================================
# FIG 5 — the regime confound, before and after per-mode standardisation
# =========================================================================
for mode, t in THEMES.items():
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 6.0), sharex=True)
    fig.patch.set_facecolor(t["surface"])
    e = train["FD002"][train["FD002"].unit == 1].copy()
    key = e[OPS].round({"op_setting_1": 0, "op_setting_2": 2, "op_setting_3": 0}).astype(str).agg("|".join, axis=1)
    e["mode"] = pd.factorize(key)[0]
    axes[0].plot(e.cycle, e.sensor_7, lw=1.1, color=t["series"][0])
    style(axes[0], t, ylabel="sensor_7  (raw)")
    axes[0].set_title("One engine, one sensor: the regime swing buries the fault",
                      color=t["ink"], fontsize=12.5, fontweight="600", loc="left", pad=12)
    axes[0].text(0.985, 0.90, "range ≈ 390 units", transform=axes[0].transAxes,
                 ha="right", fontsize=9.5, color=t["series"][0], fontweight="600")
    z = e.groupby("mode")["sensor_7"].transform(lambda s: (s - s.mean()) / (s.std() or 1))
    axes[1].axhline(0, color=t["grid"], lw=1)
    axes[1].plot(e.cycle, z, lw=0.9, color=t["series"][1], alpha=0.38)
    axes[1].plot(e.cycle, z.rolling(15, min_periods=1).mean(), lw=2.6,
                 color=t["series"][1], solid_capstyle="round")
    style(axes[1], t, xlabel="cycle", ylabel="sensor_7  (per-mode z)")
    axes[1].text(0.985, 0.90, "range ≈ 6 σ — degradation now legible",
                 transform=axes[1].transAxes, ha="right", fontsize=9.5,
                 color=t["series"][1], fontweight="600")
    fig.tight_layout()
    save(fig, "f5_confound", t, mode)

# =========================================================================
# FIG 6 — sensor correlation structure
# =========================================================================
cmapname = "div"
for mode, t in THEMES.items():
    cm = LinearSegmentedColormap.from_list(cmapname, [t["div_lo"], t["div_mid"], t["div_hi"]])
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    fig.patch.set_facecolor(t["surface"])
    C = train["FD001"][SENSORS].corr().to_numpy()
    im = ax.imshow(C, cmap=cm, vmin=-1, vmax=1)
    ax.set_xticks(range(21), [f"{i}" for i in range(1, 22)], fontsize=8)
    ax.set_yticks(range(21), [f"s{i}" for i in range(1, 22)], fontsize=8)
    ax.tick_params(colors=t["muted"], length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Sensors are heavily co-linear — degradation moves them in coupled groups",
                 color=t["ink"], fontsize=12.5, fontweight="600", loc="left", pad=14)
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.03, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["−1", "0", "+1"], color=t["muted"], fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    cb.set_label("Pearson r", color=t["muted"], fontsize=9.5)
    save(fig, "f6_corr", t, mode)

# =========================================================================
# FIG 7 — test-set truncation
# =========================================================================
for mode, t in THEMES.items():
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    fig.patch.set_facecolor(t["surface"])
    allr = np.concatenate([rul_files[s] for s in SUBSETS])
    ax.hist(allr, bins=np.arange(0, 200, 6), color=t["seq"], alpha=0.9, edgecolor=t["surface"], linewidth=1.1)
    style(ax, t, xlabel="true RUL at the last observed cycle", ylabel="test engines")
    med = float(np.median(allr))
    ax.set_title("No test engine has failed at its last observed cycle - "
                 "the median still has %.0f cycles left" % med,
                 color=t["ink"], fontsize=12.5, fontweight="600", loc="left", pad=14)
    ax.axvline(med, color=t["copper"], lw=1.6)
    ax.text(med + 4, ax.get_ylim()[1] * 0.88, f"median {med:.0f}", color=t["copper"],
            fontsize=10, fontweight="600")
    ax.text(0.995, -0.24, f"707 test engines, pooled.  minimum {allr.min()}, maximum {allr.max()} cycles.",
            transform=ax.transAxes, ha="right", fontsize=9, color=t["muted"])
    stats["quality"]["test_rul_median"] = med
    stats["quality"]["test_rul_min"] = int(allr.min())
    stats["quality"]["test_rul_max"] = int(allr.max())
    stats["quality"]["test_rul_zero"] = int((allr == 0).sum())
    save(fig, "f7_truncation", t, mode)

(Path(__file__).resolve().parent / "eda_stats.json").write_text(json.dumps(stats, indent=2))
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
print("done")
