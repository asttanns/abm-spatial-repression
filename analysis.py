"""
Analysis script for spatial repression-mobilization batch run results.
Input:  results_baseline.csv (from batch_run.py)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load
df = pd.read_csv("results_baseline.csv")

mob_col = None
for c in ["MobilizedProportion", "reconstitution_rate", "Mobilized"]:
    if c in df.columns:
        mob_col = c
        break
if mob_col is None:
    print("Available columns:", df.columns.tolist())
    raise ValueError("Cannot find mobilization column.")

print(f"Using column: {mob_col}")
print(f"Rows: {len(df)}")
print("\nMean by env_type and observation_radius:")
print(df.groupby(["observation_radius", "env_type"])[mob_col].mean().unstack().round(3))

# set up
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

ENV_ORDER  = ["kampung", "mixed", "open"]
ENV_LABELS = ["Kampung", "Mixed commercial", "Planned open"]
COLORS     = {"kampung": "#0077BB", "mixed": "#EE7733", "open": "#BBBBBB"}
TM = sorted(df["threshold_mean"].unique())
TS = sorted(df["threshold_std"].unique())

# identify seed column
seed_col = "seed" if "seed" in df.columns else ("manual_seed" if "manual_seed" in df.columns else None)
seeds = sorted(df[seed_col].unique()) if seed_col else [0]
n_seeds = len(seeds)

# ── FIGURE 1: Box plot ────────────────────────────────────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
fig1.suptitle("Reconstitution Rate by Environment Typology",
              fontsize=13, fontweight="bold", y=1.01)

for ax, radius in zip(axes, [0, 1]):
    sub = df[df["observation_radius"] == radius]
    data = [sub[sub["env_type"] == env][mob_col].values for env in ENV_ORDER]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5),
                    widths=0.5)
    for patch, env in zip(bp["boxes"], ENV_ORDER):
        patch.set_facecolor(COLORS[env]); patch.set_alpha(0.85)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(ENV_LABELS, fontsize=10)
    ax.set_title(f"Cell only (radius=0)" if radius == 0 else "Neighborhood (radius=1)",
                 fontsize=11, pad=8)
    ax.set_ylabel("Mobilized proportion" if radius == 0 else "", fontsize=10)
    ax.set_ylim(-0.02, 0.75)
    ax.axhline(0.1, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
fig1.text(0.5, -0.04,
          "Dashed line at 0.10 = persistence threshold. "
          f"Each box spans {len(TM)*len(TS)} parameter cells x {n_seeds} seeds.",
          ha="center", fontsize=9, color="gray")
plt.tight_layout()
fig1.savefig("fig_boxplot.png", bbox_inches="tight", dpi=150)
print("\nSaved: fig_boxplot.png")

# FIGURE A: per-env heatmap, mean reconstitution 
fig2, axes2 = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
fig2.suptitle("Mean Reconstitution Rate\nby Environment, Threshold Mean, and Threshold Std",
              fontsize=12, fontweight="bold")

for col, env in enumerate(ENV_ORDER):
    for row_i, radius in enumerate([0, 1]):
        ax = axes2[row_i, col]
        sub = df[(df["env_type"] == env) & (df["observation_radius"] == radius)]
        matrix = np.zeros((len(TS), len(TM)))
        for i, ts in enumerate(TS):
            for j, tm in enumerate(TM):
                cell = sub[(sub["threshold_std"] == ts) & (sub["threshold_mean"] == tm)]
                matrix[i, j] = cell[mob_col].mean() if len(cell) else np.nan
        vmax = 0.7
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(TM)))
        ax.set_xticklabels([f"{v:.2f}" for v in TM], fontsize=8)
        ax.set_yticks(range(len(TS)))
        ax.set_yticklabels([f"{v:.2f}" for v in TS], fontsize=8)
        if row_i == 1: ax.set_xlabel("Threshold mean", fontsize=9)
        if col == 0:   ax.set_ylabel(f"radius={radius}\nThreshold std", fontsize=9)
        if row_i == 0: ax.set_title(ENV_LABELS[col], fontsize=10, fontweight="bold")
        for i in range(len(TS)):
            for j in range(len(TM)):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if v > 0.35 else "black")

plt.colorbar(im, ax=axes2[:, -1], label="Mean mobilized proportion",
             fraction=0.05, pad=0.04)
fig2.savefig("fig_A_env_heatmap.png", bbox_inches="tight", dpi=150)
print("Saved: fig_A_env_heatmap.png")

# ── FIGURE B: faceted heatmap with mean +/- std ───────────────────────────────
fig3, axes3 = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
fig3.suptitle("Reconstitution Rate Across All Parameter Combinations\n"
              "(mean ± std, rows = observation radius, cols = environment typology)",
              fontsize=12, fontweight="bold")

for col, env in enumerate(ENV_ORDER):
    for row_i, radius in enumerate([0, 1]):
        ax = axes3[row_i, col]
        sub = df[(df["env_type"] == env) & (df["observation_radius"] == radius)]
        matrix = np.zeros((len(TS), len(TM)))
        text_mat = np.empty((len(TS), len(TM)), dtype=object)
        for i, ts in enumerate(TS):
            for j, tm in enumerate(TM):
                cell = sub[(sub["threshold_std"] == ts) & (sub["threshold_mean"] == tm)]
                m = cell[mob_col].mean() if len(cell) else 0
                s = cell[mob_col].std()  if len(cell) else 0
                matrix[i, j] = m
                text_mat[i, j] = f"{m:.2f}\n±{s:.2f}"
        im3 = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=0.7,
                        aspect="auto", origin="lower")
        ax.set_xticks(range(len(TM)))
        ax.set_xticklabels([f"{v:.2f}" for v in TM], fontsize=8)
        ax.set_yticks(range(len(TS)))
        ax.set_yticklabels([f"{v:.2f}" for v in TS], fontsize=8)
        if row_i == 1: ax.set_xlabel("Threshold mean", fontsize=9)
        if col == 0:   ax.set_ylabel(f"radius={radius}\nThreshold std", fontsize=9)
        if row_i == 0: ax.set_title(ENV_LABELS[col], fontsize=10, fontweight="bold")
        for i in range(len(TS)):
            for j in range(len(TM)):
                v = matrix[i, j]
                ax.text(j, i, text_mat[i, j], ha="center", va="center",
                        fontsize=7.5, color="white" if v > 0.35 else "black")

plt.colorbar(im3, ax=axes3[:, -1], label="Mean mobilized proportion",
             fraction=0.05, pad=0.04)
fig3.savefig("fig_B_faceted_heatmap.png", bbox_inches="tight", dpi=150)
print("Saved: fig_B_faceted_heatmap.png")

# FIGURE C: proportion rank order holds 
fig4, axes4 = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig4.suptitle("Proportion of Seeds Where Rank Order Holds\n"
              "kampung > mixed > open (per parameter cell)",
              fontsize=12, fontweight="bold")

for ax, radius in zip(axes4, [0, 1]):
    sub = df[df["observation_radius"] == radius]
    matrix = np.zeros((len(TS), len(TM)))
    for i, ts in enumerate(TS):
        for j, tm in enumerate(TM):
            holds = 0
            cell = sub[(sub["threshold_std"] == ts) & (sub["threshold_mean"] == tm)]
            if seed_col:
                for s in seeds:
                    sc = cell[cell[seed_col] == s]
                    k = sc[sc["env_type"] == "kampung"][mob_col].values
                    m = sc[sc["env_type"] == "mixed"][mob_col].values
                    o = sc[sc["env_type"] == "open"][mob_col].values
                    if len(k) and len(m) and len(o):
                        if k[0] > m[0] > o[0]:
                            holds += 1
                matrix[i, j] = holds / n_seeds
            else:
                k = cell[cell["env_type"] == "kampung"][mob_col].mean()
                m = cell[cell["env_type"] == "mixed"][mob_col].mean()
                o = cell[cell["env_type"] == "open"][mob_col].mean()
                matrix[i, j] = 1.0 if k > m > o else 0.0

    im4 = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1,
                    aspect="auto", origin="lower")
    ax.set_xticks(range(len(TM)))
    ax.set_xticklabels([f"{v:.2f}" for v in TM], fontsize=9)
    ax.set_yticks(range(len(TS)))
    ax.set_yticklabels([f"{v:.2f}" for v in TS], fontsize=9)
    ax.set_xlabel("Threshold mean", fontsize=10)
    if radius == 0: ax.set_ylabel("Threshold std", fontsize=10)
    ax.set_title(f"Cell only (radius=0)" if radius == 0 else "Neighborhood (radius=1)",
                 fontsize=11)
    for i in range(len(TS)):
        for j in range(len(TM)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if v > 0.6 else "black")

plt.colorbar(im4, ax=axes4[1],
             label="Proportion of seeds where rank order holds",
             fraction=0.046, pad=0.04)
fig4.savefig("fig_C_rankorder.png", bbox_inches="tight", dpi=150)
print("Saved: fig_C_rankorder.png")

# Summary 
print("\n── Summary ──────────────────────────────────────────────────────")
for radius in [0, 1]:
    sub = df[df["observation_radius"] == radius]
    k = sub[sub["env_type"] == "kampung"][mob_col]
    m = sub[sub["env_type"] == "mixed"][mob_col]
    o = sub[sub["env_type"] == "open"][mob_col]
    print(f"\nRadius={radius}:")
    print(f"  Kampung:  mean={k.mean():.3f}, std={k.std():.3f}")
    print(f"  Mixed:    mean={m.mean():.3f}, std={m.std():.3f}")
    print(f"  Open:     mean={o.mean():.3f}, std={o.std():.3f}")
    print(f"  Kampung/Open ratio: {k.mean()/o.mean():.1f}x")

# FIGURE C v2: fixed rank order proportion 
# Pairs kampung/mixed/open values by row index within each parameter cell
# (corresponds to matched seed runs without requiring explicit seed column)

fig5, axes5 = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig5.suptitle("Proportion of Runs Where Rank Order Holds\n"
              "kampung > mixed > open (per parameter cell, 10 runs each)",
              fontsize=12, fontweight="bold")

for ax, radius in zip(axes5, [0, 1]):
    sub = df[df["observation_radius"] == radius]
    matrix = np.zeros((len(TS), len(TM)))

    for i, ts in enumerate(TS):
        for j, tm in enumerate(TM):
            cell = sub[(sub["threshold_std"] == ts) & (sub["threshold_mean"] == tm)]

            # Get values per typology, sort by seed if available
            def get_vals(env):
                rows = cell[cell["env_type"] == env]
                if seed_col and seed_col in rows.columns:
                    rows = rows.sort_values(seed_col)
                return rows[mob_col].values

            k_vals = get_vals("kampung")
            m_vals = get_vals("mixed")
            o_vals = get_vals("open")

            n = min(len(k_vals), len(m_vals), len(o_vals))
            if n == 0:
                matrix[i, j] = np.nan
            else:
                holds = sum(
                    1 for k, m, o in zip(k_vals[:n], m_vals[:n], o_vals[:n])
                    if k > m > o
                )
                matrix[i, j] = holds / n

    im5 = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1,
                    aspect="auto", origin="lower")
    ax.set_xticks(range(len(TM)))
    ax.set_xticklabels([f"{v:.2f}" for v in TM], fontsize=9)
    ax.set_yticks(range(len(TS)))
    ax.set_yticklabels([f"{v:.2f}" for v in TS], fontsize=9)
    ax.set_xlabel("Threshold mean", fontsize=10)
    if radius == 0: ax.set_ylabel("Threshold std", fontsize=10)
    ax.set_title("Cell only (radius=0)" if radius == 0 else "Neighborhood (radius=1)",
                 fontsize=11)
    for i in range(len(TS)):
        for j in range(len(TM)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if v > 0.6 else "black")

plt.colorbar(im5, ax=axes5[1],
             label="Proportion of runs where rank order holds",
             fraction=0.046, pad=0.04)
fig5.savefig("fig_C_rankorder.png", bbox_inches="tight", dpi=150)
print("Saved: fig_C_rankorder.png (fixed)")