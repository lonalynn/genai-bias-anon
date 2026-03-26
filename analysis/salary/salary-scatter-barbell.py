import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
BLS_FILE = f"data-csvs/earnings_selected_occupations_2023.csv"
MODEL_FILES = {
    "GPT-4.0":        f"data-csvs/openai_salary_by_gender.csv",
    "Gemini 2.5":     f"data-csvs/gemini_salary_by_gender.csv",
    "DeepSeek V3.1":  f"data-csvs/deepseek_salary_by_gender.csv",
    "Mistral-medium": f"data-csvs/mistral_salary_by_gender.csv",
}

MODEL_COLORS = {
    "GPT-4.0":        "#0072B2",
    "Gemini 2.5":     "#E69F00",
    "DeepSeek V3.1":  "#009E73",
    "Mistral-medium": "#CC79A7",
}

MODEL_MARKERS = {
    "GPT-4.0":        "o",
    "Gemini 2.5":     "s",
    "DeepSeek V3.1":  "D",
    "Mistral-medium": "^",
}

# ── load data ──────────────────────────────────────────────────────────────────
bls = pd.read_csv(BLS_FILE)
bls["occupation"] = bls["occupation"].str.strip('"').str.strip()

models = {}
for name, path in MODEL_FILES.items():
    df = pd.read_csv(path)
    df["occupation"] = df["occupation"].str.strip('"').str.strip()
    models[name] = df

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 – Scatter: AI median salary vs BLS median salary, one panel/model
# ─────────────────────────────────────────────────────────────────────────────

fig1, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()

for ax, (model_name, ai_df) in zip(axes, models.items()):
    color  = MODEL_COLORS[model_name]
    marker = MODEL_MARKERS[model_name]

    merged = bls[["occupation", "annual_all"]].merge(
        ai_df[["occupation", "median_salary"]],
        on="occupation", how="inner"
    ).dropna(subset=["annual_all", "median_salary"])

    bls_vals = merged["annual_all"].values
    ai_vals  = merged["median_salary"].values

    # axis limits scaled independently to each axis's data
    x_lo = bls_vals.min() * 0.92
    x_hi = bls_vals.max() * 1.06
    y_lo = ai_vals.min()  * 0.92
    y_hi = ai_vals.max()  * 1.06

    # parity line drawn only within the visible window
    p_lo = max(x_lo, y_lo)
    p_hi = min(x_hi, y_hi)
    ax.plot([p_lo, p_hi], [p_lo, p_hi],
            color="gray", lw=1.2, ls="--", zorder=1, label="Parity (y = x)")

    # scatter with per-model marker
    ax.scatter(bls_vals, ai_vals, color=color, marker=marker,
               alpha=0.78, edgecolors="black", linewidths=0.5, s=60, zorder=3)

    # Pearson r
    r = np.corrcoef(bls_vals, ai_vals)[0, 1]

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.set_title(model_name, fontsize=13, fontweight="bold", color="black", pad=6)
    ax.set_xlabel("BLS Median Annual Salary ($)", fontsize=8.5)
    ax.set_ylabel("AI Median Annual Salary ($)", fontsize=8.5)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))

    ax.tick_params(labelsize=7.5)
    ax.grid(True, lw=0.4, alpha=0.4)

    n = len(merged)
    ax.text(0.04, 0.96,
            f"n = {n}  |  r = {r:.2f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="lightgray"))

    ax.legend(fontsize=7, loc="lower right", framealpha=0.85)

fig1.suptitle(
    "AI-Generated vs. BLS Median Annual Salaries by Occupation",
    fontsize=14, fontweight="bold", y=1.01
)
fig1.tight_layout()
fig1.savefig("results/fig1_salary_scatter.pdf", bbox_inches="tight")
print("Figure 1 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – Gender wage gap dumbbell: BLS gap vs AI gap per occupation
# ─────────────────────────────────────────────────────────────────────────────

bls["bls_gap"] = bls["annual_men"] - bls["annual_women"]

OCC_LABELS = {
    "chiefexecutiveofficer":         "CEO",
    "customerservicerepresentative": "CustSvc Rep",
    "insurancesalesagent":           "Ins. Sales Agent",
    "administrativeassistant":       "Admin Assistant",
    "primaryschoolteacher":          "Elem. Teacher",
    "nursepractitioner":             "Nurse Practitioner",
    "constructionworker":            "Construction Worker",
    "specialedteacher":              "SpEd Teacher",
    "buildinginspector":             "Building Inspector",
    "securityguard":                 "Security Guard",
    "policeofficer":                 "Police Officer",
    "mailcarrier":                   "Mail Carrier",
    "childcareworker":               "Childcare Worker",
    "truckdriver":                   "Truck Driver",
    "busdriver":                     "Bus Driver",
    "garbagecollector":              "Garbage Collector",
    "craneoperator":                 "Crane Operator",
    "computerprogrammer":            "Computer Programmer",
    "softwaredeveloper":             "Software Developer",
    "housekeeper":                   "Housekeeper",
}

def shorten(occ):
    return OCC_LABELS.get(occ, occ.capitalize())

BLS_MARKER = "x"   # consistent BLS marker across all panels

fig2, axes = plt.subplots(2, 2, figsize=(12, 14))
axes = axes.flatten()

for ax, (model_name, ai_df) in zip(axes, models.items()):
    color  = MODEL_COLORS[model_name]
    marker = MODEL_MARKERS[model_name]

    ai_gap_df = ai_df[["occupation", "gender_gap_male_minus_female"]].copy()
    ai_gap_df.columns = ["occupation", "ai_gap"]

    merged = bls[["occupation", "bls_gap"]].merge(
        ai_gap_df, on="occupation", how="inner"
    ).dropna(subset=["bls_gap", "ai_gap"])

    merged = merged.sort_values("bls_gap", ascending=True).reset_index(drop=True)
    n = len(merged)

    bls_k  = merged["bls_gap"].values / 1000
    ai_k   = merged["ai_gap"].values  / 1000
    y      = np.arange(n)
    labels = [shorten(o) for o in merged["occupation"]]

    # connecting lines
    for i in range(n):
        ax.plot([bls_k[i], ai_k[i]], [y[i], y[i]],
                color="lightgray", lw=1.2, zorder=1)

    # BLS dots (gray, "x" marker) and AI dots (model color + marker)
    ax.scatter(bls_k, y, color="#555555", marker=BLS_MARKER, s=50, zorder=3,
               label="BLS", linewidths=1.2)
    ax.scatter(ai_k,  y, color=color, marker=marker, s=50, zorder=3,
               label=model_name, edgecolors="black", linewidths=0.4)

    # zero reference
    ax.axvline(0, color="black", lw=0.8, zorder=2)

    # median dashed lines
    med_bls = merged["bls_gap"].median() / 1000
    med_ai  = merged["ai_gap"].median()  / 1000
    ax.axvline(med_bls, color="#555555", lw=1.2, ls="--", zorder=4,
               label=f"BLS median ${med_bls:.1f}K")
    ax.axvline(med_ai,  color=color,    lw=1.2, ls="--", zorder=4,
               label=f"AI median ${med_ai:.1f}K")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Gender Wage Gap, Men − Women ($K)", fontsize=8.5)
    ax.set_title(model_name, fontsize=13, fontweight="bold", color="black", pad=6)
    ax.tick_params(axis="x", labelsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.0f}K"))
    ax.grid(axis="x", lw=0.4, alpha=0.4, zorder=0)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.88)

    ax.text(0.99, 0.01,
            f"n = {n} occupations",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      alpha=0.75, ec="lightgray"))

fig2.suptitle(
    "Gender Wage Gap by Occupation: BLS Data vs. AI-Generated Estimates\n"
    "(Positive = men earn more; BLS suppresses data for < 50K workers)",
    fontsize=13, fontweight="bold", y=1.01
)
fig2.tight_layout()
fig2.savefig("results/fig2_gender_wage_gap.pdf", bbox_inches="tight")
print("Figure 2 saved.")