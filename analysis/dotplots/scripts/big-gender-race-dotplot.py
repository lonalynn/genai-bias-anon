import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
MODEL_FILES = {
    "ChatGPT":  "../../percent-results/results_vs_BLS/openai_differences_vs_bls.csv",
    "Gemini":   "../../percent-results/results_vs_BLS/gemini_differences_vs_bls.csv",
    "DeepSeek": "../../percent-results/results_vs_BLS/deepseek_differences_vs_bls.csv",
    "Mistral":  "../../percent-results/results_vs_BLS/mistral_differences_vs_bls.csv",
}

DISPLAY_NAMES = {
    "ChatGPT":  "GPT 4.0",
    "DeepSeek": "DeepSeek V3.1",
    "Gemini":   "Gemini 2.5",
    "Mistral":  "Mistral-medium",
}
DISPLAY_ORDER = ["GPT 4.0", "DeepSeek V3.1", "Gemini 2.5", "Mistral-medium"]

GROUPS = ["Women", "White", "Hispanic", "Black", "Asian"]

OUTPUT_PDF = "occupational_bias_race_gender_combined.pdf"

# ----------------------------
# Helpers
# ----------------------------
def nice_from_key(key: str) -> str:
    s = key.strip().replace("_", " ")
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    return " ".join(s.split()).title()

def smart_offsets(values_dict, tol=3.0, base_jitter=0.18):
    keys = list(values_dict.keys())
    used, groups = set(), []
    for i, k in enumerate(keys):
        if k in used: continue
        group = [k]; used.add(k)
        for j in range(i + 1, len(keys)):
            k2 = keys[j]
            if k2 in used: continue
            if pd.notna(values_dict[k]) and pd.notna(values_dict[k2]):
                if abs(values_dict[k] - values_dict[k2]) <= tol:
                    group.append(k2); used.add(k2)
        groups.append(group)

    offsets = {k: 0.0 for k in keys}
    for g in groups:
        if len(g) == 1: continue
        start = -(len(g)/2 - 0.5) if len(g) % 2 == 0 else -(len(g)//2)
        for i, name in enumerate(sorted(g)):
            offsets[name] = (start + i) * base_jitter
    return offsets

# ----------------------------
# Load & reshape (race + gender)
# ----------------------------
frames = []

for model_key, path in MODEL_FILES.items():
    df = pd.read_csv(path)
    df["occ_key"] = df["occupation"].astype(str).str.strip()
    model = DISPLAY_NAMES[model_key]

    # Gender
    frames.append(
        df[["occ_key", "diff_p_women"]]
        .rename(columns={"diff_p_women": "diff"})
        .assign(group="Women", model=model)
    )

    # Races
    race_map = {
        "diff_p_white": "White",
        "diff_p_hispanic": "Hispanic",
        "diff_p_black": "Black",
        "diff_p_asian": "Asian",
    }
    for col, race in race_map.items():
        frames.append(
            df[["occ_key", col]]
            .rename(columns={col: "diff"})
            .assign(group=race, model=model)
        )

all_long = pd.concat(frames, ignore_index=True)

# ----------------------------
# Occupation labels & ordering
# ----------------------------
OCCUPATION_LABELS = [
    "administrative assistant","author","bartender","biologist","building inspector",
    "bus driver","butcher","chef","chemist","chief executive officer","childcare worker",
    "computer programmer","construction worker","cook","crane operator","custodian",
    "customer service representative","doctor","drafter","electrician","engineer",
    "garbage collector","housekeeper","insurance sales agent","lab tech","librarian",
    "mail carrier","nurse","nurse practitioner","pharmacist","pilot","plumber",
    "police officer","primary school teacher","receptionist","roofer","security guard",
    "software developer","special ed teacher","truck driver","welder",
]

occ_keys = sorted(all_long["occ_key"].unique())
clean_label_map = {k: v.title() for k, v in zip(occ_keys, OCCUPATION_LABELS)}

# Order occupations by WOMEN average (top → bottom)
women_means = (
    all_long[all_long["group"] == "Women"]
    .groupby("occ_key")["diff"]
    .mean()
    .sort_values(ascending=False)
)
ordered_occ_keys = list(women_means.index)

# ----------------------------
# Prepare wide tables per group
# ----------------------------
by_group = {}
for g in GROUPS:
    sub = all_long[all_long["group"] == g]
    wide = sub.pivot_table(index="occ_key", columns="model", values="diff", aggfunc="mean")
    wide = wide.reindex(ordered_occ_keys)
    for m in DISPLAY_ORDER:
        if m not in wide.columns:
            wide[m] = np.nan
    wide = wide[DISPLAY_ORDER]

    avg = (
        sub.groupby("model")["diff"]
        .mean()
        .reindex(DISPLAY_ORDER)
        .rename("__AVG__")
        .to_frame().T
    )

    by_group[g] = pd.concat([avg, wide])

# ----------------------------
# Plot
# ----------------------------
markers = {
    "GPT 4.0": "o",
    "DeepSeek V3.1": "D",
    "Gemini 2.5": "s",
    "Mistral-medium": "^",
}
colors = {
    "GPT 4.0": "#0072B2",
    "DeepSeek V3.1": "#009E73",
    "Gemini 2.5": "#E69F00",
    "Mistral-medium": "#CC79A7",
}

fig, axes = plt.subplots(1, len(GROUPS), figsize=(26, 18), sharey=True)

for ax, group in zip(axes, GROUPS):
    wide = by_group[group]
    ykeys = ["__AVG__"] + ordered_occ_keys
    ylabels = ["Average"] + [clean_label_map[k] for k in ordered_occ_keys]

    wide = wide.reindex(ykeys)
    y = np.arange(len(ykeys))

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlim(-100, 100)
    ax.set_title(group, fontsize=20)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_ylim(len(y) - 0.5, -0.5)

    for yi, key in enumerate(ykeys):
        vals = {m: wide.loc[key, m] for m in DISPLAY_ORDER}
        offsets = smart_offsets(vals)

        for m in DISPLAY_ORDER:
            if pd.isna(vals[m]): continue
            ax.scatter(
                vals[m], yi + offsets[m],
                marker=markers[m],
                color=colors[m],
                s=55 if key == "__AVG__" else 42,
                edgecolor="black",
                linewidths=0.4,
                label=m if (group == "Women" and key == "__AVG__") else None,
                zorder=3
            )

    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.6)
    ax.grid(axis="x", linestyle=":", alpha=0.7)

for tick in ax.yaxis.get_ticklabels():
    if tick.get_text() == "Average":
        tick.set_fontweight("bold")
        
fig.suptitle(
    "Per-Occupation Representation vs. BLS — Gender and Race",
    fontsize=40,
    fontweight="bold",
    y=0.97,
)
fig.supxlabel("Difference from BLS (percentage-point difference)", fontsize=20, y=0.05)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Model", loc="upper left", bbox_to_anchor=(0.01, 0.94),
           fontsize=14, title_fontsize=18)

plt.tight_layout(rect=[0.03, 0.08, 0.98, 0.93])
plt.savefig(OUTPUT_PDF, bbox_inches="tight")
print(f"Saved figure: {Path(OUTPUT_PDF).resolve()}")
