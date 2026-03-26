"""
4-column entropy figure (Figure 2) with alpha=1.0 and black edges.
Okabe-Ito colorblind-safe palette matching Figure 3 dotplot.

Run from the directory containing this script:
    cd analysis/entropy
    python make_fig_entropy_4col.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

df = pd.read_csv(os.path.join(SCRIPT_DIR, 'modal_distributional_results.csv'))

MODEL_ORDER = ['GPT-4', 'DeepSeek', 'Gemini', 'Mistral']

MODEL_COLORS = {
    'GPT-4':    '#0072B2',
    'DeepSeek': '#009E73',
    'Gemini':   '#E69F00',
    'Mistral':  '#CC79A7',
}

MODEL_MARKERS = {
    'GPT-4':    'o',
    'DeepSeek': 'D',
    'Gemini':   's',
    'Mistral':  '^',
}

fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=False, sharey=False)

for col, model in enumerate(MODEL_ORDER):
    subset = df[df['model'] == model]
    color = MODEL_COLORS[model]
    marker = MODEL_MARKERS[model]

    # Gender (top row)
    ax = axes[0, col]
    ax.scatter(subset['gender_entropy_bls'], subset['gender_entropy'],
               alpha=1.0, s=45, color=color, marker=marker,
               edgecolors='black', linewidth=0.4)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(model, fontsize=13, fontweight='bold')
    if col == 0:
        ax.set_ylabel('Model Entropy', fontsize=11)
    ax.set_xlabel('')
    ax.tick_params(labelsize=9)

    # Race (bottom row)
    ax = axes[1, col]
    ax.scatter(subset['race_entropy_bls'], subset['race_entropy'],
               alpha=1.0, s=45, color=color, marker=marker,
               edgecolors='black', linewidth=0.4)
    ax.plot([0, 2], [0, 2], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-0.05, 2.05)
    if col == 0:
        ax.set_ylabel('Model Entropy', fontsize=11)
    ax.set_xlabel('BLS Entropy', fontsize=11)
    ax.tick_params(labelsize=9)

fig.suptitle('Within-Occupation Entropy: Model vs. BLS', fontsize=16, fontweight='bold', y=1.0)

fig.text(0.01, 0.73, 'Gender', ha='center', va='center', fontsize=14,
         fontweight='bold', rotation=90, transform=fig.transFigure)
fig.text(0.01, 0.30, 'Race', ha='center', va='center', fontsize=14,
         fontweight='bold', rotation=90, transform=fig.transFigure)

plt.tight_layout(rect=[0.03, 0.02, 1, 0.95])
plt.subplots_adjust(hspace=0.30)

fig.savefig(os.path.join(SCRIPT_DIR, 'fig_entropy_4col.png'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(SCRIPT_DIR, 'fig_entropy_4col.pdf'),
            bbox_inches='tight', dpi=300)
print("Saved 4-column entropy figure v2.")
plt.close()
