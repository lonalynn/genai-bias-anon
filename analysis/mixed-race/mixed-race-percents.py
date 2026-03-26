"""
compute_mixed_race_differences.py

Computes per-occupation mixed-race % for each of the 4 GenAI models and
subtracts the BLS baseline mixed-race %, producing a CSV in the same format
as the existing *_differences_vs_bls.csv files:

    occupation, diff_p_mixed_openai, diff_p_mixed_gemini,
                diff_p_mixed_deepseek, diff_p_mixed_mistral

Mixed-race detection (same method as the original mixed-race-analysis.py):
  - GenAI:    a profile is mixed-race if its "ethnicity" cell contains a comma
  - BLS:      baseline mixed-race % = max(0, sum_of_ethnicity_percents - 100)
              (any percentage-points above 100 represent people counted in
               multiple racial categories, i.e. multiracial respondents)

Usage:
    python compute_mixed_race_differences.py

Outputs:
    mixed_race_differences_vs_bls.csv  (same directory as this script)
"""

import os
import re
import pandas as pd 

# ---------------------------------------------------------------------------
# CONFIG — adjust paths if needed
# ---------------------------------------------------------------------------
MODELS = {
    "openai":   "../../profiles/openai/",
    "gemini":   "../../profiles/gemini/",
    "deepseek": "../../profiles/deepseek/",
    "mistral":  "../../profiles/mistral/",
}

# Each model folder is expected to contain:
#   <career_term>profiles_<model>.csv   (GenAI-generated profiles)
# AND one shared BLS baselines file:
BASELINE_PATH = "../../profiles/bls-baselines.csv"

OUTPUT_CSV = "mixed_race_differences_vs_bls.csv"

# BLS ethnicity percent columns → display names
BLS_RACE_COLS = ["p_white", "p_black", "p_asian", "p_hispanic"]

 # ------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helper: derive a canonical occupation key from a filename
#   e.g. "nurse practitionerprofiles_openai.csv"  →  "nursepracticioner"
#   (mirrors the format used in the existing differences CSVs)
# ---------------------------------------------------------------------------
def career_key_from_filename(filename: str, model_tag: str) -> str:
    """
    Strip the model-specific suffix from a profile filename, returning a
    normalised (lower-case, no spaces/underscores) occupation key.
 
    Two filename conventions are supported:
      - OpenAI / Mistral:  <career>profiles_<model>.csv
      - Gemini / DeepSeek: <career>_<model>.csv
    """
    base = os.path.basename(filename)
    # Try the "profiles?_<model>" pattern first (OpenAI, Mistral — singular or plural)
    key = re.sub(r"profiles?_" + re.escape(model_tag) + r"\.csv$", "", base, flags=re.IGNORECASE)
    if key == base:
        # Fall back to the "_<model>" pattern (Gemini, DeepSeek)
        key = re.sub(r"_" + re.escape(model_tag) + r"\.csv$", "", base, flags=re.IGNORECASE)
    # Normalise: lower-case, strip whitespace/underscores
    key = key.strip().lower().replace(" ", "").replace("_", "")
    return key
 
 
# ---------------------------------------------------------------------------
# Load BLS baselines once
# ---------------------------------------------------------------------------
baselines_df = pd.read_csv(BASELINE_PATH, encoding="cp1252")
 
def bls_mixed_pct(career_term: str) -> float:
    """
    Return the BLS baseline mixed-race % for a given career term.
    Method (same as original script):
      sum p_white + p_black + p_asian + p_hispanic for that career;
      any total above 100 is interpreted as multiracial overlap → mixed %.
    """
    rows = baselines_df[baselines_df["genai_bias_search_term"] == career_term]
    if rows.empty:
        return 0.0
    # Use the first matching row
    total = sum(rows.iloc[0].get(col, 0) for col in BLS_RACE_COLS)
    return round(max(0.0, total - 100.0), 1)
 
 
# ---------------------------------------------------------------------------
# Process each model
# ---------------------------------------------------------------------------
# results[occupation_key][model] = diff_p_mixed
results: dict[str, dict[str, float]] = {}
 
for model_tag, folder in MODELS.items():
    if not os.path.isdir(folder):
        print(f"WARNING: folder not found for {model_tag}: {folder}")
        continue
 
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".csv"):
            continue
        # Skip the baselines file itself if it lives in the same folder
        if "bls-baselines" in fname.lower():
            continue
 
        fpath = os.path.join(folder, fname)
        for encoding in ("utf-8", "cp1252", "latin-1"):
            try:
                genai_df = pd.read_csv(fpath, encoding=encoding)
                break
            except Exception:
                genai_df = None
        if genai_df is None:
            print(f"WARNING: could not read {fpath} with any encoding — skipping")
            continue
 
        if "ethnicity" not in genai_df.columns:
            continue
 
        # --- GenAI mixed-race % ---
        eth = genai_df["ethnicity"].fillna("").str.lower()
        mixed_count = eth.str.contains(",", regex=False).sum()
        total = len(eth)
        genai_pct = round((mixed_count / total) * 100.0, 1) if total > 0 else 0.0
 
        # --- BLS baseline mixed-race % ---
        # career_term_raw keeps spaces so it matches genai_bias_search_term in bls-baselines.csv.
        # Apply the same two-pattern logic as career_key_from_filename.
        base_no_ext = os.path.basename(fname)
        career_term_raw = re.sub(r"profiles?_" + re.escape(model_tag) + r"\.csv$", "", base_no_ext, flags=re.IGNORECASE)
        if career_term_raw == base_no_ext:
            career_term_raw = re.sub(r"_" + re.escape(model_tag) + r"\.csv$", "", base_no_ext, flags=re.IGNORECASE)
        career_term_raw = career_term_raw.strip()  # keep spaces for BLS lookup
        baseline_pct = bls_mixed_pct(career_term_raw)
 
        diff = round(genai_pct - baseline_pct, 1)
 
        # Canonical occupation key (no spaces, lower-case) for the output CSV
        occ_key = career_term_raw.lower().replace(" ", "").replace("_", "")
 
        if occ_key not in results:
            results[occ_key] = {}
        results[occ_key][model_tag] = diff
 
# ---------------------------------------------------------------------------
# Build output DataFrame
# ---------------------------------------------------------------------------
model_order = ["openai", "gemini", "deepseek", "mistral"]
col_names   = [f"diff_p_mixed_{m}" for m in model_order]
 
rows = []
for occ_key in sorted(results.keys()):
    row = {"occupation": occ_key}
    for m, col in zip(model_order, col_names):
        row[col] = results[occ_key].get(m, 0.0)
    rows.append(row)
 
out_df = pd.DataFrame(rows, columns=["occupation"] + col_names)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
print(out_df.to_string(index=False))
 