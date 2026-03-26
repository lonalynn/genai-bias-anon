import pandas as pd

# -----------------------------
# File paths
# -----------------------------

bls_file = "earnings_selected_occupations_2023.csv"

ai_files = {
    "openai": "openai_salary_analysis.csv",
    "gemini": "gemini_salary_analysis.csv",
    "deepseek": "deepseek_salary_analysis.csv",
    "mistral": "mistral_salary_analysis.csv"
}

# -----------------------------
# Load BLS data
# -----------------------------

bls = pd.read_csv(bls_file)

bls = bls.rename(columns={
    "annual_all": "bls",
    "annual_men": "bls_men",
    "annual_women": "bls_women"
})

combined = bls[["occupation", "bls"]].copy()

# -----------------------------
# Process each AI file
# -----------------------------

for ai_name, file in ai_files.items():

    ai = pd.read_csv(file)

    # Convert numeric columns safely
    ai["median_salary"] = pd.to_numeric(ai["median_salary"], errors="coerce")
    ai["median_male"] = pd.to_numeric(ai["median_male"], errors="coerce")
    ai["median_female"] = pd.to_numeric(ai["median_female"], errors="coerce")

    # Rename for clarity
    ai = ai.rename(columns={
        "median_salary": "ai_all",
        "median_male": "ai_men",
        "median_female": "ai_women"
    })

    # Merge
    merged = pd.merge(
        bls,
        ai[["occupation", "ai_all", "ai_men", "ai_women"]],
        on="occupation",
        how="left"
    )

    # -----------------------------
    # Overall diff
    # -----------------------------

    merged[f"diff_{ai_name}"] = merged["ai_all"] - merged["bls"]

    combined = pd.merge(
        combined,
        merged[["occupation", "ai_all", f"diff_{ai_name}"]].rename(
            columns={"ai_all": ai_name}
        ),
        on="occupation",
        how="left"
    )

    # -----------------------------
    # Gender output (per AI)
    # -----------------------------

    merged["diff_men"] = merged["ai_men"] - merged["bls_men"]
    merged["diff_women"] = merged["ai_women"] - merged["bls_women"]

    gender = merged[[
        "occupation",
        "bls_men",
        "ai_men",
        "diff_men",
        "bls_women",
        "ai_women",
        "diff_women"
    ]]

    gender.to_csv(f"results/comparison_{ai_name}_gender.csv", index=False)

# -----------------------------
# Save combined file
# -----------------------------

combined.to_csv("results/combined_ai_salary_comparison.csv", index=False)

print("All comparison files generated.")