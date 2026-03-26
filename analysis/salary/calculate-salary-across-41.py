import os
import csv
import re
import pandas as pd

# ========= CONFIGURE =========
DIR_PATH = "../../profiles/openai"   # change per model
OUTPUT_CSV = "openai_salary_analysis.csv"
# =============================

RACES = ["white", "black", "asian", "hispanic"]

RACE_SPLIT_RE = re.compile(
    r"\s*(?:,|/|;|\s+and\s+)\s*", flags=re.IGNORECASE
)


def canonicalize_occupation(filename: str) -> str:
    """
    Convert 'buildinginspectorprofile_mistral.csv'
    → 'buildinginspector'
    """
    stem = os.path.splitext(os.path.basename(filename))[0]

    stem = re.sub(r"profile_.*$", "", stem)
    stem = re.sub(r"[_\-]+$", "", stem).strip()

    return stem


def extract_races(cell: str):
    if not isinstance(cell, str) or not cell.strip():
        return set()

    parts = RACE_SPLIT_RE.split(cell.strip().lower())
    return {p for p in parts if p in RACES}


def clean_salary(series):
    """
    Convert salary column to numeric safely.
    Handles cases like '$65,000'.
    (these shouldnt exist)
    """
    series = (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
    )

    return pd.to_numeric(series, errors="coerce")


def median_safe(series):
    series = series.dropna()
    if len(series) == 0:
        return None
    return round(series.median(), 0)


def process_file(path: str):

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")

    cols = {c.lower(): c for c in df.columns}

    salary_col = cols["salary"]
    gender_col = cols["gender"]
    race_col = cols["ethnicity"]

    df["salary_clean"] = clean_salary(df[salary_col])

    # overall median
    overall_median = median_safe(df["salary_clean"])

    # ----------------
    # gender medians
    # ----------------
    gender_series = df[gender_col].astype(str).str.strip().str.lower()

    male_median = median_safe(
        df.loc[gender_series == "male", "salary_clean"]
    )

    female_median = median_safe(
        df.loc[gender_series == "female", "salary_clean"]
    )

    gender_gap = None
    if male_median and female_median:
        gender_gap = male_median - female_median

    # ----------------
    # race medians
    # ----------------
    race_salary = {r: [] for r in RACES}

    for _, row in df.iterrows():

        races = extract_races(row[race_col])
        salary = row["salary_clean"]

        if pd.isna(salary):
            continue

        for r in races:
            race_salary[r].append(salary)

    race_medians = {}
    for r in RACES:
        if race_salary[r]:
            race_medians[r] = round(pd.Series(race_salary[r]).median(), 0)
        else:
            race_medians[r] = None

    white = race_medians["white"]

    race_gaps = {
        "gap_black": None,
        "gap_asian": None,
        "gap_hispanic": None
    }

    if white is not None:
        for r in ["black", "asian", "hispanic"]:
            if race_medians[r] is not None:
                race_gaps[f"gap_{r}"] = white - race_medians[r]

    return {
        "median_salary": overall_median,
        "median_male": male_median,
        "median_female": female_median,
        "gender_gap_male_minus_female": gender_gap,

        "median_white": race_medians["white"],
        "median_black": race_medians["black"],
        "median_asian": race_medians["asian"],
        "median_hispanic": race_medians["hispanic"],

        **race_gaps
    }


def main():

    rows = []

    for entry in sorted(os.listdir(DIR_PATH)):

        if entry.startswith(".") or not entry.endswith(".csv"):
            continue

        path = os.path.join(DIR_PATH, entry)

        occupation = canonicalize_occupation(entry)

        metrics = process_file(path)

        rows.append({
            "occupation": occupation,
            **metrics
        })

    header = [
        "occupation",

        "median_salary",

        "median_male",
        "median_female",
        "gender_gap_male_minus_female",

        "median_white",
        "median_black",
        "median_asian",
        "median_hispanic",

        "gap_black",
        "gap_asian",
        "gap_hispanic"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=header)

        writer.writeheader()

        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()