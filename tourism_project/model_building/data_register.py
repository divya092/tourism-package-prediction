"""
Data registration and validation.

Reads the dataset that is stored inside the repository, verifies that it matches the
expected schema, and prints a summary. Exits non-zero on any validation failure so the
GitHub Actions workflow stops before the downstream jobs run.
"""

import os
import sys

import pandas as pd

# Path is relative to the repository root, which is the working directory in GitHub Actions
DATASET_PATH = "tourism_project/data/tourism.csv"
TARGET_COLUMN = "ProdTaken"

# The schema the pipeline is built against
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]

# Columns produced by the export that are tolerated but not required
OPTIONAL_COLUMNS = ["Unnamed: 0"]


def fail(message):
    """Print an error and exit non-zero so the workflow stops."""
    print(f"VALIDATION FAILED: {message}")
    sys.exit(1)


def main():
    print("=" * 70)
    print("DATASET REGISTRATION AND VALIDATION")
    print("=" * 70)

    # ---- Check 1: the file exists -------------------------------------------------
    if not os.path.exists(DATASET_PATH):
        fail(f"{DATASET_PATH} was not found in the repository.")
    print(f"Located dataset : {DATASET_PATH}")
    print(f"File size       : {os.path.getsize(DATASET_PATH) / 1024:.1f} KB")

    df = pd.read_csv(DATASET_PATH)

    # ---- Check 2: the dataset is not empty ----------------------------------------
    if df.empty:
        fail("the dataset contains no rows.")

    # ---- Check 3: all expected columns are present --------------------------------
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        fail(f"expected columns are missing: {missing_columns}")

    # ---- Check 4: no unexpected columns have appeared -----------------------------
    unexpected = [c for c in df.columns
                  if c not in EXPECTED_COLUMNS + OPTIONAL_COLUMNS]
    if unexpected:
        fail(f"unexpected columns found: {unexpected}")

    # ---- Check 5: the target is binary --------------------------------------------
    target_values = set(df[TARGET_COLUMN].dropna().unique())
    if not target_values.issubset({0, 1}):
        fail(f"{TARGET_COLUMN} must contain only 0 and 1, found {sorted(target_values)}")

    # ---- Summary ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")
    print()

    print("Column types:")
    print(df.dtypes.to_string())
    print()

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("Missing values:")
    print(missing.to_string() if len(missing) else "  none")
    print()

    counts = df[TARGET_COLUMN].value_counts().sort_index()
    percent = df[TARGET_COLUMN].value_counts(normalize=True).sort_index().mul(100).round(2)
    print(f"Target distribution ({TARGET_COLUMN}):")
    for label in counts.index:
        print(f"  {label} : {counts[label]:>6,}  ({percent[label]:>5.2f}%)")
    print()

    print("=" * 70)
    print("DATASET REGISTERED SUCCESSFULLY - all validation checks passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
