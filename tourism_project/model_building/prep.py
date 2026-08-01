"""
Data preparation.

Loads the registered dataset from the repository, applies cleaning, splits it into
train and test sets, and saves the four CSV files at the repository root so the
workflow can pass them to the training job as an artifact.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "tourism_project/data/tourism.csv"
TARGET_COLUMN = "ProdTaken"

# Columns dropped before modelling:
#   Unnamed: 0 -> leftover export index, no predictive value
#   CustomerID -> unique identifier, would let the model memorise individual rows
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]

TEST_SIZE = 0.2
RANDOM_STATE = 42


def clean(df):
    """Apply the cleaning decisions identified during exploratory analysis."""
    rows_before = len(df)

    # 1. Remove identifier and index columns
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    print(f"Dropped columns    : {[c for c in DROP_COLUMNS]}")

    # 2. Fix the 'Fe Male' data-entry error so encoding does not create a third category
    if "Gender" in df.columns:
        before = sorted(df["Gender"].dropna().unique())
        df["Gender"] = df["Gender"].replace("Fe Male", "Female")
        after = sorted(df["Gender"].dropna().unique())
        print(f"Gender categories  : {before} -> {after}")

    # 3. Remove duplicate customer profiles so the same record cannot land in
    #    both the train and test split and inflate the test score
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate rows     : removed {rows_before - len(df)}")

    return df


def main():
    print("=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset     : {DATASET_PATH}  shape={df.shape}")
    print()

    df = clean(df)
    print(f"Shape after clean  : {df.shape}")
    print()

    # Separate predictors from the target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Stratified split keeps the ~19% positive rate in both train and test
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("-" * 70)
    print("TRAIN / TEST SPLIT")
    print("-" * 70)
    print(f"Xtrain : {Xtrain.shape}   positive rate: {ytrain.mean():.4f}")
    print(f"Xtest  : {Xtest.shape}   positive rate: {ytest.mean():.4f}")
    print()

    # Saved at the repository root so the workflow artifact paths resolve
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
    print("=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
