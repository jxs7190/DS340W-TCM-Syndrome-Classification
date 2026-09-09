"""Create 70/20/10 train, test, and final-unseen validation splits.

For the stomach dataset, split by the clinical-case identifier (编号) rather
than by spreadsheet row. This prevents the same case from leaking across
partitions when one case has multiple syndrome-label rows.
"""

from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xls":
        return pd.read_excel(path, engine="xlrd")
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def split_by_case(
    df: pd.DataFrame,
    case_column: str,
    random_state: int = 42,
):
    if case_column not in df.columns:
        raise KeyError(
            f"Case column {case_column!r} was not found. "
            f"Available columns: {list(df.columns)}"
        )

    case_ids = pd.Series(df[case_column].dropna().unique())

    # First hold out 10% as final unseen validation.
    development_ids, validation_ids = train_test_split(
        case_ids,
        test_size=0.10,
        random_state=random_state,
        shuffle=True,
    )

    # 20/90 = 2/9, so this yields approximately 70/20/10 overall.
    training_ids, test_ids = train_test_split(
        development_ids,
        test_size=2 / 9,
        random_state=random_state,
        shuffle=True,
    )

    training = df[df[case_column].isin(training_ids)].copy()
    test = df[df[case_column].isin(test_ids)].copy()
    validation = df[df[case_column].isin(validation_ids)].copy()

    # Leakage checks.
    train_set = set(training[case_column].dropna())
    test_set = set(test[case_column].dropna())
    validation_set = set(validation[case_column].dropna())

    assert train_set.isdisjoint(test_set)
    assert train_set.isdisjoint(validation_set)
    assert test_set.isdisjoint(validation_set)

    return training, test, validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument(
        "--case-column",
        default="编号",
        help="Column that uniquely identifies a clinical case.",
    )
    parser.add_argument("--name", default="stomach")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = read_table(args.input_file)
    training, test, validation = split_by_case(
        df,
        case_column=args.case_column,
        random_state=args.random_state,
    )

    train_dir = args.output_dir / "train"
    test_dir = args.output_dir / "test"
    validation_dir = args.output_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    training.to_excel(train_dir / f"{args.name}_training.xlsx", index=False)
    test.to_excel(test_dir / f"{args.name}_test.xlsx", index=False)
    validation.to_excel(
        validation_dir / f"{args.name}_validation_UNSEEN.xlsx",
        index=False,
    )

    print(f"Total rows: {len(df)}")
    print(f"Unique cases: {df[args.case_column].nunique()}")
    print(
        "Training:",
        len(training),
        "rows /",
        training[args.case_column].nunique(),
        "cases",
    )
    print(
        "Test:",
        len(test),
        "rows /",
        test[args.case_column].nunique(),
        "cases",
    )
    print(
        "Validation:",
        len(validation),
        "rows /",
        validation[args.case_column].nunique(),
        "cases",
    )
    print("Leakage check: PASS")


if __name__ == "__main__":
    main()
