from pathlib import Path

import pandas as pd
from datasets import load_dataset


OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(exist_ok=True)

MB = 1_048_576


# saves a dataframe as utf encoded csv
def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8")
    size_mb = path.stat().st_size / MB
    print(f"  Saved → {path.name:<30} ({len(frame):>8,} rows, {size_mb:>6.1f} MB)")


# filters a dataframe to the requested existing columns
def _filter_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]]


# downloads and saves the jigsaw dataset
def fetch_jigsaw() -> None:
    print("\nfetching jigsaw ")

    try:
        dataset = load_dataset(
            "thesofakillers/jigsaw-toxic-comment-classification-challenge",
            split="train",
        )
        frame = dataset.to_pandas()
        columns = [
            "comment_text",
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
        frame = _filter_columns(frame, columns)
        save_csv(frame, OUT_DIR / "jigsaw.csv")
    except Exception as error:
        print(f"  ERROR: {error}")


# downloads and saves the civil comments dataset
def fetch_civil_comments() -> None:
    print("\nfetching civil comments ")

    try:
        dataset = load_dataset("google/civil_comments", split="train")
        frame = dataset.to_pandas()
        frame["toxic"] = (frame["toxicity"] >= 0.5).astype(int)
        frame = frame.rename(columns={"text": "comment_text"})
        columns = [
            "comment_text",
            "toxic",
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack",
        ]
        frame = _filter_columns(frame, columns)
        save_csv(frame, OUT_DIR / "civil_comments.csv")
    except Exception as error:
        print(f"  ERROR: {error}")


# downloads and saves the twitter hate dataset
def fetch_twitter_hate() -> None:
    print("\nfetching twitter hate speech ")

    try:
        splits = []

        for split_name in ("train", "validation", "test"):
            try:
                dataset = load_dataset("cardiffnlp/tweet_eval", "hate", split=split_name)
                frame = dataset.to_pandas()
                frame["split"] = split_name
                splits.append(frame)
            except Exception:
                pass

        if not splits:
            raise RuntimeError("no splits loaded")

        frame = pd.concat(splits, ignore_index=True)
        frame = frame.rename(columns={"text": "comment_text", "label": "toxic"})
        save_csv(frame, OUT_DIR / "twitter_hate.csv")
    except Exception as error:
        print(f"  ERROR: {error}")


# prints a summary of saved csv files
def summary() -> None:
    total_rows = 0

    for csv_file in sorted(OUT_DIR.glob("*.csv")):
        try:
            with csv_file.open(encoding="utf-8") as handle:
                row_count = sum(1 for _ in handle) - 1

            size_mb = csv_file.stat().st_size / MB
            print(f"  {csv_file.name:<30} {row_count:>10,} rows  {size_mb:>7.1f} MB")
            total_rows += row_count
        except Exception:
            print(f"  {csv_file.name:<30} (could not read)")

    print(f"  {'TOTAL':<30} {total_rows:>10,} rows")
    print(f"\nFiles saved in: {OUT_DIR}\n")


# downloads all datasets and prints a summary
def main() -> None:
    print(f"Fetching datasets into: {OUT_DIR}\n")
    fetch_jigsaw()
    fetch_civil_comments()
    fetch_twitter_hate()
    summary()


if __name__ == "__main__":
    main()
