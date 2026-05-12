from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(exist_ok=True)

MB = 1_048_576  

def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")
    size_mb = path.stat().st_size / MB
    print(f"  Saved → {path.name:<30} ({len(df):>8,} rows, {size_mb:>6.1f} MB)")


def _filter_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[c for c in columns if c in df.columns]]


def fetch_jigsaw() -> None:
    print("\nfetching jigsaw ")
    try:
        ds = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge", split="train")
        df = ds.to_pandas()
        keep = ["comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        df = _filter_columns(df, keep)
        save_csv(df, OUT_DIR / "jigsaw.csv")
    except Exception as e:
        print(f"  ERROR: {e}")


def fetch_civil_comments() -> None:
    print("\nfetching civil comments ")
    try:
        ds = load_dataset("google/civil_comments", split="train")
        df = ds.to_pandas()
        df["toxic"] = (df["toxicity"] >= 0.5).astype(int)
        df = df.rename(columns={"text": "comment_text"})
        
        keep = ["comment_text", "toxic", "toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
        df = _filter_columns(df, keep)
        save_csv(df, OUT_DIR / "civil_comments.csv")
    except Exception as e:
        print(f"  ERROR: {e}")


def fetch_twitter_hate() -> None:
    print("\nfetching twitter hate speech ")
    try:
        splits = []
        for split_name in ("train", "validation", "test"):
            try:
                ds = load_dataset("cardiffnlp/tweet_eval", "hate", split=split_name)
                df = ds.to_pandas()
                df["split"] = split_name
                splits.append(df)
            except Exception:
                pass
        
        if not splits:
            raise RuntimeError("no splits loaded")
        
        df = pd.concat(splits, ignore_index=True)
        df = df.rename(columns={"text": "comment_text", "label": "toxic"})
        save_csv(df, OUT_DIR / "twitter_hate.csv")
    except Exception as e:
        print(f"  ERROR: {e}")


def summary() -> None:    
    total_rows = 0
    for csv_file in sorted(OUT_DIR.glob("*.csv")):
        try:
            n = sum(1 for _ in open(csv_file, encoding="utf-8")) - 1
            size_mb = csv_file.stat().st_size / MB
            print(f"  {csv_file.name:<30} {n:>10,} rows  {size_mb:>7.1f} MB")
            total_rows += n
        except Exception:
            print(f"  {csv_file.name:<30} (could not read)")
    
    print(f"  {'TOTAL':<30} {total_rows:>10,} rows")
    print(f"\nFiles saved in: {OUT_DIR}\n")


if __name__ == "__main__":
    print(f"Fetching datasets into: {OUT_DIR}\n")
    fetch_jigsaw()
    fetch_civil_comments()
    fetch_twitter_hate()
    summary()
