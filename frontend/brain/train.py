import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from polish import train_polish_layer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CURATED_DATA_DIR = BASE_DIR / "curated_data"
MODEL_ROOT = BASE_DIR / "models"
BRED_DIR = MODEL_ROOT / "bred_bert"
LSTM_PATH = MODEL_ROOT / "lstm_classifier.pt"
CURATED_REPEAT = 40
BASE_BERT_MODEL = "distilbert-base-uncased"
MAX_SAMPLE_WEIGHT = 6.0

TOXIC_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
WORD_PATTERN = re.compile(r"[A-Za-z']+")
HIGH_PRIORITY_CATEGORIES = {
    "threat": 5.0,
    "indirect_threat": 5.5,
    "intimidation": 4.5,
    "coercion": 4.0,
    "identity_attack": 3.5,
    "dehumanization": 3.5,
    "sexual_harassment": 3.5,
    "harassment": 3.0,
    "insult": 2.5,
    "quoted_abuse": 2.0,
    "hard_negative": 2.0,
    "neutral_context": 1.8,
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected — running on CPU.")


def tokenize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall((text or "").lower())


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(token, 1) for token in tokenize_words(text)[:max_len]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


def split_tags(value) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        tag.strip().lower()
        for tag in re.split(r"[|,;/\s]+", str(value))
        if tag.strip()
    }


def compute_sample_weight(row: pd.Series) -> float:
    explicit_weight = row.get("sample_weight", row.get("weight"))
    if explicit_weight is not None and not pd.isna(explicit_weight):
        try:
            return max(0.1, min(MAX_SAMPLE_WEIGHT, float(explicit_weight)))
        except ValueError:
            pass

    weight = 1.0

    for column, column_weight in (
        ("threat", 5.0),
        ("severe_toxic", 4.0),
        ("severe_toxicity", 4.0),
        ("identity_hate", 3.5),
        ("identity_attack", 3.5),
        ("insult", 2.5),
        ("obscene", 2.0),
    ):
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        try:
            if float(value) > 0:
                weight = max(weight, column_weight)
        except ValueError:
            continue

    for column in ("category", "flags"):
        for tag in split_tags(row.get(column)):
            weight = max(weight, HIGH_PRIORITY_CATEGORIES.get(tag, 1.0))

    severity = row.get("severity")
    if severity is not None and not pd.isna(severity):
        try:
            weight = max(weight, 1.0 + float(severity) / 2.5)
        except ValueError:
            pass

    return min(weight, MAX_SAMPLE_WEIGHT)


def load_training_frame(csv_file: Path, nrows: int | None = None) -> pd.DataFrame | None:
    df = pd.read_csv(csv_file, nrows=nrows)

    if "comment_text" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "comment_text"})

    if "toxic" not in df.columns:
        label_columns = [c for c in TOXIC_LABELS if c in df.columns]
        if label_columns:
            df["toxic"] = df[label_columns].fillna(0).max(axis=1).astype(int)
        elif "label" in df.columns:
            df["toxic"] = df["label"].astype(int)
        elif "toxicity" in df.columns:
            df["toxic"] = (df["toxicity"] >= 0.5).astype(int)
        else:
            return None

    if "comment_text" not in df.columns:
        return None

    frame = df.dropna(subset=["comment_text"]).copy()
    frame["comment_text"] = frame["comment_text"].astype(str)
    frame["toxic"] = frame["toxic"].astype(int)
    frame["sample_weight"] = frame.apply(compute_sample_weight, axis=1).astype(float)
    frame = frame[["comment_text", "toxic", "sample_weight"]]
    return frame


def load_training_data(
    max_train_samples: int | None = None,
    curated_repeat: int = CURATED_REPEAT,
) -> tuple[list[str], list[int], list[float], list[str]]:
    texts: list[str] = []
    labels: list[int] = []
    sample_weights: list[float] = []
    loaded_files: list[str] = []

    for csv_file in sorted(DATA_DIR.glob("*.csv")):
        remaining = None if max_train_samples is None else max_train_samples - len(texts)
        if remaining is not None and remaining <= 0:
            break

        frame = load_training_frame(csv_file, nrows=remaining)
        if frame is None:
            continue

        if frame.empty:
            continue

        texts.extend(frame["comment_text"].tolist())
        labels.extend(frame["toxic"].tolist())
        sample_weights.extend(frame["sample_weight"].tolist())
        loaded_files.append(f"data/{csv_file.name}")

    if curated_repeat > 0 and CURATED_DATA_DIR.exists():
        for csv_file in sorted(CURATED_DATA_DIR.glob("*.csv")):
            frame = load_training_frame(csv_file)
            if frame is None or frame.empty:
                continue

            curated_texts = frame["comment_text"].tolist()
            curated_labels = frame["toxic"].tolist()
            curated_weights = frame["sample_weight"].tolist()
            for _ in range(curated_repeat):
                texts.extend(curated_texts)
                labels.extend(curated_labels)
                sample_weights.extend(curated_weights)
            loaded_files.append(f"curated_data/{csv_file.name} x{curated_repeat}")

    if not texts:
        raise RuntimeError(
            f"No usable training rows found in {DATA_DIR} or {CURATED_DATA_DIR}."
        )

    return texts, labels, sample_weights, loaded_files


class BertDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        sample_weights: list[float],
        tokenizer,
        max_len: int = 160,
    ):
        self.texts = texts
        self.labels = labels
        self.sample_weights = sample_weights
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        if idx not in self._cache:
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            self._cache[idx] = {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }
        cached = self._cache[idx]
        return {
            "input_ids": cached["input_ids"],
            "attention_mask": cached["attention_mask"],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "weights": torch.tensor(self.sample_weights[idx], dtype=torch.float32),
        }


class LSTMTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        sample_weights: list[float],
        vocab: dict[str, int],
        max_len: int = 120,
    ):
        self.texts = texts
        self.labels = labels
        self.sample_weights = sample_weights
        self.vocab = vocab
        self.max_len = max_len
        self._cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        if idx not in self._cache:
            self._cache[idx] = torch.tensor(
                encode_text(self.texts[idx], self.vocab, self.max_len),
                dtype=torch.long,
            )
        return {
            "input_ids": self._cache[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
            "weights": torch.tensor(self.sample_weights[idx], dtype=torch.float32),
        }


def build_vocab(
    texts: list[str], min_freq: int = 2, max_size: int = 50_000
) -> dict[str, int]:
    counter: Counter = Counter()
    for text in tqdm(texts, desc="Building vocab", unit="text"):
        counter.update(tokenize_words(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(features)).squeeze(1)


#BERT
def train_bred_bert(
    train_texts: list[str],
    train_labels: list[int],
    train_weights: list[float],
    epochs: int,
    batch_size: int = 32,
    lr: float = 2e-5,
) -> None:
    model_source = (
        str(BRED_DIR) if (BRED_DIR / "config.json").exists() else BASE_BERT_MODEL
    )
    tokenizer = AutoTokenizer.from_pretrained(model_source)

    dataset = BertDataset(train_texts, train_labels, train_weights, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=2)
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"BRED/BERT Epoch {epoch}/{epochs}", unit="batch")

        for batch in progress:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            weights = batch["weights"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                loss = (criterion(logits, labels) * weights).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        print(f"BRED/BERT Epoch {epoch}/{epochs} — avg loss: {total_loss / len(dataloader):.4f}")

    BRED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(BRED_DIR)
    tokenizer.save_pretrained(BRED_DIR)


#LTSM
def train_lstm(
    train_texts: list[str],
    train_labels: list[int],
    train_weights: list[float],
    epochs: int,
    batch_size: int = 512,
    lr: float = 1e-3,
    max_len: int = 120,
) -> None:
    vocab = build_vocab(train_texts)
    dataset = LSTMTextDataset(train_texts, train_labels, train_weights, vocab, max_len=max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    model = LSTMClassifier(vocab_size=len(vocab)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"LSTM Epoch {epoch}/{epochs}", unit="batch")

        for batch in progress:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            weights = batch["weights"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                loss = (criterion(model(input_ids), labels) * weights).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        print(f"LSTM Epoch {epoch}/{epochs} — avg loss: {total_loss / len(dataloader):.4f}")

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "vocab": vocab, "max_len": max_len},
        LSTM_PATH,
    )


def parse_optional_int(user_input: str, default: int | None = None) -> int | None:
    user_input = user_input.strip()
    if not user_input:
        return default
    if not user_input.isdigit() or int(user_input) <= 0:
        raise ValueError("Value must be a positive integer.")
    return int(user_input)


def main() -> None:
    try:
        sample_input = input("Max training samples (press Enter for full dataset): ")
        max_samples = parse_optional_int(sample_input, default=None)
        epoch_input = input("Epochs for each model (default 2): ")
        epochs = parse_optional_int(epoch_input, default=2)
    except ValueError as error:
        print(error)
        return

    print("\nLoading dataset...")
    try:
        train_texts, train_labels, train_weights, loaded_files = load_training_data(
            max_train_samples=max_samples
        )
    except RuntimeError as error:
        print(error)
        return

    print(f"Loaded {len(train_texts):,} training samples from:")
    for name in loaded_files:
        print(f"  - {name}")

    print("Training BRED (DistilBERT)...")
    train_bred_bert(train_texts, train_labels, train_weights, epochs=epochs)

    print("\nTraining LSTM...")
    train_lstm(train_texts, train_labels, train_weights, epochs=epochs)

    print("\nTraining polish layer...")
    train_polish_layer()

    print("\nAll training complete.")


if __name__ == "__main__":
    main()
