import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_ROOT = BASE_DIR / "models"
BRED_DIR = MODEL_ROOT / "bred_bert"
LSTM_PATH = MODEL_ROOT / "lstm_classifier.pt"
LEXICON_PATH = MODEL_ROOT / "bad_words.json"
PHRASE_LEXICON_PATH = MODEL_ROOT / "bad_phrases.json"

TOXIC_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
WORD_PATTERN = re.compile(r"[A-Za-z']+")


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


def load_training_data(
    max_train_samples: int | None = None,
) -> tuple[list[str], list[int], list[str]]:
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}.")

    texts: list[str] = []
    labels: list[int] = []
    loaded_files: list[str] = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

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
                continue

        if "comment_text" not in df.columns:
            continue

        frame = df[["comment_text", "toxic"]].dropna(subset=["comment_text"]).copy()
        frame["comment_text"] = frame["comment_text"].astype(str)

        if max_train_samples is not None:
            remaining = max_train_samples - len(texts)
            if remaining <= 0:
                break
            frame = frame.head(remaining)

        texts.extend(frame["comment_text"].tolist())
        labels.extend(frame["toxic"].astype(int).tolist())
        loaded_files.append(csv_file.name)

    if not texts:
        raise RuntimeError(f"No usable training rows found in {DATA_DIR}.")

    return texts, labels, loaded_files


def build_lexicons(
    texts: list[str],
    labels: list[int],
    word_min_count: int = 25,
    word_toxic_threshold: float = 0.62,
    max_words: int = 3000,
    phrase_min_count: int = 6,
    phrase_toxic_threshold: float = 0.72,
    max_phrases: int = 4000,
    max_n: int = 4,
) -> tuple[set[str], set[str]]:

    toxic_word_counts: Counter = Counter()
    clean_word_counts: Counter = Counter()
    toxic_phrase_counts: Counter = Counter()

    for text, label in tqdm(
        zip(texts, labels),
        total=len(texts),
        desc="Lexicon pass 1/2",
        unit="text",
    ):
        tokens = tokenize_words(text)
        if not tokens:
            continue

        word_set = set(tokens)

        if label == 1:
            toxic_word_counts.update(word_set)
            ngrams: set[tuple] = set()
            for n in range(2, max_n + 1):
                if len(tokens) < n:
                    break
                ngrams.update(
                    tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
                )
            toxic_phrase_counts.update(ngrams)
        else:
            clean_word_counts.update(word_set)

    candidate_phrases: set[tuple] = {
        phrase
        for phrase, count in toxic_phrase_counts.items()
        if count >= phrase_min_count
    }
    print(f"  Candidate phrases after pass 1: {len(candidate_phrases):,}")

    clean_phrase_counts: Counter = Counter()

    for text, label in tqdm(
        zip(texts, labels),
        total=len(texts),
        desc="Lexicon pass 2/2",
        unit="text",
    ):
        if label == 1:
            continue 
        tokens = tokenize_words(text)
        if len(tokens) < 2:
            continue
        for n in range(2, max_n + 1):
            if len(tokens) < n:
                break
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i : i + n])
                if ngram in candidate_phrases:
                    clean_phrase_counts[ngram] += 1

    def score_and_filter(
        toxic_counts: Counter,
        clean_counts: Counter,
        min_count: int,
        threshold: float,
        max_items: int,
    ) -> list:
        scored = []
        for key, toxic_count in toxic_counts.items():
            total = toxic_count + clean_counts[key]
            if total < min_count:
                continue
            ratio = (toxic_count + 1) / (total + 2)
            if ratio >= threshold:
                scored.append((key, ratio, total))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return scored[:max_items]

    word_scored = score_and_filter(
        toxic_word_counts, clean_word_counts,
        word_min_count, word_toxic_threshold, max_words,
    )
    phrase_scored = score_and_filter(
        toxic_phrase_counts, clean_phrase_counts,
        phrase_min_count, phrase_toxic_threshold, max_phrases,
    )

    bad_words = {w for w, _, _ in word_scored}
    bad_phrases = {" ".join(p) for p, _, _ in phrase_scored}
    return bad_words, bad_phrases


def save_lexicon(words: set[str], phrases: set[str]) -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    with LEXICON_PATH.open("w", encoding="utf-8") as f:
        json.dump(sorted(words), f, ensure_ascii=True, indent=2)
    with PHRASE_LEXICON_PATH.open("w", encoding="utf-8") as f:
        json.dump(sorted(phrases), f, ensure_ascii=True, indent=2)

class BertDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_len: int = 160,
    ):
        self.texts = texts
        self.labels = labels
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
        }


class LSTMTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: dict[str, int],
        max_len: int = 120,
    ):
        self.texts = texts
        self.labels = labels
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
    epochs: int,
    batch_size: int = 32,
    lr: float = 2e-5,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    dataset = BertDataset(train_texts, train_labels, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"BRED/BERT Epoch {epoch}/{epochs}", unit="batch")

        for batch in progress:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss

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
    epochs: int,
    batch_size: int = 512,
    lr: float = 1e-3,
    max_len: int = 120,
) -> None:
    vocab = build_vocab(train_texts)
    dataset = LSTMTextDataset(train_texts, train_labels, vocab, max_len=max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    model = LSTMClassifier(vocab_size=len(vocab)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"LSTM Epoch {epoch}/{epochs}", unit="batch")

        for batch in progress:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                loss = criterion(model(input_ids), labels)

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
        train_texts, train_labels, loaded_files = load_training_data(max_train_samples=max_samples)
    except RuntimeError as error:
        print(error)
        return

    print(f"Loaded {len(train_texts):,} training samples from:")
    for name in loaded_files:
        print(f"  - {name}")

    print("\nBuilding toxic word/phrase lexicons...")
    bad_words, bad_phrases = build_lexicons(train_texts, train_labels)
    save_lexicon(bad_words, bad_phrases)
    print(f"Saved {len(bad_words):,} words and {len(bad_phrases):,} phrases.\n")

    print("Training BRED (DistilBERT)...")
    train_bred_bert(train_texts, train_labels, epochs=epochs)

    print("\nTraining LSTM...")
    train_lstm(train_texts, train_labels, epochs=epochs)

    print("\nAll training complete.")


if __name__ == "__main__":
    main()