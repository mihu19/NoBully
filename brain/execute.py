import re
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "models"
BRED_DIR = MODEL_ROOT / "bred_bert"
LSTM_PATH = MODEL_ROOT / "lstm_classifier.pt"
WORD_PATTERN = re.compile(r"[A-Za-z']+")
NEUTRAL_THRESHOLD = 0.20
BULLY_THRESHOLD = 0.50
SEVERITY_BANDS = (
    (0.25, 1),
    (0.30, 2),
    (0.35, 3),
    (0.425, 4),
    (BULLY_THRESHOLD, 5),
    (0.60, 6),
    (0.70, 7),
    (0.85, 8),
    (0.95, 9),
)


def tokenize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall((text or "").lower())


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(token, 1) for token in tokenize_words(text)[:max_len]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


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
        features = self.dropout(features)
        return self.fc(features).squeeze(1)


def load_bred_bert():
    if not BRED_DIR.exists():
        return None, None, None
    tokenizer = AutoTokenizer.from_pretrained(BRED_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BRED_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def load_lstm_model():
    if not LSTM_PATH.exists():
        return None, None, None, None
    checkpoint = torch.load(LSTM_PATH, map_location="cpu")
    vocab = checkpoint["vocab"]
    max_len = checkpoint["max_len"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(vocab_size=len(vocab))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, max_len, device


def predict_bred_probability(text: str, tokenizer, model, device) -> float:
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=160,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    return float(prob)


def predict_lstm_probability(text: str, model, vocab, max_len: int, device) -> float:
    encoded = torch.tensor([encode_text(text, vocab, max_len)], dtype=torch.long).to(device)
    with torch.no_grad():
        logit = model(encoded)
        prob = torch.sigmoid(logit)[0].item()
    return float(prob)


def score_toxicity(
    probability: float,
    neutral_threshold: float = NEUTRAL_THRESHOLD,
) -> tuple[int, int]:
    probability = max(0.0, min(1.0, probability))
    toxicity_percent = int(round(probability * 100))
    if probability <= neutral_threshold:
        return 0, toxicity_percent

    for upper_bound, severity in SEVERITY_BANDS:
        if probability < upper_bound:
            return severity, toxicity_percent

    return 10, toxicity_percent


def main() -> None:
    tokenizer, bred_model, bred_device = load_bred_bert()
    lstm_model, vocab, max_len, lstm_device = load_lstm_model()
    if tokenizer is None or bred_model is None or lstm_model is None:
        print("Models were not found. Please run train.py first.")
        return

    print("Models loaded. Type 'exit' to quit.")

    while True:
        phrase = input("\nEnter a phrase: ").strip()
        if phrase.lower() == "exit":
            break
        if not phrase:
            continue

        bred_prob = predict_bred_probability(phrase, tokenizer, bred_model, bred_device)
        lstm_prob = predict_lstm_probability(phrase, lstm_model, vocab, max_len, lstm_device)
        combined_prob = 0.6 * bred_prob + 0.4 * lstm_prob
        severity, toxicity_percent = score_toxicity(combined_prob)

        print(f"Severity (0-10): {severity}")
        print(f"Toxicity: {toxicity_percent}%")


if __name__ == "__main__":
    main()
