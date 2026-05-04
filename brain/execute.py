import json
import re
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "models"
BRED_DIR = MODEL_ROOT / "bred_bert"
LSTM_PATH = MODEL_ROOT / "lstm_classifier.pt"
LEXICON_PATH = MODEL_ROOT / "bad_words.json"
PHRASE_LEXICON_PATH = MODEL_ROOT / "bad_phrases.json"
WORD_PATTERN = re.compile(r"[A-Za-z']+")


def tokenize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall((text or "").lower())


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(token, 1) for token in tokenize_words(text)[:max_len]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


def load_lexicon() -> tuple[set[str], set[str]]:
    words = set()
    phrases = set()
    if LEXICON_PATH.exists():
        with LEXICON_PATH.open("r", encoding="utf-8") as file:
            words = set(json.load(file))
    if PHRASE_LEXICON_PATH.exists():
        with PHRASE_LEXICON_PATH.open("r", encoding="utf-8") as file:
            phrases = set(json.load(file))
    return words, phrases


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


def highlight_toxic_content(
    text: str, bad_words: set[str], bad_phrases: set[str]
) -> tuple[str, list[str], list[str]]:
    tokens = [(m.group(0), m.group(0).lower(), m.start(), m.end()) for m in WORD_PATTERN.finditer(text)]
    if not tokens:
        return text, [], []

    phrase_tuples = {tuple(phrase.split()) for phrase in bad_phrases if phrase}
    phrases_by_length = {}
    for phrase_tokens in phrase_tuples:
        phrases_by_length.setdefault(len(phrase_tokens), set()).add(phrase_tokens)
    phrase_lengths = sorted(phrases_by_length.keys(), reverse=True)

    matched_token_indexes = set()
    spans: list[tuple[int, int]] = []
    found_phrases = []
    seen_phrases = set()
    lower_tokens = [token[1] for token in tokens]

    i = 0
    while i < len(tokens):
        matched = False
        for length in phrase_lengths:
            if i + length > len(tokens):
                continue
            candidate = tuple(lower_tokens[i : i + length])
            if candidate in phrases_by_length[length]:
                start = tokens[i][2]
                end = tokens[i + length - 1][3]
                spans.append((start, end))
                for token_idx in range(i, i + length):
                    matched_token_indexes.add(token_idx)
                canonical_phrase = " ".join(candidate)
                if canonical_phrase not in seen_phrases:
                    found_phrases.append(" ".join(tokens[token_idx][0] for token_idx in range(i, i + length)))
                    seen_phrases.add(canonical_phrase)
                i += length
                matched = True
                break
        if not matched:
            i += 1

    found_words = []
    seen_words = set()
    for token_idx, (word, lower_word, start, end) in enumerate(tokens):
        if token_idx in matched_token_indexes:
            continue
        if lower_word in bad_words:
            spans.append((start, end))
            if lower_word not in seen_words:
                found_words.append(word)
                seen_words.add(lower_word)

    spans.sort(key=lambda x: x[0])
    highlighted_parts = []
    last_idx = 0
    for start, end in spans:
        if start < last_idx:
            continue
        highlighted_parts.append(text[last_idx:start])
        highlighted_parts.append(f"\033[91m{text[start:end]}\033[0m")
        last_idx = end
    highlighted_parts.append(text[last_idx:])
    highlighted = "".join(highlighted_parts)
    return highlighted, found_phrases, found_words


def score_toxicity(
    probability: float,
    phrase_match_count: int,
    word_match_count: int,
    token_count: int,
    neutral_threshold: float = 0.20,
) -> tuple[int, int]:
    probability = max(0.0, min(1.0, probability))

    if token_count <= 0:
        token_count = 1

    model_component = probability * 8.0
    lexicon_density = (phrase_match_count * 2 + word_match_count) / token_count
    lexicon_component = min(2.0, lexicon_density * 3.5)
    raw_score = max(0.0, min(10.0, model_component + lexicon_component))

    has_lexicon_match = (phrase_match_count + word_match_count) > 0
    if not has_lexicon_match and probability < neutral_threshold:
        return 0, int(round(raw_score * 10))

    severity = int(round(raw_score))
    severity = max(1, min(10, severity))
    return severity, int(round(raw_score * 10))


def main() -> None:
    tokenizer, bred_model, bred_device = load_bred_bert()
    lstm_model, vocab, max_len, lstm_device = load_lstm_model()
    if tokenizer is None or bred_model is None or lstm_model is None:
        print("Models were not found. Please run train.py first.")
        return

    bad_words, bad_phrases = load_lexicon()
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
        highlighted_text, found_phrases, found_words = highlight_toxic_content(phrase, bad_words, bad_phrases)
        has_highlights = bool(found_phrases or found_words)
        token_count = len(tokenize_words(phrase))
        severity, toxicity_percent = score_toxicity(
            combined_prob,
            phrase_match_count=len(found_phrases),
            word_match_count=len(found_words),
            token_count=token_count,
        )

        print(f"Severity (0-10): {severity}")
        print(f"Toxicity: {toxicity_percent}%")
        if has_highlights:
            print(f"Highlighted: {highlighted_text}")
        if found_phrases:
            print(f"Detected bad phrases: {', '.join(found_phrases)}")
        else:
            print("Detected bad phrases: none")
        if found_words:
            print(f"Detected bad words: {', '.join(found_words)}")
        else:
            print("Detected bad words: none")


if __name__ == "__main__":
    main()