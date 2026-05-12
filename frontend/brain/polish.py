import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import execute


BASE_DIR = Path(__file__).resolve().parent
CURATED_DATA_DIR = BASE_DIR / "curated_data"
MODEL_ROOT = BASE_DIR / "models"
POLISH_DATA_PATH = CURATED_DATA_DIR / "polish_false_positives.csv"
POLISH_LAYER_PATH = MODEL_ROOT / "polish_layer.pt"
WORD_PATTERN = re.compile(r"[A-Za-z']+")

POLISH_FEATURE_NAMES = (
    "bert_probability",
    "lstm_probability",
    "combined_probability",
    "probability_gap",
    "combined_probability_squared",
    "profanity_count",
    "positive_count",
    "insult_count",
    "threat_count",
    "second_person_context",
    "directed_profanity_context",
    "positive_profanity_context",
    "object_context",
    "quote_context",
    "complaint_context",
    "token_count",
)

PROFANITY_WORDS = {
    "bullshit",
    "crap",
    "dammit",
    "damn",
    "fuck",
    "fucked",
    "fucker",
    "fuckers",
    "fucking",
    "hell",
    "shit",
    "shitty",
    "wtf",
}
POSITIVE_CONTEXT_WORDS = {
    "amazing",
    "awesome",
    "beautiful",
    "best",
    "brilliant",
    "charming",
    "clean",
    "cool",
    "creative",
    "excellent",
    "fantastic",
    "fresh",
    "fun",
    "funny",
    "good",
    "great",
    "love",
    "nice",
    "perfect",
    "smart",
    "smooth",
    "solid",
    "sweet",
    "wild",
}
INSULT_TARGET_PROBABILITIES = {
    "clown": 0.55,
    "donkey": 0.60,
    "dumb": 0.60,
    "dummy": 0.55,
    "fool": 0.55,
    "idiot": 0.70,
    "jerk": 0.60,
    "loser": 0.60,
    "moron": 0.70,
    "pathetic": 0.70,
    "stupid": 0.65,
    "trash": 0.70,
    "ugly": 0.60,
    "worthless": 0.75,
}
INSULT_WORDS = set(INSULT_TARGET_PROBABILITIES)
THREAT_WORDS = {
    "beat",
    "destroy",
    "harm",
    "hurt",
    "kill",
    "killed",
    "killing",
    "murder",
    "murdered",
    "smash",
}
SECOND_PERSON_WORDS = {"you", "your", "yours", "yourself"}
OBJECT_CONTEXT_WORDS = {
    "app",
    "battery",
    "browser",
    "bug",
    "build",
    "camera",
    "code",
    "coffee",
    "compiler",
    "demo",
    "download",
    "episode",
    "exam",
    "feature",
    "game",
    "homework",
    "keyboard",
    "laptop",
    "level",
    "movie",
    "phone",
    "photo",
    "printer",
    "puzzle",
    "recipe",
    "router",
    "scene",
    "script",
    "server",
    "song",
    "test",
    "tests",
    "traffic",
    "upload",
    "weather",
}
QUOTE_CONTEXT_WORDS = {
    "article",
    "book",
    "class",
    "dictionary",
    "discussed",
    "explained",
    "guide",
    "lesson",
    "mentioned",
    "moderation",
    "quoted",
    "transcript",
    "word",
}
COMPLAINT_CONTEXT_WORDS = {
    "again",
    "broke",
    "crashed",
    "failed",
    "fixable",
    "froze",
    "lost",
    "restarted",
    "slow",
    "stopped",
    "timed",
}


@dataclass(frozen=True)
class PolishExample:
    text: str
    target_probability: float
    sample_weight: float


class PolishLayer(nn.Module):
    def __init__(self, feature_count: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def tokenize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall((text or "").lower())


def clamp_probability(probability: float) -> float:
    return max(0.0, min(1.0, float(probability)))


def capped_count(words: list[str], vocabulary: set[str], cap: int = 4) -> float:
    count = sum(1 for word in words if word in vocabulary)
    return min(count, cap) / cap


def has_directed_profanity(words: list[str]) -> bool:
    return any(
        word in PROFANITY_WORDS
        and index + 1 < len(words)
        and words[index + 1] in SECOND_PERSON_WORDS
        for index, word in enumerate(words)
    )


def has_positive_profanity(words: list[str]) -> bool:
    return any(word in PROFANITY_WORDS for word in words) and any(
        word in POSITIVE_CONTEXT_WORDS for word in words
    )


def directed_insult_cap(words: list[str]) -> float | None:
    if not any(word in SECOND_PERSON_WORDS for word in words):
        return None

    insult_scores = [
        INSULT_TARGET_PROBABILITIES[word]
        for word in words
        if word in INSULT_TARGET_PROBABILITIES
    ]

    if not insult_scores:
        return None

    repeated_insult_boost = 0.10 * (len(insult_scores) - 1)
    profanity_boost = 0.10 if any(word in PROFANITY_WORDS for word in words) else 0.0
    return min(0.85, max(insult_scores) + repeated_insult_boost + profanity_boost)


def build_polish_feature_map(
    text: str,
    bert_probability: float,
    lstm_probability: float,
    combined_probability: float,
) -> dict[str, float]:
    words = tokenize_words(text)
    combined_probability = clamp_probability(combined_probability)
    bert_probability = clamp_probability(bert_probability)
    lstm_probability = clamp_probability(lstm_probability)

    return {
        "bert_probability": bert_probability,
        "lstm_probability": lstm_probability,
        "combined_probability": combined_probability,
        "probability_gap": abs(bert_probability - lstm_probability),
        "combined_probability_squared": combined_probability * combined_probability,
        "profanity_count": capped_count(words, PROFANITY_WORDS),
        "positive_count": capped_count(words, POSITIVE_CONTEXT_WORDS),
        "insult_count": capped_count(words, INSULT_WORDS),
        "threat_count": capped_count(words, THREAT_WORDS),
        "second_person_context": float(any(word in SECOND_PERSON_WORDS for word in words)),
        "directed_profanity_context": float(has_directed_profanity(words)),
        "positive_profanity_context": float(has_positive_profanity(words)),
        "object_context": float(any(word in OBJECT_CONTEXT_WORDS for word in words)),
        "quote_context": float(any(word in QUOTE_CONTEXT_WORDS for word in words)),
        "complaint_context": float(any(word in COMPLAINT_CONTEXT_WORDS for word in words)),
        "token_count": min(len(words), 30) / 30,
    }


def build_polish_features(
    text: str,
    bert_probability: float,
    lstm_probability: float,
    combined_probability: float,
    feature_names: tuple[str, ...] = POLISH_FEATURE_NAMES,
) -> list[float]:
    feature_map = build_polish_feature_map(
        text=text,
        bert_probability=bert_probability,
        lstm_probability=lstm_probability,
        combined_probability=combined_probability,
    )
    return [feature_map[name] for name in feature_names]


def row_float(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)

    if value is None or pd.isna(value):
        return default

    try:
        return float(value)
    except ValueError:
        return default


def load_false_positive_examples(path: Path = POLISH_DATA_PATH) -> list[PolishExample]:
    if not path.exists():
        raise FileNotFoundError(f"Polish data file not found: {path}")

    frame = pd.read_csv(path)

    if "comment_text" not in frame.columns and "text" in frame.columns:
        frame = frame.rename(columns={"text": "comment_text"})

    if "comment_text" not in frame.columns:
        raise RuntimeError(f"Polish data file has no comment_text column: {path}")

    examples = []
    for _, row in frame.dropna(subset=["comment_text"]).iterrows():
        target_probability = row_float(row, "target_toxicity", default=0.05)
        category = str(row.get("category", ""))
        sample_weight = max(0.1, row_float(row, "sample_weight", default=1.8))

        if category in {"quoted_or_meta_context", "idiom_or_meta_context"}:
            sample_weight *= 3.0
        elif category in {"positive_intensifier", "profane_frustration"}:
            sample_weight *= 2.0

        examples.append(
            PolishExample(
                text=str(row["comment_text"]),
                target_probability=clamp_probability(target_probability),
                sample_weight=sample_weight,
            )
        )

    return examples


def row_is_positive_anchor(row: pd.Series) -> bool:
    for column in ("toxic", "threat", "insult", "harassment", "identity_attack"):
        if row_float(row, column) > 0:
            return True

    return False


def positive_target_for_row(row: pd.Series) -> float:
    severity = row_float(row, "severity", default=-1.0)

    if severity >= 0:
        return max(0.55, min(0.98, severity / 10))

    if row_float(row, "threat") > 0:
        return 0.95

    if row_float(row, "harassment") > 0 or row_float(row, "identity_attack") > 0:
        return 0.88

    if row_float(row, "insult") > 0:
        return 0.72

    return 0.85


def load_curated_positive_anchors() -> list[PolishExample]:
    examples = []

    for csv_file in sorted(CURATED_DATA_DIR.glob("*.csv")):
        if csv_file == POLISH_DATA_PATH:
            continue

        frame = pd.read_csv(csv_file)
        if "comment_text" not in frame.columns and "text" in frame.columns:
            frame = frame.rename(columns={"text": "comment_text"})

        if "comment_text" not in frame.columns:
            continue

        for _, row in frame.dropna(subset=["comment_text"]).iterrows():
            if not row_is_positive_anchor(row):
                continue

            examples.append(
                PolishExample(
                    text=str(row["comment_text"]),
                    target_probability=positive_target_for_row(row),
                    sample_weight=max(1.5, row_float(row, "sample_weight", default=2.5)),
                )
            )

    return examples


def generate_positive_anchor_examples() -> list[PolishExample]:
    insults = ("stupid", "idiot", "loser", "worthless", "donkey", "moron")
    threats = (
        "I will hurt you",
        "I will find you after school",
        "watch your back when you leave",
        "I am going to smash your face",
    )
    examples = [
        PolishExample("fuck you", 0.95, 6.0),
        PolishExample("you are a stupid donkey", 0.75, 7.0),
        PolishExample("you are a donkey", 0.60, 7.0),
        PolishExample("you are fucking stupid", 0.80, 6.0),
    ]

    for insult in insults:
        examples.append(PolishExample(f"you are {insult}", 0.68, 5.0))
        examples.append(PolishExample(f"everyone thinks you are {insult}", 0.76, 5.0))

    for threat in threats:
        examples.append(PolishExample(threat, 0.95, 4.0))

    return examples


def load_polish_examples() -> list[PolishExample]:
    return (
        load_false_positive_examples()
        + load_curated_positive_anchors()
        + generate_positive_anchor_examples()
    )


def predict_raw_model_probabilities(
    text: str,
    models: execute.LoadedModels,
    config: execute.InferenceConfig,
) -> tuple[float, float, float]:
    bert_probability = execute.predict_bert_probability(text, models, config)
    lstm_probability = execute.predict_lstm_probability(text, models, config)
    combined_probability = execute.combine_probabilities(
        bert_probability=bert_probability,
        lstm_probability=lstm_probability,
        config=config,
    )
    return bert_probability, lstm_probability, combined_probability


def build_training_tensors(
    examples: list[PolishExample],
    models: execute.LoadedModels,
    config: execute.InferenceConfig,
    feature_names: tuple[str, ...] = POLISH_FEATURE_NAMES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = []
    targets = []
    weights = []

    for example in tqdm(examples, desc="Scoring polish examples", unit="row"):
        bert_probability, lstm_probability, combined_probability = (
            predict_raw_model_probabilities(example.text, models, config)
        )
        features.append(
            build_polish_features(
                text=example.text,
                bert_probability=bert_probability,
                lstm_probability=lstm_probability,
                combined_probability=combined_probability,
                feature_names=feature_names,
            )
        )
        targets.append(example.target_probability)
        weights.append(example.sample_weight)

    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


def train_polish_layer(
    models: execute.LoadedModels | None = None,
    config: execute.InferenceConfig = execute.DEFAULT_INFERENCE_CONFIG,
    paths: execute.ModelPaths = execute.DEFAULT_MODEL_PATHS,
    output_path: Path = POLISH_LAYER_PATH,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    hidden_size: int = 32,
) -> Path:
    if models is None:
        models = execute.load_models(paths, config)

    examples = load_polish_examples()
    device = execute.get_available_device()
    feature_names = POLISH_FEATURE_NAMES
    features, targets, weights = build_training_tensors(
        examples=examples,
        models=models,
        config=config,
        feature_names=feature_names,
    )
    dataset = TensorDataset(features, targets, weights)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    layer = PolishLayer(feature_count=len(feature_names), hidden_size=hidden_size).to(device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss(reduction="none")

    layer.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for batch_features, batch_targets, batch_weights in dataloader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            batch_weights = batch_weights.to(device)

            optimizer.zero_grad()
            probabilities = torch.sigmoid(layer(batch_features))
            loss = (criterion(probabilities, batch_targets) * batch_weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            average_loss = total_loss / max(1, len(dataloader))
            print(f"Polish layer epoch {epoch}/{epochs} - avg loss: {average_loss:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": layer.state_dict(),
            "feature_names": feature_names,
            "hidden_size": hidden_size,
        },
        output_path,
    )
    print(f"Saved polish layer to {output_path}")
    return output_path


def load_polish_layer(
    path: Path = POLISH_LAYER_PATH,
    device: torch.device | None = None,
) -> tuple[PolishLayer, tuple[str, ...]]:
    if device is None:
        device = execute.get_available_device()

    checkpoint: dict[str, Any] = torch.load(path, map_location=device)
    feature_names = tuple(checkpoint.get("feature_names", POLISH_FEATURE_NAMES))
    hidden_size = int(checkpoint.get("hidden_size", 16))
    layer = PolishLayer(feature_count=len(feature_names), hidden_size=hidden_size)
    layer.load_state_dict(checkpoint["state_dict"])
    layer.to(device)
    layer.eval()
    return layer, feature_names


def predict_polished_probability(
    text: str,
    bert_probability: float,
    lstm_probability: float,
    combined_probability: float,
    polish_layer: PolishLayer,
    feature_names: tuple[str, ...],
    device: torch.device,
) -> float:
    feature_map = build_polish_feature_map(
        text=text,
        bert_probability=bert_probability,
        lstm_probability=lstm_probability,
        combined_probability=combined_probability,
    )
    features = build_polish_features(
        text=text,
        bert_probability=bert_probability,
        lstm_probability=lstm_probability,
        combined_probability=combined_probability,
        feature_names=feature_names,
    )
    feature_tensor = torch.tensor([features], dtype=torch.float32).to(device)

    with torch.no_grad():
        probability = torch.sigmoid(polish_layer(feature_tensor))[0].item()

    words = tokenize_words(text)
    insult_cap = directed_insult_cap(words)

    if feature_map["quote_context"] and feature_map["threat_count"] == 0:
        probability = min(probability, 0.10)

    if (
        feature_map["positive_profanity_context"]
        and feature_map["insult_count"] == 0
        and feature_map["threat_count"] == 0
    ):
        probability = min(probability, 0.12)

    if (
        feature_map["object_context"]
        and feature_map["second_person_context"] == 0
        and feature_map["threat_count"] == 0
    ):
        probability = min(probability, 0.15)

    if insult_cap is not None and feature_map["threat_count"] == 0:
        probability = min(probability, insult_cap)

    return clamp_probability(probability)


def main() -> None:
    train_polish_layer()


if __name__ == "__main__":
    main()
