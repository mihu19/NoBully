import re
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "models"
WORD_PATTERN = re.compile(r"[A-Za-z']+")
WHITESPACE_PATTERN = re.compile(r"\s+")
EXIT_COMMANDS = {"exit", "quit"}


@dataclass(frozen=True)
class ModelPaths:
    bert_directory: Path
    lstm_checkpoint: Path


@dataclass(frozen=True)
class SeverityBand:
    upper_probability: float
    severity_percent: int


@dataclass(frozen=True)
class InferenceConfig:
    bert_max_tokens: int
    lstm_padding_token_id: int
    lstm_unknown_token_id: int
    lstm_embedding_dim: int
    lstm_hidden_size: int
    lstm_layer_count: int
    lstm_bidirectional: bool
    lstm_dropout_probability: float
    bert_score_weight: float
    lstm_score_weight: float
    neutral_probability_threshold: float
    flagged_word_min_probability_drop: float
    max_flagged_words: int
    severity_bands: tuple[SeverityBand, ...]


@dataclass(frozen=True)
class LoadedModels:
    tokenizer: Any
    bert_model: Any
    bert_device: torch.device
    lstm_model: "LSTMClassifier"
    lstm_vocab: dict[str, int]
    lstm_max_tokens: int
    lstm_device: torch.device


@dataclass(frozen=True)
class PhraseAnalysis:
    severity_percent: int
    toxicity_percent: int
    flagged_words: list[str]


@dataclass(frozen=True)
class WordImpact:
    word: str
    probability_drop: float


DEFAULT_MODEL_PATHS = ModelPaths(
    bert_directory=MODEL_ROOT / "bred_bert",
    lstm_checkpoint=MODEL_ROOT / "lstm_classifier.pt",
)

DEFAULT_INFERENCE_CONFIG = InferenceConfig(
    bert_max_tokens=160,
    lstm_padding_token_id=0,
    lstm_unknown_token_id=1,
    lstm_embedding_dim=128,
    lstm_hidden_size=128,
    lstm_layer_count=1,
    lstm_bidirectional=True,
    lstm_dropout_probability=0.3,
    bert_score_weight=0.6,
    lstm_score_weight=0.4,
    neutral_probability_threshold=0.20,
    flagged_word_min_probability_drop=0.04,
    max_flagged_words=8,
    severity_bands=(
        SeverityBand(upper_probability=0.25, severity_percent=10),
        SeverityBand(upper_probability=0.30, severity_percent=20),
        SeverityBand(upper_probability=0.35, severity_percent=30),
        SeverityBand(upper_probability=0.425, severity_percent=40),
        SeverityBand(upper_probability=0.50, severity_percent=50),
        SeverityBand(upper_probability=0.60, severity_percent=60),
        SeverityBand(upper_probability=0.70, severity_percent=70),
        SeverityBand(upper_probability=0.85, severity_percent=80),
        SeverityBand(upper_probability=0.95, severity_percent=90),
    ),
)


# split text into lowercase word tokens
def tokenize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall((text or "").lower())


# convert text into padded token ids for the lstm model
def encode_text_for_lstm(
    text: str,
    vocab: dict[str, int],
    max_tokens: int,
    config: InferenceConfig,
) -> list[int]:
    token_ids = [
        vocab.get(token, config.lstm_unknown_token_id)
        for token in tokenize_words(text)[:max_tokens]
    ]
    padding_needed = max_tokens - len(token_ids)

    if padding_needed > 0:
        token_ids.extend([config.lstm_padding_token_id] * padding_needed)

    return token_ids


# matches the lstm architecture used by train.py checkpoints
class LSTMClassifier(nn.Module):
    # set up the lstm layers used during inference
    def __init__(
        self,
        vocab_size: int,
        config: InferenceConfig,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            config.lstm_embedding_dim,
            padding_idx=config.lstm_padding_token_id,
        )
        self.lstm = nn.LSTM(
            input_size=config.lstm_embedding_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layer_count,
            bidirectional=config.lstm_bidirectional,
            batch_first=True,
        )
        self.is_bidirectional = config.lstm_bidirectional
        direction_count = 2 if config.lstm_bidirectional else 1
        self.dropout = nn.Dropout(config.lstm_dropout_probability)
        self.fc = nn.Linear(config.lstm_hidden_size * direction_count, 1)

    # run a batch of token ids through the lstm classifier
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        _, (hidden_states, _) = self.lstm(embeddings)

        if self.is_bidirectional:
            final_forward_state = hidden_states[-2]
            final_backward_state = hidden_states[-1]
            features = torch.cat((final_forward_state, final_backward_state), dim=1)
        else:
            features = hidden_states[-1]

        return self.fc(self.dropout(features)).squeeze(1)


# choose gpu when available, otherwise use cpu
def get_available_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load the saved bert tokenizer and classifier
def load_bert_model(paths: ModelPaths) -> tuple[Any, Any, torch.device]:
    if not paths.bert_directory.exists():
        raise FileNotFoundError(f"BERT model directory not found: {paths.bert_directory}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(paths.bert_directory)
        model = AutoModelForSequenceClassification.from_pretrained(paths.bert_directory)
    except Exception as error:
        raise RuntimeError(
            f"Could not load BERT model from: {paths.bert_directory}"
        ) from error

    device = get_available_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


# load the saved lstm checkpoint file
def load_lstm_checkpoint(paths: ModelPaths) -> dict[str, Any]:
    if not paths.lstm_checkpoint.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found: {paths.lstm_checkpoint}")

    try:
        checkpoint = torch.load(paths.lstm_checkpoint, map_location="cpu")
    except Exception as error:
        raise RuntimeError(
            f"Could not load LSTM checkpoint from: {paths.lstm_checkpoint}"
        ) from error

    if not isinstance(checkpoint, dict):
        raise RuntimeError("LSTM checkpoint has an invalid format.")

    return checkpoint


# read and validate a required checkpoint value
def require_checkpoint_value(
    checkpoint: dict[str, Any],
    key: str,
    expected_type: type,
) -> Any:
    value = checkpoint.get(key)

    if not isinstance(value, expected_type):
        raise RuntimeError(f"LSTM checkpoint is missing a valid '{key}' value.")

    return value


# build the lstm model from its saved checkpoint
def load_lstm_model(
    paths: ModelPaths,
    config: InferenceConfig,
) -> tuple[LSTMClassifier, dict[str, int], int, torch.device]:
    checkpoint = load_lstm_checkpoint(paths)
    vocab = require_checkpoint_value(checkpoint, "vocab", dict)
    max_tokens = require_checkpoint_value(checkpoint, "max_len", int)
    state_dict = require_checkpoint_value(checkpoint, "state_dict", dict)

    device = get_available_device()
    model = LSTMClassifier(
        vocab_size=len(vocab),
        config=config,
    )

    try:
        model.load_state_dict(state_dict)
    except Exception as error:
        raise RuntimeError("Could not apply the LSTM checkpoint weights.") from error

    model.to(device)
    model.eval()
    return model, vocab, max_tokens, device


# load all models needed for phrase analysis
def load_models(
    paths: ModelPaths = DEFAULT_MODEL_PATHS,
    config: InferenceConfig = DEFAULT_INFERENCE_CONFIG,
) -> LoadedModels:
    tokenizer, bert_model, bert_device = load_bert_model(paths)
    lstm_model, lstm_vocab, lstm_max_tokens, lstm_device = load_lstm_model(
        paths,
        config,
    )
    return LoadedModels(
        tokenizer=tokenizer,
        bert_model=bert_model,
        bert_device=bert_device,
        lstm_model=lstm_model,
        lstm_vocab=lstm_vocab,
        lstm_max_tokens=lstm_max_tokens,
        lstm_device=lstm_device,
    )


# predict the toxic class probability with bert
def predict_bert_probability(
    text: str,
    models: LoadedModels,
    config: InferenceConfig,
) -> float:
    encoded_text = models.tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=config.bert_max_tokens,
        return_tensors="pt",
    )
    input_ids = encoded_text["input_ids"].to(models.bert_device)
    attention_mask = encoded_text["attention_mask"].to(models.bert_device)

    with torch.no_grad():
        logits = models.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
        probability = torch.softmax(logits, dim=1)[0, 1].item()

    return float(probability)


# predict the toxic class probability with the lstm model
def predict_lstm_probability(
    text: str,
    models: LoadedModels,
    config: InferenceConfig,
) -> float:
    encoded_text = encode_text_for_lstm(
        text=text,
        vocab=models.lstm_vocab,
        max_tokens=models.lstm_max_tokens,
        config=config,
    )
    input_tensor = torch.tensor([encoded_text], dtype=torch.long).to(models.lstm_device)

    with torch.no_grad():
        logit = models.lstm_model(input_tensor)
        probability = torch.sigmoid(logit)[0].item()

    return float(probability)


# combine bert and lstm probabilities into one score
def combine_probabilities(
    bert_probability: float,
    lstm_probability: float,
    config: InferenceConfig,
) -> float:
    return (
        config.bert_score_weight * bert_probability
        + config.lstm_score_weight * lstm_probability
    )


# predict the final toxicity probability for a phrase
def predict_toxicity_probability(
    text: str,
    models: LoadedModels,
    config: InferenceConfig,
) -> float:
    bert_probability = predict_bert_probability(text, models, config)
    lstm_probability = predict_lstm_probability(text, models, config)
    return combine_probabilities(bert_probability, lstm_probability, config)


# keep a probability inside the valid zero to one range
def clamp_probability(probability: float) -> float:
    return max(0.0, min(1.0, probability))


# convert a probability into a rounded percentage
def probability_to_percent(probability: float) -> int:
    return int(round(clamp_probability(probability) * 100))


# map a toxicity probability to a severity percentage
def calculate_severity_percent(
    probability: float,
    config: InferenceConfig,
) -> int:
    probability = clamp_probability(probability)

    if probability <= config.neutral_probability_threshold:
        return 0

    for band in config.severity_bands:
        if probability < band.upper_probability:
            return band.severity_percent

    return 100


# remove all occurrences of one word from text
def remove_word_occurrences(text: str, word: str) -> str:
    cleaned_text = WORD_PATTERN.sub(
        lambda match: "" if match.group(0).lower() == word else match.group(0),
        text,
    )
    return WHITESPACE_PATTERN.sub(" ", cleaned_text).strip()


# collect unique words while keeping their first spelling
def unique_words_with_original_spelling(text: str) -> dict[str, str]:
    words: dict[str, str] = {}

    for match in WORD_PATTERN.finditer(text):
        lowercased_word = match.group(0).lower()
        words.setdefault(lowercased_word, match.group(0))

    return words


# measure how much one word changes the toxicity score
def calculate_word_impact(
    text: str,
    word: str,
    base_probability: float,
    models: LoadedModels,
    config: InferenceConfig,
) -> float:
    text_without_word = remove_word_occurrences(text, word)
    probability_without_word = predict_toxicity_probability(
        text_without_word,
        models,
        config,
    )
    return base_probability - probability_without_word


# find words that strongly increase the toxicity score
def find_flagged_words(
    text: str,
    base_probability: float,
    models: LoadedModels,
    config: InferenceConfig,
) -> list[str]:
    if base_probability <= config.neutral_probability_threshold:
        return []

    word_impacts = []
    for lowercased_word, display_word in unique_words_with_original_spelling(text).items():
        probability_drop = calculate_word_impact(
            text=text,
            word=lowercased_word,
            base_probability=base_probability,
            models=models,
            config=config,
        )

        if probability_drop >= config.flagged_word_min_probability_drop:
            word_impacts.append(
                WordImpact(word=display_word, probability_drop=probability_drop)
            )

    word_impacts.sort(key=lambda impact: impact.probability_drop, reverse=True)
    return [impact.word for impact in word_impacts[: config.max_flagged_words]]


# analyze one phrase and return printable results
def analyze_phrase(
    phrase: str,
    models: LoadedModels,
    config: InferenceConfig,
) -> PhraseAnalysis:
    toxicity_probability = predict_toxicity_probability(phrase, models, config)
    return PhraseAnalysis(
        severity_percent=calculate_severity_percent(toxicity_probability, config),
        toxicity_percent=probability_to_percent(toxicity_probability),
        flagged_words=find_flagged_words(
            text=phrase,
            base_probability=toxicity_probability,
            models=models,
            config=config,
        ),
    )


# format flagged words for command line output
def format_flagged_words(flagged_words: list[str]) -> str:
    if not flagged_words:
        return "None"

    return ", ".join(flagged_words)


# print the analysis in the requested three line format
def print_analysis(analysis: PhraseAnalysis) -> None:
    print(f"Severity: {analysis.severity_percent}%")
    print(f"Toxicity: {analysis.toxicity_percent}%")
    print(f"Flagged words: {format_flagged_words(analysis.flagged_words)}")


# check whether the user wants to stop the session
def should_exit(user_input: str) -> bool:
    return user_input.strip().lower() in EXIT_COMMANDS


# keep asking for phrases until the user exits
def run_interactive_session(
    models: LoadedModels,
    config: InferenceConfig,
) -> None:
    print("Models loaded. Type 'exit' or 'quit' to stop.")

    while True:
        phrase = input("\nEnter a phrase: ").strip()

        if should_exit(phrase):
            break

        if not phrase:
            continue

        analysis = analyze_phrase(phrase, models, config)
        print_analysis(analysis)


# load models and start the command line session
def main() -> None:
    try:
        models = load_models(DEFAULT_MODEL_PATHS, DEFAULT_INFERENCE_CONFIG)
    except FileNotFoundError as error:
        print(error)
        print("Models were not found. Please run train.py first.")
        return
    except RuntimeError as error:
        print(error)
        return

    run_interactive_session(models, DEFAULT_INFERENCE_CONFIG)


if __name__ == "__main__":
    main()
