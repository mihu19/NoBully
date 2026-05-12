import re
import html
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - fallback keeps the CLI usable without bs4.
    BeautifulSoup = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "models"
WORD_PATTERN = re.compile(r"[A-Za-z']+")
WHITESPACE_PATTERN = re.compile(r"\s+")
HTML_SCRIPT_STYLE_PATTERN = re.compile(r"(?is)<(script|style).*?>.*?</\1>") #removes entire <script> and <style> blocks
HTML_COMMENT_PATTERN = re.compile(r"(?is)<!--.*?-->") #removes html comments only
ANGLE_BRACKET_PATTERN = re.compile(r"<[^>]*>") #removes empty angle brackets <>, handles edge cases where html tags are not properly closed
EXIT_COMMANDS = {"exit", "quit"}
FLAGGED_WORD_MIN_SALIENCY_RATIO = 0.75
MODEL_NEGATIVE_WORD_STOP_WORDS = {
    "a",
    "about",
    "all",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "no",
    "of",
    "on",
    "or",
    "our",
    "she",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "these",
    "they",
    "this",
    "those",
    "to",
    "us",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
    "yours",
}


@dataclass(frozen=True)
class ModelPaths:
    bert_directory: Path
    lstm_checkpoint: Path
    polish_checkpoint: Path


@dataclass(frozen=True)
class SeverityBand:
    upper_probability: float
    severity_percent: int


@dataclass(frozen=True) # holds all the settings related to model inference and how probabilities are combined and interpreted
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
class PageSafetyConfig: # holds all the settings related to page analysis and blocking decisions
    max_text_characters: int
    page_chunk_word_count: int
    max_page_chunks: int
    severity_threshold_percent: int
    negative_word_threshold: int
    max_returned_negative_words: int
    model_negative_word_min_probability: float
    model_negative_word_min_probability_drop: float
    model_negative_word_min_saliency_ratio: float
    model_negative_word_candidate_limit: int
    max_negative_word_saliency_chunks: int


@dataclass(frozen=True) # holds all the loaded models and related info needed for analysis
class LoadedModels: 
    tokenizer: Any
    bert_model: Any
    bert_device: torch.device
    lstm_model: "LSTMClassifier"
    lstm_vocab: dict[str, int]
    lstm_max_tokens: int
    lstm_device: torch.device
    polish_layer: Any | None
    polish_feature_names: tuple[str, ...]
    polish_device: torch.device


@dataclass(frozen=True) # holds the results of analyzing a single phrase, used for both CLI and page analysis
class PhraseAnalysis:
    severity_percent: int
    toxicity_percent: int
    flagged_words: list[str]


@dataclass(frozen=True) # holds the results of analyzing a page snapshot, including the final block decision and details about the analysis
class PageSafetyAnalysis:
    blocked: bool
    block_reasons: list[str]
    severity_percent: int
    toxicity_percent: int
    flagged_words: list[str]
    negative_word_count: int
    negative_word_matches: list[str]
    blur_words: list[str]
    threshold_percent: int
    negative_word_threshold: int
    analyzed_character_count: int
    analyzed_word_count: int
    chunks_analyzed: int
    message: str


@dataclass(frozen=True)
class WordImpact:
    word: str
    probability_drop: float


DEFAULT_MODEL_PATHS = ModelPaths(
    bert_directory=MODEL_ROOT / "bred_bert",
    lstm_checkpoint=MODEL_ROOT / "lstm_classifier.pt",
    polish_checkpoint=MODEL_ROOT / "polish_layer.pt",
)

DEFAULT_INFERENCE_CONFIG = InferenceConfig( #matches the training setup in train.py
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

DEFAULT_PAGE_SAFETY_CONFIG = PageSafetyConfig( # holds all the settings related to page analysis and blocking decisions, can be adjusted independently from the main model inference config
    max_text_characters=50000,
    page_chunk_word_count=120,
    max_page_chunks=32,
    severity_threshold_percent=65,
    negative_word_threshold=30,
    max_returned_negative_words=200,
    model_negative_word_min_probability=0.45,
    model_negative_word_min_probability_drop=0.03,
    model_negative_word_min_saliency_ratio=0.45,
    model_negative_word_candidate_limit=120,
    max_negative_word_saliency_chunks=8,
)



def sanitize_input(text: str) -> str:
    text = text or ""
    text = HTML_SCRIPT_STYLE_PATTERN.sub(" ", text)
    text = HTML_COMMENT_PATTERN.sub(" ", text)
    text = ANGLE_BRACKET_PATTERN.sub(" ", text)
    text = html.unescape(text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()

# split text into lowercase word tokens
def tokenize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall((text or "").lower())

# remove html tags and unescape entities to get the visible text, basically parse the html into text
def html_to_text(raw: str) -> str:
    raw = raw or ""

    if BeautifulSoup is not None:
        soup = BeautifulSoup(raw, "html.parser")

        for tag in soup(["script", "style", "noscript", "template", "svg", "canvas"]):
            tag.decompose()

        return WHITESPACE_PATTERN.sub(" ", soup.get_text(" ")).strip()

    text = HTML_SCRIPT_STYLE_PATTERN.sub(" ", raw)
    text = HTML_COMMENT_PATTERN.sub(" ", text)
    text = ANGLE_BRACKET_PATTERN.sub(" ", text)
    text = html.unescape(text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


# turn a browser snapshot into compact visible text for the model
def prepare_page_text(
    html_snapshot: str | None = None,
    text_snapshot: str | None = None,
    config: PageSafetyConfig = DEFAULT_PAGE_SAFETY_CONFIG,
) -> str:
    if text_snapshot and text_snapshot.strip():
        text = sanitize_input(text_snapshot)
    else:
        text = html_to_text(html_snapshot or "")

    return text[: config.max_text_characters].strip()


# split a long page into model-sized chunks so toxic content later in the page is not missed
def split_text_for_page_analysis(
    text: str,
    config: PageSafetyConfig = DEFAULT_PAGE_SAFETY_CONFIG,
) -> list[str]:
    words = WORD_PATTERN.findall(text or "")

    if not words:
        return []

    chunks = [
        " ".join(words[start : start + config.page_chunk_word_count])
        for start in range(0, len(words), config.page_chunk_word_count)
    ]
    return chunks[: config.max_page_chunks]


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


# load the optional polish layer that calibrates the main model outputs
def load_polish_model(paths: ModelPaths) -> tuple[Any | None, tuple[str, ...], torch.device]:
    device = get_available_device()

    if not paths.polish_checkpoint.exists():
        return None, (), device

    try:
        try:
            from .polish import load_polish_layer
        except ImportError:
            from polish import load_polish_layer

        polish_layer, feature_names = load_polish_layer(
            path=paths.polish_checkpoint,
            device=device,
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not load polish layer from: {paths.polish_checkpoint}"
        ) from error

    return polish_layer, feature_names, device


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
    polish_layer, polish_feature_names, polish_device = load_polish_model(paths)
    return LoadedModels(
        tokenizer=tokenizer,
        bert_model=bert_model,
        bert_device=bert_device,
        lstm_model=lstm_model,
        lstm_vocab=lstm_vocab,
        lstm_max_tokens=lstm_max_tokens,
        lstm_device=lstm_device,
        polish_layer=polish_layer,
        polish_feature_names=polish_feature_names,
        polish_device=polish_device,
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


def predict_bert_probabilities(
    texts: list[str],
    models: LoadedModels,
    config: InferenceConfig,
) -> list[float]:
    if not texts:
        return []

    encoded_text = models.tokenizer(
        texts,
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
        probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()

    return [float(probability) for probability in probabilities]


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


def predict_lstm_probabilities(
    texts: list[str],
    models: LoadedModels,
    config: InferenceConfig,
) -> list[float]:
    if not texts:
        return []

    encoded_texts = [
        encode_text_for_lstm(
            text=text,
            vocab=models.lstm_vocab,
            max_tokens=models.lstm_max_tokens,
            config=config,
        )
        for text in texts
    ]
    input_tensor = torch.tensor(encoded_texts, dtype=torch.long).to(models.lstm_device)

    with torch.no_grad():
        logits = models.lstm_model(input_tensor)
        probabilities = torch.sigmoid(logits).detach().cpu().tolist()

    return [float(probability) for probability in probabilities]


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


def predict_base_toxicity_probability( #predicts the toxicity probability using just the main models, without the polish layer, used for measuring word impacts on the base model score
    text: str,
    models: LoadedModels,
    config: InferenceConfig,
) -> float:
    bert_probability = predict_bert_probability(text, models, config)
    lstm_probability = predict_lstm_probability(text, models, config)
    return combine_probabilities(
        bert_probability=bert_probability,
        lstm_probability=lstm_probability,
        config=config,
    )


def predict_base_toxicity_probabilities(
    texts: list[str],
    models: LoadedModels,
    config: InferenceConfig,
) -> list[float]:
    bert_probabilities = predict_bert_probabilities(texts, models, config)
    lstm_probabilities = predict_lstm_probabilities(texts, models, config)
    return [
        combine_probabilities(
            bert_probability=bert_probability,
            lstm_probability=lstm_probability,
            config=config,
        )
        for bert_probability, lstm_probability in zip(
            bert_probabilities,
            lstm_probabilities,
        )
    ]


# run the optional polish layer over raw main-model probabilities
def apply_polish_layer(
    text: str,
    bert_probability: float,
    lstm_probability: float,
    combined_probability: float,
    models: LoadedModels,
) -> float:
    if models.polish_layer is None:
        return combined_probability

    try:
        try:
            from .polish import predict_polished_probability
        except ImportError:
            from polish import predict_polished_probability

        return predict_polished_probability(
            text=text,
            bert_probability=bert_probability,
            lstm_probability=lstm_probability,
            combined_probability=combined_probability,
            polish_layer=models.polish_layer,
            feature_names=models.polish_feature_names,
            device=models.polish_device,
        )
    except Exception:
        return combined_probability


# predict the final toxicity probability for a phrase
def predict_toxicity_probability(
    text: str,
    models: LoadedModels,
    config: InferenceConfig,
) -> float:
    bert_probability = predict_bert_probability(text, models, config)
    lstm_probability = predict_lstm_probability(text, models, config)
    combined_probability = combine_probabilities(
        bert_probability=bert_probability,
        lstm_probability=lstm_probability,
        config=config,
    )
    return apply_polish_layer(
        text=text,
        bert_probability=bert_probability,
        lstm_probability=lstm_probability,
        combined_probability=combined_probability,
        models=models,
    )


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

    return probability_to_percent(probability)


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


# measure which words the BERT model used for the toxic class
def calculate_all_bert_word_saliency(
    text: str,
    models: LoadedModels,
    config: InferenceConfig,
) -> list[WordImpact]:
    word_matches = list(WORD_PATTERN.finditer(text))

    if not word_matches:
        return []

    try:
        encoded_text = models.tokenizer(
            text,
            truncation=True,
            max_length=config.bert_max_tokens,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
    except Exception:
        return []

    offset_mapping = encoded_text.pop("offset_mapping")[0].tolist()
    input_ids = encoded_text["input_ids"].to(models.bert_device)
    attention_mask = encoded_text["attention_mask"].to(models.bert_device)
    embedding_layer = models.bert_model.get_input_embeddings()
    input_embeddings = embedding_layer(input_ids).detach()
    input_embeddings.requires_grad_(True)

    try:
        with torch.enable_grad():
            models.bert_model.zero_grad(set_to_none=True)
            toxic_logit = models.bert_model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
            ).logits[0, 1]
            toxic_logit.backward()
    except Exception:
        return []

    if input_embeddings.grad is None:
        return []

    token_scores = (
        (input_embeddings.grad[0] * input_embeddings[0]).abs().sum(dim=1).detach().cpu()
    )
    word_scores: dict[str, WordImpact] = {}

    for token_index, (token_start, token_end) in enumerate(offset_mapping):
        if token_end <= token_start:
            continue

        for match in word_matches:
            overlap_start = max(token_start, match.start())
            overlap_end = min(token_end, match.end())

            if overlap_start >= overlap_end:
                continue

            lowercased_word = match.group(0).lower()
            score = float(token_scores[token_index])
            existing_impact = word_scores.get(lowercased_word)

            if existing_impact is None:
                word_scores[lowercased_word] = WordImpact(
                    word=match.group(0),
                    probability_drop=score,
                )
            else:
                word_scores[lowercased_word] = WordImpact(
                    word=existing_impact.word,
                    probability_drop=existing_impact.probability_drop + score,
                )

    if not word_scores:
        return []

    word_impacts = list(word_scores.values())
    word_impacts.sort(key=lambda impact: impact.probability_drop, reverse=True)
    return word_impacts


# measure which words the BERT model used most for the toxic class
def calculate_bert_word_saliency(
    text: str,
    models: LoadedModels,
    config: InferenceConfig,
) -> list[WordImpact]:
    word_impacts = calculate_all_bert_word_saliency(text, models, config)

    if not word_impacts:
        return []

    max_score = max(impact.probability_drop for impact in word_impacts)

    if max_score <= 0:
        return []

    filtered_impacts = [
        impact
        for impact in word_impacts
        if impact.probability_drop >= max_score * FLAGGED_WORD_MIN_SALIENCY_RATIO
    ]
    filtered_impacts.sort(key=lambda impact: impact.probability_drop, reverse=True)
    return filtered_impacts


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


def calculate_base_model_word_impact(
    text: str,
    word: str,
    base_probability: float,
    models: LoadedModels,
    config: InferenceConfig,
) -> float:
    text_without_word = remove_word_occurrences(text, word)
    probability_without_word = predict_base_toxicity_probability(
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

    saliency_impacts = calculate_bert_word_saliency(
        text=text,
        models=models,
        config=config,
    )

    if saliency_impacts:
        return [impact.word for impact in saliency_impacts[: config.max_flagged_words]]

    word_impacts = []
    unique_words = unique_words_with_original_spelling(text)

    for lowercased_word, display_word in unique_words.items():
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


def collect_word_occurrences(text: str, lowercased_word: str) -> list[str]:
    return [
        match.group(0)
        for match in WORD_PATTERN.finditer(text)
        if match.group(0).lower() == lowercased_word
    ]


def find_model_negative_words_for_chunk(
    chunk: str,
    base_probability: float,
    models: LoadedModels,
    inference_config: InferenceConfig,
    safety_config: PageSafetyConfig,
) -> list[str]:
    if base_probability < safety_config.model_negative_word_min_probability:
        return []

    saliency_impacts = calculate_all_bert_word_saliency(
        text=chunk,
        models=models,
        config=inference_config,
    )

    if saliency_impacts:
        max_saliency = max(impact.probability_drop for impact in saliency_impacts)
        saliency_cutoff = max_saliency * safety_config.model_negative_word_min_saliency_ratio
        negative_words = []

        for impact in saliency_impacts[: safety_config.model_negative_word_candidate_limit]:
            if impact.probability_drop < saliency_cutoff:
                continue

            if len(impact.word.strip("'")) < 3:
                continue

            lowercased_word = impact.word.lower()

            if lowercased_word in MODEL_NEGATIVE_WORD_STOP_WORDS:
                continue

            negative_words.extend(collect_word_occurrences(chunk, lowercased_word))

        return negative_words

    negative_words: list[str] = []
    unique_words = list(unique_words_with_original_spelling(chunk).items())
    candidates = unique_words[: safety_config.model_negative_word_candidate_limit]

    for lowercased_word, _display_word in candidates:
        if len(lowercased_word.strip("'")) < 2:
            continue

        if lowercased_word in MODEL_NEGATIVE_WORD_STOP_WORDS:
            continue

        probability_drop = calculate_base_model_word_impact(
            text=chunk,
            word=lowercased_word,
            base_probability=base_probability,
            models=models,
            config=inference_config,
        )

        if probability_drop >= safety_config.model_negative_word_min_probability_drop:
            negative_words.extend(collect_word_occurrences(chunk, lowercased_word))

    return negative_words


def summarize_model_negative_words(
    chunk_probabilities: list[tuple[str, float]],
    models: LoadedModels,
    inference_config: InferenceConfig,
    safety_config: PageSafetyConfig,
) -> tuple[int, list[str]]:
    matches: list[str] = []
    toxic_chunk_probabilities = [
        (chunk, probability)
        for chunk, probability in chunk_probabilities
        if probability >= safety_config.model_negative_word_min_probability
    ]
    toxic_chunk_probabilities.sort(key=lambda item: item[1], reverse=True)

    for chunk, probability in toxic_chunk_probabilities[
        : safety_config.max_negative_word_saliency_chunks
    ]:
        matches.extend(
            find_model_negative_words_for_chunk(
                chunk=chunk,
                base_probability=probability,
                models=models,
                inference_config=inference_config,
                safety_config=safety_config,
            )
        )

    unique_matches = sorted(set(matches))[: safety_config.max_returned_negative_words]
    return len(matches), unique_matches


# analyze a browser page snapshot and decide whether it should be blocked
def analyze_page_snapshot(
    html_snapshot: str | None,
    text_snapshot: str | None,
    models: LoadedModels,
    inference_config: InferenceConfig = DEFAULT_INFERENCE_CONFIG,
    safety_config: PageSafetyConfig = DEFAULT_PAGE_SAFETY_CONFIG,
) -> PageSafetyAnalysis:
    page_text = prepare_page_text(
        html_snapshot=html_snapshot,
        text_snapshot=text_snapshot,
        config=safety_config,
    )
    page_words = tokenize_words(page_text)
    chunks = split_text_for_page_analysis(page_text, safety_config)

    if not chunks:
        return PageSafetyAnalysis(
            blocked=False,
            block_reasons=[],
            severity_percent=0,
            toxicity_percent=0,
            flagged_words=[],
            negative_word_count=0,
            negative_word_matches=[],
            blur_words=[],
            threshold_percent=safety_config.severity_threshold_percent,
            negative_word_threshold=safety_config.negative_word_threshold,
            analyzed_character_count=0,
            analyzed_word_count=0,
            chunks_analyzed=0,
            message="No readable page text was found.",
        )

    highest_probability = 0.0
    highest_probability_chunk = chunks[0]
    toxicity_probabilities = predict_base_toxicity_probabilities(
        chunks,
        models,
        inference_config,
    )
    chunk_probabilities = list(zip(chunks, toxicity_probabilities))

    if chunk_probabilities:
        highest_probability_chunk, highest_probability = max(
            chunk_probabilities,
            key=lambda item: item[1],
        )

    severity_percent = calculate_severity_percent(
        highest_probability,
        inference_config,
    )
    toxicity_percent = probability_to_percent(highest_probability)
    negative_word_count, negative_word_matches = summarize_model_negative_words(
        chunk_probabilities=chunk_probabilities,
        models=models,
        inference_config=inference_config,
        safety_config=safety_config,
    )
    flagged_words = negative_word_matches[: inference_config.max_flagged_words]
    block_reasons = []

    # Block only based on severity percentage (map from model probability).
    if severity_percent >= safety_config.severity_threshold_percent:
        block_reasons.append("severity_threshold")

    blocked = bool(block_reasons)

    if "severity_threshold" in block_reasons:
        message = (
            "This page was blocked because its severity exceeded the configured "
            "threshold."
        )
    else:
        message = "This page was allowed."

    return PageSafetyAnalysis(
        blocked=blocked,
        block_reasons=block_reasons,
        severity_percent=severity_percent,
        toxicity_percent=toxicity_percent,
        flagged_words=flagged_words,
        negative_word_count=negative_word_count,
        negative_word_matches=negative_word_matches,
        blur_words=negative_word_matches,
        threshold_percent=safety_config.severity_threshold_percent,
        negative_word_threshold=safety_config.negative_word_threshold,
        analyzed_character_count=len(page_text),
        analyzed_word_count=len(page_words),
        chunks_analyzed=len(chunks),
        message=message,
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
        raw_phrase = input("\nEnter a phrase: ").strip()

        if should_exit(raw_phrase):
            break

        phrase = sanitize_input(raw_phrase) #added html parsing with angle bracket stripping 

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
