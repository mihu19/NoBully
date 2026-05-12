# Models

This folder stores the trained model artifacts used by the NoBully backend during text analysis.

## Contents

```text
models/
├── bred_bert/
├── lstm_classifier.pt
└── polish_layer.pt
```

## What the models do

NoBully does not rely on only one model. It uses an ensemble-style pipeline:

```text
input text
   |
   v
text cleaning and chunking
   |
   v
BERT classifier + LSTM classifier
   |
   v
weighted probability combination
   |
   v
optional polish layer calibration
   |
   v
toxicity probability
   |
   v
severity, flagged words, blur words, and block decision
```

The goal is to decide how likely a piece of text is to be toxic, insulting, threatening, or harmful.

## Model 1: BRED BERT

The `bred_bert/` folder contains the transformer-based model.

```text
bred_bert/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

The backend loads this model with Hugging Face Transformers using:

```python
AutoTokenizer.from_pretrained(...)
AutoModelForSequenceClassification.from_pretrained(...)
```

### How the BERT algorithm works

BERT is a transformer model. Instead of reading text word by word like a simple sequence model, it looks at the whole sentence and learns relationships between words using attention.

For example, these two sentences contain similar words, but they should not be judged the same way:

```text
that word is offensive
you are offensive
```

A transformer can look at context around each word. This helps the model understand whether a word is used as an explanation, a quote, an insult, or a direct attack.

### How NoBully uses BERT

During inference, the backend:

1. tokenizes the text with the saved tokenizer
2. truncates or pads the text to the configured maximum length
3. sends `input_ids` and `attention_mask` into the BERT classifier
4. receives raw class scores called logits
5. applies `softmax`
6. reads the probability of class `1`, which represents the toxic class

The simplified logic is:

```python
encoded_text = tokenizer(
    text,
    truncation=True,
    padding="max_length",
    max_length=160,
    return_tensors="pt"
)

logits = bert_model(
    input_ids=encoded_text["input_ids"],
    attention_mask=encoded_text["attention_mask"]
).logits

bert_probability = softmax(logits)[0, 1]
```

So if BERT returns `0.82`, that means the BERT model thinks the text has an 82 percent chance of being toxic.

### Why BERT is useful here

BERT is the strongest model in this project because it understands context better than a simple word counter. It can learn patterns such as:

- direct insults
- threats
- harassment phrases
- toxic sentence structure
- harmful meaning that depends on context

In the final score, BERT has the higher weight.

## Model 2: LSTM classifier

The `lstm_classifier.pt` file contains the saved LSTM model checkpoint.

The checkpoint stores:

- the model weights
- the vocabulary
- the maximum sequence length

### How the LSTM algorithm works

LSTM means Long Short-Term Memory. It is a recurrent neural network made to read sequences.

Unlike BERT, which looks at the whole text using attention, an LSTM processes the text as a sequence of tokens. It keeps an internal memory while reading. This lets it detect patterns across word order.

For example:

```text
you are ...
I will ...
go and ...
```

The model can learn that certain word sequences are more suspicious than isolated words.

### How NoBully prepares text for the LSTM

The backend:

1. extracts lowercase words using a regular expression
2. converts each word into an integer id from the saved vocabulary
3. uses the unknown token id for words not in the vocabulary
4. cuts the sequence to the maximum token length
5. pads shorter sequences with the padding token id

Example:

```text
you are bad
```

becomes something like:

```text
[25, 91, 348, 0, 0, 0, ...]
```

Here, `0` is padding.

### LSTM architecture

The model architecture is:

```text
word ids
   |
   v
embedding layer
   |
   v
bidirectional LSTM
   |
   v
forward final state + backward final state
   |
   v
dropout
   |
   v
linear layer
   |
   v
single toxicity logit
   |
   v
sigmoid probability
```

The default inference configuration uses:

```text
embedding dimension: 128
hidden size: 128
layers: 1
bidirectional: true
dropout: 0.3
```

The backend applies `sigmoid` to the LSTM output because the LSTM produces one binary-classification logit.

Simplified logic:

```python
logit = lstm_model(input_tensor)
lstm_probability = sigmoid(logit)
```

So if the LSTM returns `0.70`, that means the LSTM model thinks the text has a 70 percent chance of being toxic.

### Why the LSTM is useful here

The LSTM is simpler than BERT, but it gives a second opinion. It is useful because it can learn repeated word-order patterns and can sometimes catch phrases that the BERT model may score too weakly.

## Combining BERT and LSTM

After both models return probabilities, NoBully combines them using a weighted average:

```text
combined_probability = 0.6 * bert_probability + 0.4 * lstm_probability
```

BERT receives 60 percent of the weight.

LSTM receives 40 percent of the weight.

Example:

```text
bert_probability = 0.80
lstm_probability = 0.65
```

Then:

```text
combined_probability = 0.6 * 0.80 + 0.4 * 0.65
combined_probability = 0.48 + 0.26
combined_probability = 0.74
```

So the combined toxicity probability is 74 percent.

This combined score is the base toxicity score before the polish layer.

## Model 3: polish layer

The `polish_layer.pt` file stores a smaller correction model.

It is called the polish layer because it polishes or calibrates the raw output from BERT and LSTM.

### Why the polish layer exists

The main models can sometimes overreact to words that look offensive but are not actually being used as harassment.

Example:

```text
this game crashed and killed my save file
```

The word `killed` can look dangerous, but the sentence is about a game or software problem, not a real threat.

The polish layer helps reduce these false positives by looking at extra features around the text and the model scores.

### How the polish layer algorithm works

The polish layer is a small neural network:

```text
input features
   |
   v
linear layer
   |
   v
ReLU activation
   |
   v
linear layer
   |
   v
corrected toxicity score
```

It receives engineered features, not raw text only.

These features include the main model probabilities and contextual signals, such as whether the text looks like:

- direct second-person harassment
- a threat
- an insult
- profanity in a harmless or positive context
- object or software-related context
- technical failure context
- neutral context

The polish layer is trained to output a target probability that is more careful than the raw combined model score.

### Polish layer fallback behavior

The polish layer is optional.

If `polish_layer.pt` is missing or fails during prediction, the backend simply uses the combined BERT and LSTM probability.

This means the project can still work without the polish layer, but the final score may be less calibrated.

## Page analysis

The backend does not analyze very long pages as one huge text block. It splits a page into smaller word chunks.

The default page analysis settings are:

```text
maximum page text characters: 50000
words per chunk: 120
maximum chunks: 32
default severity threshold: 65
```

Each chunk is analyzed separately. This prevents toxic content near the end of a long page from being ignored.

## Severity calculation

The final toxicity probability is converted into a severity percentage.

The configured severity bands are approximately:

```text
probability <= 0.25  -> 10 severity
probability <= 0.30  -> 20 severity
probability <= 0.35  -> 30 severity
probability <= 0.425 -> 40 severity
probability <= 0.50  -> 50 severity
probability <= 0.60  -> 60 severity
probability <= 0.70  -> 70 severity
probability <= 0.85  -> 80 severity
probability <= 0.95  -> 90 severity
probability >  0.95  -> 100 severity
```

The extension and backend can then use this severity score to decide whether a page should be allowed, blurred, warned, or blocked.

## Flagged words

NoBully also tries to identify which words contributed most to the toxic prediction.

For the BERT model, the backend measures word saliency. In simple terms, it checks how much the toxic score depends on certain words. If removing or weakening a word greatly lowers the toxicity probability, that word is treated as important.

This helps the extension know which words should be blurred.

## Training process

Training is handled mainly by `backend/brain/train.py`.

The training pipeline is:

```text
load CSV datasets
   |
   v
normalize text and labels
   |
   v
compute sample weights
   |
   v
train BERT classifier
   |
   v
train LSTM classifier
   |
   v
train polish layer
   |
   v
save models into backend/brain/models/
```

### BERT training

The BERT classifier is trained using:

```text
base model: distilbert-base-uncased
classification labels: 2
optimizer: AdamW
loss: weighted cross entropy
default learning rate: 2e-5
default batch size: 32
default max tokens: 160
```

The model is saved into:

```text
backend/brain/models/bred_bert/
```

### LSTM training

The LSTM classifier is trained using:

```text
vocabulary max size: 50000
minimum word frequency: 2
embedding dimension: 128
hidden size: 128
loss: weighted binary cross entropy with logits
optimizer: Adam
default learning rate: 1e-3
default batch size: 512
default max tokens: 120
```

The checkpoint is saved into:

```text
backend/brain/models/lstm_classifier.pt
```

### Polish layer training

The polish layer is trained after the main models exist.

It uses examples with target probabilities and engineered features. It learns to correct the raw model score so harmless contexts are less likely to be blocked while truly harmful contexts remain high.

The checkpoint is saved into:

```text
backend/brain/models/polish_layer.pt
```

## Inference summary

When the backend receives text, the final model flow is:

```text
1. clean the text
2. split long page text into chunks
3. run BERT on each chunk
4. run LSTM on each chunk
5. calculate weighted probability
6. apply polish layer if available
7. convert probability into toxicity percent
8. convert probability into severity percent
9. calculate important flagged words
10. decide whether to blur or block
```

## Important notes

- `bred_bert/` must contain the tokenizer and model files together.
- `lstm_classifier.pt` must match the same architecture used in the code.
- `polish_layer.pt` is optional, but recommended.
- The models are local files and are loaded when the backend starts.
- If model files are missing, the backend may fail to start or may fall back to less calibrated scoring.
- The model output is probabilistic, not perfect.
- False positives and false negatives are still possible.
