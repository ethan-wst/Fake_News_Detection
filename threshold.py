"""
Threshold-based filtering system for fake news detection.

This module implements a gating mechanism that classifies statements into three categories:
- keep: High confidence statements (true, mostly-true)
- review: Medium confidence statements requiring human review
- reject: Low confidence statements (pants-fire, false)

The system uses a RoBERTa model to predict truthfulness probabilities and applies
threshold-based logic to make filtering decisions.
"""

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# CONFIGURATION

LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

#Indices for most truthful classes (used for confidence calculation)
TRUE_INDEX = 5
MOSTLY_TRUE_INDEX = 4

#Thresholds for gating decisions
KEEP_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.30

MODEL_PATH = "best_roberta_liar_model.pt"
MODEL_NAME = "roberta-base"
MAX_LEN = 128


# LOAD MODEL + TOKENIZER

def load_roberta_model():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=6
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    model.eval()  

    return model, tokenizer


# TRUTH PROBABILITY LOGIC

def compute_truth_confidence(probabilities):
    """
    Calculate confidence score from class probabilities.
    Combines the probabilities of "true" and "mostly-true" classes to create
    a single confidence metric representing how truthful the statement is.
    """
    p_true = probabilities[TRUE_INDEX]
    p_mostly_true = probabilities[MOSTLY_TRUE_INDEX]
    combined = float(p_true + p_mostly_true)
    return combined, float(p_true), float(p_mostly_true)


def gate_decision(confidence):
    """
    Make a filtering decision based on confidence threshold.
    """
    if confidence >= KEEP_THRESHOLD:
        return "keep"
    elif confidence <= REVIEW_THRESHOLD:
        return "reject"
    else:
        return "review"


# MAIN INFERENCE LOGIC

def evaluate_statement(model, tokenizer, statement):
    """
    Evaluate a statement and return truthfulness prediction with gating decision.
    """
    encoding = tokenizer(
        statement,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )
        logits = outputs.logits[0]
        probabilities = F.softmax(logits, dim=-1).tolist()

    # Apply thresholding logic to make gating decision
    confidence, p_true, p_mostly = compute_truth_confidence(probabilities)
    decision = gate_decision(confidence)

    return {
        "probabilities": {label: round(prob, 4) for label, prob in zip(LABELS, probabilities)},
        "p_true": p_true,
        "p_mostly_true": p_mostly,
        "combined_confidence": confidence,
        "decision": decision
    }


# DEMO USAGE

if __name__ == "__main__":
    model, tokenizer = load_roberta_model()

    text = " This statement is certainly true"

    result = evaluate_statement(model, tokenizer, text)

    print("\n=== Roberta Ingestion Gate Output ===")
    for k, v in result.items():
        print(f"{k}: {v}")
