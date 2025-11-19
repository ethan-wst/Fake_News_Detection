import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# CONFIGURATION

LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

TRUE_INDEX = 5
MOSTLY_TRUE_INDEX = 4

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
    p_true = probabilities[TRUE_INDEX]
    p_mostly_true = probabilities[MOSTLY_TRUE_INDEX]
    combined = float(p_true + p_mostly_true)
    return combined, float(p_true), float(p_mostly_true)


def gate_decision(confidence):
    """
    keep / review / reject
    """
    if confidence >= KEEP_THRESHOLD:
        return "keep"
    elif confidence <= REVIEW_THRESHOLD:
        return "reject"
    else:
        return "review"


# MAIN INFERENCE LOGIC

def evaluate_statement(model, tokenizer, statement):
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

    # Apply thresholding logic
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
