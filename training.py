import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Dataset Class: Holds statements & labels
class LIARDataset(Dataset):
    def __init__(self, statements, labels):
        self.statements = statements
        self.labels = labels

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, idx):
        return {
            'text': str(self.statements[idx]),
            'label': self.labels[idx]
        }

# Load Function: Loads passed dataset, gets only statement & lables, cleans columns
def load_liar_data(file_path):

    # Read TSV with no header, specify only the columns we need
    df = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        usecols=[1, 2],  # Only read columns 2 and 3 (0-indexed: 1 and 2)
        names=['label', 'statement']
    )

    # Clean text
    df['statement'] = df['statement'].str.strip()
    df['statement'] = df['statement'].fillna('')

    # Map labels to integers (0-5)
    label_map = {
        'pants-fire': 0,
        'false': 1,
        'barely-true': 2,
        'half-true': 3,
        'mostly-true': 4,
        'true': 5
    }
    df['label_encoded'] = df['label'].map(label_map)

    print(f"Loaded {len(df)} samples from {file_path}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    return df

# Implements RoBERTa tokenizer, encodes inputs into tensors/vectors
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def collate_fn(batch):
    """Tokenize batches efficiently"""
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

    # Tokenize entire batch
    encoding = tokenizer(
        texts,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'label': labels
    }

# Load data
train_df = load_liar_data('liar_dataset/train.tsv')
val_df = load_liar_data('liar_dataset/valid.tsv')
test_df = load_liar_data('liar_dataset/test.tsv')

# Create datasets
train_dataset = LIARDataset(
    train_df['statement'].values,
    train_df['label_encoded'].values
)

val_dataset = LIARDataset(
    val_df['statement'].values,
    val_df['label_encoded'].values
)

test_dataset = LIARDataset(
    test_df['statement'].values,
    test_df['label_encoded'].values
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn
)

# Allows accelerated training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}\n")

# Load pre-trained RoBERTa
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=6  # 6 truthfulness classes
)
model.to(device)

# Training hyperparameters
num_epochs = 3
learning_rate = 2e-5
weight_decay = 0.01

# Calculate total training steps
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)

print(f"Training configuration:")
print(f"  Epochs: {num_epochs}")
print(f"  Learning rate: {learning_rate}")
print(f"  Batch size: 16")
print(f"  Total training steps: {num_training_steps}")
print(f"  Warmup steps: {num_warmup_steps}\n")

# AdamW optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-6,
    weight_decay=weight_decay
)

# Learning rate scheduler with warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# ===================================================
# TRAINING AND EVALUATION
# ===================================================

# Sinlge training loop
def train_epoch(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy

# Evaluate model on val/test set for fine-tuning
def evaluate_model(model, data_loader, device):

    model.eval()  # Set model to evaluation mode
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():  # No gradient calculation
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            # Get probabilities (0-1 scale)
            probabilities = F.softmax(logits, dim=-1)

            # Get predictions
            predictions = torch.argmax(probabilities, dim=-1)

            # Store results
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)

    return avg_loss, accuracy, all_predictions, all_labels, all_probabilities

# Training Loop

print("=" * 60)
print("STARTING TRAINING ON LIAR DATASET")
print("=" * 60)

best_val_accuracy = 0
best_model_state = None

for epoch in range(1, num_epochs + 1):
    print(f"\n{'=' * 60}")
    print(f"EPOCH {epoch}/{num_epochs}")
    print(f"{'=' * 60}")

    # Train for one epoch
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, device, epoch
    )

    print(f"\nTraining Results:")
    print(f"  Loss: {train_loss:.4f}")
    print(f"  Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")

    # Evaluate on validation set
    val_loss, val_acc, val_preds, val_labels, val_probs = evaluate_model(
        model, val_loader, device
    )

    print(f"\nValidation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

    # Save best model
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model_state = model.state_dict().copy()
        torch.save(best_model_state, 'best_roberta_liar_model.pt')
        print(f"\n  ✓ New best model saved! (accuracy: {val_acc:.4f})")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nBest Validation Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
