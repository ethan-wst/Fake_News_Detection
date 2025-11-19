import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics imfport accuracy_score, classification_report, f1_score

# ===================================================
# OPTION A: MULTI-CLASS CLASSIFICATION (RECOMMENDED)
# ===================================================

class LIARDataset(Dataset):
    def __init__(self, texts, labels, subjects, contexts):
        self.texts = texts
        self.labels = labels
        self.subjects = subjects
        self.contexts = contexts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': str(self.texts[idx]),
            'subject': str(self.subjects[idx]),
            'context': str(self.contexts[idx]),
            'label': self.labels[idx]
        }

def load_liar_data(file_path):
    """Load LIAR dataset with proper label encoding"""
    df = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        usecols=[1, 2, 3, 13],
        names=['label', 'statement', 'subject', 'context']
    )

    # Clean text columns
    for col in ['statement', 'subject', 'context']:
        df[col] = df[col].fillna('').astype(str).str.strip()

    # Ordinal label encoding (maintains order but treats as classes)
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

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def collate_fn(batch):
    """Tokenize with structured metadata using special tokens"""
    texts = [item['text'] for item in batch]
    subjects = [item['subject'] for item in batch]
    contexts = [item['context'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

    # Structure input with clear delineation
    # Format: [CLS] statement [SEP] subject [SEP] context [SEP]
    structured_inputs = []
    for text, subj, ctx in zip(texts, subjects, contexts):
        # Add subject and context only if they're non-empty
        if subj and ctx:
            structured_inputs.append(f"{text} {tokenizer.sep_token} {subj} {tokenizer.sep_token} {ctx}")
        elif subj:
            structured_inputs.append(f"{text} {tokenizer.sep_token} {subj}")
        else:
            structured_inputs.append(text)

    encoding = tokenizer(
        structured_inputs,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'label': labels
    }

# ===================================================
# CUSTOM MODEL WITH ORDINAL REGRESSION
# ===================================================

class RobertaOrdinalClassifier(nn.Module):
    """
    Custom RoBERTa with ordinal regression approach.
    Uses cumulative link model for ordered categories.
    """
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        # Output layer for ordinal regression
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ===================================================
# TRAINING FUNCTIONS
# ===================================================

def compute_truthfulness_score(class_probs):
    """
    Convert multi-class probabilities to continuous truthfulness score (0.0 to 1.0)
    
    Three approaches available:
    1. Expected Value: Weighted average of class positions
    2. Cumulative Probability: Sum probabilities of "truthful" classes
    3. Sigmoid Transform: Smooth continuous mapping
    
    Args:
        class_probs: Tensor of shape (batch_size, num_classes) with softmax probabilities
    
    Returns:
        Tensor of shape (batch_size,) with truthfulness scores from 0.0 to 1.0
    """
    
    # APPROACH 1: Expected Value (Recommended)
    # Calculate weighted average of class positions normalized to [0, 1]
    class_values = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=class_probs.device)
    truthfulness = torch.sum(class_probs * class_values, dim=1)
    
    return truthfulness

def compute_truthfulness_score_cumulative(class_probs):
    """
    Alternative: Cumulative probability approach
    Score = P(mostly-true) + P(true) + 0.5 * P(half-true)
    """
    # Weight classes by truthfulness contribution
    weights = torch.tensor([0.0, 0.0, 0.1, 0.3, 0.8, 1.0], device=class_probs.device)
    truthfulness = torch.sum(class_probs * weights, dim=1)
    return truthfulness

def compute_truthfulness_score_sigmoid(class_probs):
    """
    Alternative: Sigmoid-transformed expected class
    Smooth transformation centered at "half-true" (class 3)
    """
    # Calculate expected class (0-5)
    class_positions = torch.arange(6, dtype=torch.float32, device=class_probs.device)
    expected_class = torch.sum(class_probs * class_positions, dim=1)
    
    # Transform to [0, 1] with sigmoid centered at class 3 (half-true)
    truthfulness = torch.sigmoid((expected_class - 2.5) / 1.5)
    return truthfulness

def compute_class_weights(train_labels):
    """Compute class weights for imbalanced dataset"""
    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    weights = total / (len(unique) * counts)
    return torch.FloatTensor(weights)

def train_epoch(model, data_loader, optimizer, scheduler, device, epoch, class_weights=None):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    # Cross entropy loss with class weights
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_predictions/total_samples:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy

def evaluate_model(model, data_loader, device):
    """Comprehensive evaluation with ordinal metrics and truthfulness scores"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_truthfulness_scores = []
    
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            # Get class probabilities
            probs = F.softmax(logits, dim=1)
            
            # Convert to truthfulness score (0.0 to 1.0)
            truthfulness_scores = compute_truthfulness_score(probs)

            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_truthfulness_scores.extend(truthfulness_scores.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    
    # Convert to numpy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_truthfulness_scores = np.array(all_truthfulness_scores)
    
    # Standard accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Ordinal metrics: accuracy within k classes
    mae = np.mean(np.abs(all_predictions - all_labels))
    acc_off_by_1 = np.mean(np.abs(all_predictions - all_labels) <= 1)
    acc_off_by_2 = np.mean(np.abs(all_predictions - all_labels) <= 2)
    
    # Macro F1 score
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'mae': mae,
        'acc_±1': acc_off_by_1,
        'acc_±2': acc_off_by_2,
        'f1_macro': f1
    }

    return metrics, all_predictions, all_labels, all_truthfulness_scores

# ===================================================
# MAIN TRAINING LOOP
# ===================================================

def main():
    # Load data
    print("Loading LIAR dataset...")
    train_df = load_liar_data('liar_dataset/train.tsv')
    val_df = load_liar_data('liar_dataset/valid.tsv')
    test_df = load_liar_data('liar_dataset/test.tsv')

    # Create datasets
    train_dataset = LIARDataset(
        train_df['statement'].values,
        train_df['label_encoded'].values,
        train_df['subject'].values,
        train_df['context'].values
    )

    val_dataset = LIARDataset(
        val_df['statement'].values,
        val_df['label_encoded'].values,
        val_df['subject'].values,
        val_df['context'].values
    )

    test_dataset = LIARDataset(
        test_df['statement'].values,
        test_df['label_encoded'].values,
        test_df['subject'].values,
        test_df['context'].values
    )

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weights(train_df['label_encoded'].values)
    print(f"\nClass weights: {class_weights}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")

    # Initialize model (using custom ordinal classifier)
    model = RobertaOrdinalClassifier(num_classes=6, dropout=0.3)
    model.to(device)

    # Training hyperparameters
    num_epochs = 10
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1

    # Calculate training steps
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(warmup_ratio * num_training_steps)

    print(f"Training configuration:")
    print(f"  Model: RoBERTa with ordinal classification")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: 16")
    print(f"  Total steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Weight decay: {weight_decay}\n")

    # Optimizer with layer-wise learning rate decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8
    )

    # Scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training loop
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    best_val_f1 = 0
    best_model_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'=' * 60}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, class_weights
        )

        print(f"\nTraining Results:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")

        # Validate
        val_metrics, val_preds, val_labels, val_truthfulness = evaluate_model(model, val_loader, device)

        print(f"\nValidation Results:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Exact Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
        print(f"  MAE: {val_metrics['mae']:.4f}")
        print(f"  Accuracy ±1 class: {val_metrics['acc_±1']:.4f} ({val_metrics['acc_±1']*100:.2f}%)")
        print(f"  Accuracy ±2 classes: {val_metrics['acc_±2']:.4f} ({val_metrics['acc_±2']*100:.2f}%)")
        print(f"  Macro F1: {val_metrics['f1_macro']:.4f}")
        print(f"  Truthfulness Score Range: [{val_truthfulness.min():.3f}, {val_truthfulness.max():.3f}]")
        print(f"  Mean Truthfulness: {val_truthfulness.mean():.3f}")

        # Save best model (using F1 score)
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, 'liar_model.pt')
            print(f"\n   New best model saved! (F1: {best_val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"\n  No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\n  Early stopping triggered after {epoch} epochs")
            break

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    model.load_state_dict(best_model_state)
    test_metrics, test_preds, test_labels, test_truthfulness = evaluate_model(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  Exact Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  Accuracy ±1 class: {test_metrics['acc_±1']:.4f} ({test_metrics['acc_±1']*100:.2f}%)")
    print(f"  Accuracy ±2 classes: {test_metrics['acc_±2']:.4f} ({test_metrics['acc_±2']*100:.2f}%)")
    print(f"  Macro F1: {test_metrics['f1_macro']:.4f}")
    
    print(f"\nTruthfulness Score Statistics:")
    print(f"  Range: [{test_truthfulness.min():.3f}, {test_truthfulness.max():.3f}]")
    print(f"  Mean: {test_truthfulness.mean():.3f} ± {test_truthfulness.std():.3f}")
    print(f"  Median: {np.median(test_truthfulness):.3f}")
    
    # Show distribution of scores by true label
    print(f"\nTruthfulness Score by True Label:")
    for label in range(6):
        label_scores = test_truthfulness[test_labels == label]
        if len(label_scores) > 0:
            label_name = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true'][label]
            print(f"  {label_name:15s}: {label_scores.mean():.3f} ± {label_scores.std():.3f}")
    
    # Detailed classification report
    label_names = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=label_names, digits=4))

if __name__ == '__main__':
    main()