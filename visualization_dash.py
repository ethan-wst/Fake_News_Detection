"""
Visualization Dashboard for Fake News Detection Model
Creates comprehensive visualizations and analysis reports
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# ===================================================
# MODEL ARCHITECTURE (must match training.py)
# ===================================================

class RobertaOrdinalClassifier(nn.Module):
    """Custom RoBERTa with ordinal classification"""
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# ===================================================
# DATA LOADING AND INFERENCE
# ===================================================

def load_liar_data(file_path):
    """Load LIAR dataset"""
    df = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        usecols=[1, 2, 3, 13],
        names=['label', 'statement', 'subject', 'context']
    )
    
    for col in ['statement', 'subject', 'context']:
        df[col] = df[col].fillna('').astype(str).str.strip()
    
    label_map = {
        'pants-fire': 0, 'false': 1, 'barely-true': 2,
        'half-true': 3, 'mostly-true': 4, 'true': 5
    }
    df['label_encoded'] = df['label'].map(label_map)
    
    return df


def compute_truthfulness_score(class_probs):
    """Convert class probabilities to truthfulness score (0.0-1.0)"""
    class_values = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=class_probs.device)
    return torch.sum(class_probs * class_values, dim=1)


def get_predictions(model, df, tokenizer, device, batch_size=32):
    """Run inference on dataset and return predictions"""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_truthfulness = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Running inference"):
            batch = df.iloc[i:i+batch_size]
            
            # Prepare inputs
            statements = batch['statement'].tolist()
            subjects = batch['subject'].tolist()
            contexts = batch['context'].tolist()
            
            structured_inputs = []
            for stmt, subj, ctx in zip(statements, subjects, contexts):
                if subj and ctx:
                    structured_inputs.append(f"{stmt} {tokenizer.sep_token} {subj} {tokenizer.sep_token} {ctx}")
                elif subj:
                    structured_inputs.append(f"{stmt} {tokenizer.sep_token} {subj}")
                else:
                    structured_inputs.append(stmt)
            
            # Tokenize
            encoding = tokenizer(
                structured_inputs,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Get predictions
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, dim=1)
            
            # Compute truthfulness scores
            truthfulness = compute_truthfulness_score(probs)
            
            # Store results
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_truthfulness.extend(truthfulness.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_truthfulness)


# ===================================================
# VISUALIZATION FUNCTIONS
# ===================================================

def plot_confusion_matrix(y_true, y_pred, labels, save_path='confusion_matrix.png'):
    """Plot confusion matrix for 6-way classification"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - 6-Way Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_gate_confusion_matrix(y_true, truthfulness_scores, thresholds, save_path='gate_confusion_matrix.png'):
    """Plot confusion matrix for 3-way gate system (KEEP/REVIEW/REJECT)"""
    keep_thresh, reject_thresh = thresholds
    
    # Convert truthfulness scores to gate decisions
    gate_decisions = np.where(truthfulness_scores >= keep_thresh, 2,  # KEEP
                     np.where(truthfulness_scores <= reject_thresh, 0,  # REJECT
                             1))  # REVIEW
    
    # Convert true labels to gate categories (simplification)
    # 0-1 (pants-fire, false) -> should REJECT
    # 2-3 (barely-true, half-true) -> should REVIEW  
    # 4-5 (mostly-true, true) -> should KEEP
    true_gate = np.where(y_true >= 4, 2,  # KEEP
                np.where(y_true <= 1, 0,  # REJECT
                        1))  # REVIEW
    
    cm = confusion_matrix(true_gate, gate_decisions, labels=[0, 1, 2])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=['REJECT', 'REVIEW', 'KEEP'],
                yticklabels=['REJECT', 'REVIEW', 'KEEP'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Ingress Gate Confusion Matrix\n(Keep≥{keep_thresh}, Reject≤{reject_thresh})', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_score_distributions(y_true, truthfulness_scores, labels, save_path='score_distributions.png'):
    """Plot truthfulness score distributions by true label"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#d62728', '#ff7f0e', '#ffbb78', '#98df8a', '#2ca02c', '#1f77b4']
    
    for i, label in enumerate(labels):
        scores = truthfulness_scores[y_true == i]
        
        axes[i].hist(scores, bins=30, alpha=0.7, color=colors[i], edgecolor='black')
        axes[i].axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.3f}')
        axes[i].axvline(np.median(scores), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.3f}')
        axes[i].set_title(f'{label.upper()}\n(n={len(scores)})', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Truthfulness Score', fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.suptitle('Truthfulness Score Distributions by True Label', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_score_violin(y_true, truthfulness_scores, labels, save_path='score_violin.png'):
    """Violin plot of truthfulness scores by true label"""
    # Prepare data
    df_plot = pd.DataFrame({
        'True Label': [labels[i] for i in y_true],
        'Truthfulness Score': truthfulness_scores
    })
    
    plt.figure(figsize=(14, 6))
    sns.violinplot(data=df_plot, x='True Label', y='Truthfulness Score', 
                   palette='RdYlGn', inner='box')
    plt.axhline(y=0.75, color='green', linestyle='--', label='KEEP threshold (0.75)', linewidth=2)
    plt.axhline(y=0.35, color='red', linestyle='--', label='REJECT threshold (0.35)', linewidth=2)
    plt.title('Truthfulness Score Distribution by True Label', fontsize=14, fontweight='bold')
    plt.xlabel('True Label', fontsize=12)
    plt.ylabel('Truthfulness Score', fontsize=12)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_calibration_curve(y_true, class_probs, save_path='calibration_curve.png'):
    """Plot calibration curve for each class"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    labels = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
    
    for i, label in enumerate(labels):
        # Get predicted probabilities for this class
        pred_probs = class_probs[:, i]
        true_labels = (y_true == i).astype(int)
        
        # Bin predictions
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(pred_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        
        # Calculate actual frequency in each bin
        bin_counts = np.bincount(bin_indices, minlength=len(bin_centers))
        bin_true = np.bincount(bin_indices, weights=true_labels, minlength=len(bin_centers))
        
        # Avoid division by zero
        bin_true_freq = np.divide(bin_true, bin_counts, where=bin_counts>0, out=np.zeros_like(bin_true))
        
        # Plot
        axes[i].plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
        axes[i].plot(bin_centers, bin_true_freq, 'o-', label=label, markersize=8, linewidth=2)
        axes[i].set_title(f'{label.upper()}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted Probability', fontsize=10)
        axes[i].set_ylabel('True Frequency', fontsize=10)
        axes[i].legend()
        axes[i].grid(alpha=0.3)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])
    
    plt.suptitle('Calibration Curves by Class', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_threshold_analysis(y_true, truthfulness_scores, save_path='threshold_analysis.png'):
    """Analyze different threshold configurations"""
    # Test different thresholds
    keep_thresholds = np.arange(0.6, 0.95, 0.05)
    reject_thresholds = np.arange(0.2, 0.5, 0.05)
    
    results = []
    
    for keep_thresh in keep_thresholds:
        for reject_thresh in reject_thresholds:
            if keep_thresh <= reject_thresh:
                continue
            
            # Make gate decisions
            decisions = np.where(truthfulness_scores >= keep_thresh, 'KEEP',
                        np.where(truthfulness_scores <= reject_thresh, 'REJECT', 'REVIEW'))
            
            # Calculate metrics
            keep_pct = (decisions == 'KEEP').mean() * 100
            reject_pct = (decisions == 'REJECT').mean() * 100
            review_pct = (decisions == 'REVIEW').mean() * 100
            
            # Calculate accuracy for auto-decisions (simplified)
            auto_mask = (decisions != 'REVIEW')
            if auto_mask.sum() > 0:
                # Simple heuristic: true/mostly-true should be kept, pants-fire/false should be rejected
                correct_keep = ((decisions == 'KEEP') & (y_true >= 4)).sum()
                correct_reject = ((decisions == 'REJECT') & (y_true <= 1)).sum()
                auto_accuracy = (correct_keep + correct_reject) / auto_mask.sum() * 100
            else:
                auto_accuracy = 0
            
            results.append({
                'keep_threshold': keep_thresh,
                'reject_threshold': reject_thresh,
                'keep_pct': keep_pct,
                'reject_pct': reject_pct,
                'review_pct': review_pct,
                'automation_rate': keep_pct + reject_pct,
                'auto_accuracy': auto_accuracy
            })
    
    df_results = pd.DataFrame(results)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Automation rate vs review rate
    scatter = axes[0, 0].scatter(df_results['review_pct'], df_results['automation_rate'],
                                c=df_results['auto_accuracy'], cmap='RdYlGn', s=100, alpha=0.6)
    axes[0, 0].set_xlabel('Review Rate (%)', fontsize=12)
    axes[0, 0].set_ylabel('Automation Rate (%)', fontsize=12)
    axes[0, 0].set_title('Automation vs Review Trade-off', fontsize=13, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='Auto Accuracy (%)')
    
    # Plot 2: Distribution of decisions
    best_idx = df_results['auto_accuracy'].idxmax()
    best_config = df_results.iloc[best_idx]
    
    axes[0, 1].bar(['KEEP', 'REVIEW', 'REJECT'], 
                  [best_config['keep_pct'], best_config['review_pct'], best_config['reject_pct']],
                  color=['green', 'orange', 'red'], alpha=0.7)
    axes[0, 1].set_ylabel('Percentage (%)', fontsize=12)
    axes[0, 1].set_title(f'Best Config: Keep≥{best_config["keep_threshold"]:.2f}, Reject≤{best_config["reject_threshold"]:.2f}\n' +
                        f'Auto Accuracy: {best_config["auto_accuracy"]:.1f}%', 
                        fontsize=13, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Plot 3: Heatmap of automation rate
    pivot_auto = df_results.pivot(index='reject_threshold', columns='keep_threshold', values='automation_rate')
    sns.heatmap(pivot_auto, cmap='YlGnBu', annot=False, fmt='.1f', ax=axes[1, 0], cbar_kws={'label': 'Automation Rate (%)'})
    axes[1, 0].set_title('Automation Rate Heatmap', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Keep Threshold', fontsize=12)
    axes[1, 0].set_ylabel('Reject Threshold', fontsize=12)
    
    # Plot 4: Heatmap of auto accuracy
    pivot_acc = df_results.pivot(index='reject_threshold', columns='keep_threshold', values='auto_accuracy')
    sns.heatmap(pivot_acc, cmap='RdYlGn', annot=False, fmt='.1f', ax=axes[1, 1], cbar_kws={'label': 'Auto Accuracy (%)'})
    axes[1, 1].set_title('Auto-Decision Accuracy Heatmap', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Keep Threshold', fontsize=12)
    axes[1, 1].set_ylabel('Reject Threshold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")
    
    return df_results


def plot_error_analysis(y_true, y_pred, truthfulness_scores, labels, save_path='error_analysis.png'):
    """Analyze model errors"""
    errors = (y_true != y_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Error rate by true label
    error_rates = []
    for i in range(len(labels)):
        mask = (y_true == i)
        if mask.sum() > 0:
            error_rate = errors[mask].mean() * 100
        else:
            error_rate = 0
        error_rates.append(error_rate)
    
    axes[0, 0].bar(labels, error_rates, color='salmon', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('True Label', fontsize=12)
    axes[0, 0].set_ylabel('Error Rate (%)', fontsize=12)
    axes[0, 0].set_title('Error Rate by True Label', fontsize=13, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    # Plot 2: Off-by-N accuracy
    error_distances = np.abs(y_true - y_pred)
    off_by_exact = (error_distances == 0).mean() * 100
    off_by_1 = (error_distances <= 1).mean() * 100
    off_by_2 = (error_distances <= 2).mean() * 100
    off_by_3 = (error_distances <= 3).mean() * 100
    
    axes[0, 1].bar(['Exact', '±1 class', '±2 classes', '±3 classes'],
                  [off_by_exact, off_by_1, off_by_2, off_by_3],
                  color=['#2ca02c', '#98df8a', '#ffbb78', '#ff7f0e'], alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Ordinal Accuracy (±N Classes)', fontsize=13, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 105])
    
    # Add percentage labels on bars
    for i, v in enumerate([off_by_exact, off_by_1, off_by_2, off_by_3]):
        axes[0, 1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Plot 3: Confidence vs Error
    confidences = truthfulness_scores
    axes[1, 0].scatter(confidences[~errors], np.zeros(sum(~errors)), alpha=0.3, label='Correct', s=20)
    axes[1, 0].scatter(confidences[errors], np.ones(sum(errors)), alpha=0.3, label='Error', s=20, color='red')
    axes[1, 0].set_xlabel('Truthfulness Score', fontsize=12)
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(['Correct', 'Error'])
    axes[1, 0].set_title('Prediction Correctness vs Truthfulness Score', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Error distance distribution
    axes[1, 1].hist(error_distances, bins=range(7), alpha=0.7, color='salmon', edgecolor='black', rwidth=0.8)
    axes[1, 1].set_xlabel('Prediction Error Distance', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Distribution of Prediction Errors', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(range(6))
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_subject_analysis(df, y_true, y_pred, truthfulness_scores, save_path='subject_analysis.png'):
    """Analyze performance by subject/topic"""
    # Get top subjects
    top_subjects = df['subject'].value_counts().head(10).index.tolist()
    
    subject_metrics = []
    for subject in top_subjects:
        mask = df['subject'] == subject
        if mask.sum() < 10:  # Skip subjects with too few samples
            continue
        
        subject_metrics.append({
            'subject': subject[:30],  # Truncate long subjects
            'count': mask.sum(),
            'accuracy': (y_true[mask] == y_pred[mask]).mean() * 100,
            'mean_score': truthfulness_scores[mask].mean(),
            'std_score': truthfulness_scores[mask].std()
        })
    
    df_subjects = pd.DataFrame(subject_metrics).sort_values('accuracy', ascending=False)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Accuracy by subject
    axes[0].barh(df_subjects['subject'], df_subjects['accuracy'], color='skyblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Model Accuracy by Subject/Topic (Top 10)', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='x')
    
    # Plot 2: Mean truthfulness score by subject
    axes[1].barh(df_subjects['subject'], df_subjects['mean_score'], 
                xerr=df_subjects['std_score'], color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Mean Truthfulness Score', fontsize=12)
    axes[1].set_title('Mean Truthfulness Score by Subject/Topic', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def create_html_report(df, y_true, y_pred, truthfulness_scores, class_probs, labels):
    """Create interactive HTML dashboard using Plotly"""
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Truthfulness Score Distribution by True Label',
                       'Predicted vs True Label Scatter',
                       'Gate Decision Distribution',
                       'Class Probability Heatmap (Sample)',
                       'Error Distance Distribution',
                       'Automation Rate vs Accuracy Trade-off'),
        specs=[[{"type": "box"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "heatmap"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # Plot 1: Box plot of scores by true label
    colors = ['#d62728', '#ff7f0e', '#ffbb78', '#98df8a', '#2ca02c', '#1f77b4']
    for i, label in enumerate(labels):
        scores = truthfulness_scores[y_true == i]
        fig.add_trace(
            go.Box(y=scores, name=label, marker_color=colors[i]),
            row=1, col=1
        )
    
    # Plot 2: Scatter of predicted vs true
    fig.add_trace(
        go.Scatter(x=y_true, y=y_pred, mode='markers',
                  marker=dict(color=truthfulness_scores, colorscale='RdYlGn',
                             showscale=True, colorbar=dict(title="Truth Score")),
                  text=[f"Score: {s:.3f}" for s in truthfulness_scores],
                  hovertemplate='<b>True:</b> %{x}<br><b>Pred:</b> %{y}<br>%{text}'),
        row=1, col=2
    )
    
    # Plot 3: Gate decision distribution
    keep_thresh, reject_thresh = 0.75, 0.35
    gate_decisions = np.where(truthfulness_scores >= keep_thresh, 'KEEP',
                     np.where(truthfulness_scores <= reject_thresh, 'REJECT', 'REVIEW'))
    decision_counts = pd.Series(gate_decisions).value_counts()
    
    fig.add_trace(
        go.Bar(x=decision_counts.index, y=decision_counts.values,
              marker_color=['red', 'orange', 'green']),
        row=2, col=1
    )
    
    # Plot 4: Sample probability heatmap (first 50 samples)
    sample_probs = class_probs[:50]
    fig.add_trace(
        go.Heatmap(z=sample_probs.T, x=list(range(50)), y=labels,
                  colorscale='RdYlGn', showscale=False),
        row=2, col=2
    )
    
    # Plot 5: Error distance distribution
    error_distances = np.abs(y_true - y_pred)
    fig.add_trace(
        go.Histogram(x=error_distances, marker_color='salmon', nbinsx=6),
        row=3, col=1
    )
    
    # Plot 6: Threshold analysis scatter
    # Simulate different threshold configs
    automation_rates = []
    accuracies = []
    for keep_t in np.arange(0.6, 0.9, 0.05):
        decisions = np.where(truthfulness_scores >= keep_t, 'KEEP', 'OTHER')
        auto_rate = (decisions == 'KEEP').mean() * 100
        correct_keep = ((decisions == 'KEEP') & (y_true >= 4)).sum()
        acc = correct_keep / max(1, (decisions == 'KEEP').sum()) * 100
        automation_rates.append(auto_rate)
        accuracies.append(acc)
    
    fig.add_trace(
        go.Scatter(x=automation_rates, y=accuracies, mode='markers+lines',
                  marker=dict(size=10, color='blue')),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Fake News Detection Model - Interactive Dashboard",
        title_font_size=20,
        showlegend=False,
        height=1200,
        template='plotly_white'
    )
    
    # Save HTML
    fig.write_html('dashboard.html')
    print(f"✓ Saved: dashboard.html")


# ===================================================
# MAIN DASHBOARD GENERATION
# ===================================================

def generate_dashboard(model_path='liar_model.pt', test_data_path='liar_dataset/test.tsv'):
    """Generate complete visualization dashboard"""
    
    print("="*60)
    print("FAKE NEWS DETECTION - VISUALIZATION DASHBOARD")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading model...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaOrdinalClassifier(num_classes=6, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("✓ Model loaded")
    
    # Load test data
    print("\nLoading test data...")
    df = load_liar_data(test_data_path)
    y_true = df['label_encoded'].values
    labels = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
    print(f"✓ Loaded {len(df)} test samples")
    
    # Get predictions
    print("\nRunning inference on test set...")
    y_pred, class_probs, truthfulness_scores = get_predictions(model, df, tokenizer, device)
    print(f"✓ Inference complete")
    
    # Calculate metrics
    print("\nOverall Metrics:")
    accuracy = (y_true == y_pred).mean()
    mae = np.abs(y_true - y_pred).mean()
    acc_off1 = (np.abs(y_true - y_pred) <= 1).mean()
    acc_off2 = (np.abs(y_true - y_pred) <= 2).mean()
    
    print(f"  Exact Accuracy: {accuracy*100:.2f}%")
    print(f"  MAE: {mae:.3f}")
    print(f"  Accuracy ±1: {acc_off1*100:.2f}%")
    print(f"  Accuracy ±2: {acc_off2*100:.2f}%")
    print(f"  Mean Truthfulness Score: {truthfulness_scores.mean():.3f}")
    
    # Generate all visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    plot_confusion_matrix(y_true, y_pred, labels)
    plot_gate_confusion_matrix(y_true, truthfulness_scores, (0.75, 0.35))
    plot_score_distributions(y_true, truthfulness_scores, labels)
    plot_score_violin(y_true, truthfulness_scores, labels)
    plot_calibration_curve(y_true, class_probs)
    threshold_df = plot_threshold_analysis(y_true, truthfulness_scores)
    plot_error_analysis(y_true, y_pred, truthfulness_scores, labels)
    plot_subject_analysis(df, y_true, y_pred, truthfulness_scores)
    create_html_report(df, y_true, y_pred, truthfulness_scores, class_probs, labels)
    
    # Save numerical results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")
    
    results_df = pd.DataFrame({
        'statement': df['statement'].values,
        'subject': df['subject'].values,
        'true_label': [labels[i] for i in y_true],
        'predicted_label': [labels[i] for i in y_pred],
        'truthfulness_score': truthfulness_scores,
        'correct': y_true == y_pred,
        'error_distance': np.abs(y_true - y_pred)
    })
    
    results_df.to_csv('detailed_predictions.csv', index=False)
    print("✓ Saved: detailed_predictions.csv")
    
    threshold_df.to_csv('threshold_analysis.csv', index=False)
    print("✓ Saved: threshold_analysis.csv")
    
    # Print summary
    print("\n" + "="*60)
    print("DASHBOARD GENERATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  📊 confusion_matrix.png")
    print("  📊 gate_confusion_matrix.png")
    print("  📊 score_distributions.png")
    print("  📊 score_violin.png")
    print("  📊 calibration_curve.png")
    print("  📊 threshold_analysis.png")
    print("  📊 error_analysis.png")
    print("  📊 subject_analysis.png")
    print("  🌐 dashboard.html (Interactive)")
    print("  📄 detailed_predictions.csv")
    print("  📄 threshold_analysis.csv")
    print("\nOpen 'dashboard.html' in your browser for interactive visualizations!")


if __name__ == '__main__':
    generate_dashboard()