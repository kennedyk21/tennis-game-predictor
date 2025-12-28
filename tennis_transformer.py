"""
Transformer-based model for tennis match prediction.

This module implements a transformer architecture suitable for tabular data,
using attention mechanisms to learn complex feature interactions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
import logging
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


class TabularTransformer(nn.Module):
    """
    Transformer model for tabular data.
    
    Architecture:
    - Feature embeddings for each input feature
    - Learnable positional encodings
    - Multi-head self-attention layers
    - Feed-forward networks
    - Output head for binary classification
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: Optional[int] = None
    ):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        
        # Set max_len to be at least n_features + some buffer
        if max_len is None:
            max_len = max(n_features + 10, 100)
        
        # Feature embedding: each feature gets embedded to d_model dimensions
        self.feature_embedding = nn.Linear(1, d_model)
        
        # Learnable positional encoding (even though order doesn't matter much in tabular)
        self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Layer norm for final output
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Output probabilities of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # Embed each feature: (batch_size, n_features) -> (batch_size, n_features, d_model)
        # We need to add a dimension for the linear layer
        x = x.unsqueeze(-1)  # (batch_size, n_features, 1)
        x = self.feature_embedding(x)  # (batch_size, n_features, d_model)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:x.shape[1], :].unsqueeze(0)  # (1, n_features, d_model)
        x = x + pos_enc
        
        # Apply transformer
        x = self.transformer(x)  # (batch_size, n_features, d_model)
        
        # Global average pooling over features
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Layer norm
        x = self.layer_norm(x)
        
        # Output head
        output = self.output_head(x)  # (batch_size, 1)
        
        return output


class TennisDataset(Dataset):
    """Dataset class for tennis match data."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, scaler: Optional[StandardScaler] = None):
        """
        Initialize dataset.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            scaler: Optional StandardScaler (if None, will fit one)
        """
        self.X = X.values.astype(np.float32)
        self.y = y.values.astype(np.float32)
        
        if scaler is None:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        else:
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class TransformerTrainer:
    """Wrapper class for training and evaluating the transformer model."""
    
    def __init__(
        self,
        model: TabularTransformer,
        scaler: StandardScaler,
        feature_cols: List[str],
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        
    def train_epoch(self, train_loader: DataLoader, criterion, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader, criterion):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)
                
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        return avg_loss, accuracy


def train_transformer_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 3,
    d_ff: int = 256,
    dropout: float = 0.1,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 10,
    verbose: bool = True
) -> TransformerTrainer:
    """
    Train a transformer model for tennis match prediction.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        patience: Early stopping patience
        verbose: Whether to print training progress
        
    Returns:
        TransformerTrainer object with trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info(f"Using device: {device}")
    
    n_features = X_train.shape[1]
    
    # Create model
    model = TabularTransformer(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout
    )
    
    # Create datasets
    train_dataset = TennisDataset(X_train, y_train)
    scaler = train_dataset.scaler
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TennisDataset(X_val, y_val, scaler=scaler)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create trainer
    trainer = TransformerTrainer(model, scaler, list(X_train.columns), device)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader, criterion, optimizer)
        trainer.history["train_loss"].append(train_loss)
        
        if val_loader is not None:
            val_loss, val_accuracy = trainer.validate(val_loader, criterion)
            trainer.history["val_loss"].append(val_loss)
            trainer.history["val_accuracy"].append(val_accuracy)
            
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.4f}"
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                trainer.best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    # Load best model state
                    if hasattr(trainer, 'best_model_state'):
                        model.load_state_dict(trainer.best_model_state)
                    break
        else:
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
    
    if verbose:
        logger.info("Training complete!")
    
    return trainer


def evaluate_transformer_model(
    trainer: TransformerTrainer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    batch_size: int = 256
) -> Dict[str, float]:
    """
    Comprehensive evaluation of the transformer model on test data.
    
    Computes a wide range of metrics including:
      - Basic metrics: accuracy, precision, recall, F1-score
      - Probability metrics: log_loss, ROC AUC, PR AUC, Brier score
      - Confusion matrix: TP, FP, TN, FN
      - Agreement metrics: Matthews Correlation Coefficient, Cohen's Kappa
    
    Args:
        trainer: Trained TransformerTrainer object
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, log_loss, roc_auc_score, precision_score, recall_score,
        f1_score, confusion_matrix, precision_recall_curve, average_precision_score,
        brier_score_loss, matthews_corrcoef, cohen_kappa_score, roc_curve
    )
    import math
    
    # Create test dataset
    test_dataset = TennisDataset(X_test, y_test, scaler=trainer.scaler)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    trainer.model.eval()
    all_predictions = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(trainer.device)
            outputs = trainer.model(X_batch)
            
            probs = outputs.cpu().numpy().flatten()
            predictions = (outputs > 0.5).float().cpu().numpy().flatten()
            labels = y_batch.numpy().flatten()
            
            all_predictions.extend(predictions)
            all_probs.extend(probs)
            all_labels.extend(labels)
    
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Basic classification metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    # Handle different confusion matrix shapes
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1:
        # Only one class predicted
        if all_predictions[0] == 0:
            tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, int(cm[0, 0])
    else:
        # Fallback for unexpected shapes
        tn, fp, fn, tp = 0, 0, 0, 0
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall  # Same as recall
    ppv = precision  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    
    # Probability-based metrics
    try:
        log_loss_val = log_loss(all_labels, all_probs)
    except ValueError:
        log_loss_val = float('nan')
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
    except ValueError:
        roc_auc = float('nan')
        fpr, tpr = None, None
    
    try:
        pr_auc = average_precision_score(all_labels, all_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    except ValueError:
        pr_auc = float('nan')
        precision_curve, recall_curve = None, None
    
    try:
        brier_score = brier_score_loss(all_labels, all_probs)
    except ValueError:
        brier_score = float('nan')
    
    # Agreement metrics
    try:
        mcc = matthews_corrcoef(all_labels, all_predictions)
    except ValueError:
        mcc = float('nan')
    
    try:
        kappa = cohen_kappa_score(all_labels, all_predictions)
    except ValueError:
        kappa = float('nan')
    
    # Compile all metrics
    metrics = {
        # Basic metrics
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "ppv": float(ppv),  # Positive Predictive Value
        "npv": float(npv),   # Negative Predictive Value
        
        # Confusion matrix
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        
        # Probability metrics
        "log_loss": float(log_loss_val) if not np.isnan(log_loss_val) else None,
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
        "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else None,
        "brier_score": float(brier_score) if not np.isnan(brier_score) else None,
        
        # Agreement metrics
        "matthews_corrcoef": float(mcc) if not np.isnan(mcc) else None,
        "cohen_kappa": float(kappa) if not np.isnan(kappa) else None,
        
        # Additional info
        "n_test_samples": int(len(all_labels)),
        "class_distribution": dict(zip(*np.unique(all_labels, return_counts=True)))
    }
    
    # Log comprehensive metrics
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE TRANSFORMER MODEL EVALUATION METRICS")
    logger.info("=" * 60)
    logger.info(f"\nBasic Classification Metrics:")
    logger.info(f"  Accuracy:        {accuracy:.4f}")
    logger.info(f"  Precision:       {precision:.4f}")
    logger.info(f"  Recall:          {recall:.4f}")
    logger.info(f"  F1-Score:        {f1:.4f}")
    logger.info(f"  Specificity:     {specificity:.4f}")
    logger.info(f"  Sensitivity:     {sensitivity:.4f}")
    logger.info(f"  PPV:             {ppv:.4f}")
    logger.info(f"  NPV:             {npv:.4f}")
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Negatives:  {tn}")
    logger.info(f"  False Positives: {fp}")
    logger.info(f"  False Negatives: {fn}")
    logger.info(f"  True Positives:  {tp}")
    
    logger.info(f"\nProbability Metrics:")
    if not np.isnan(log_loss_val):
        logger.info(f"  Log Loss:        {log_loss_val:.4f}")
    if not np.isnan(roc_auc):
        logger.info(f"  ROC AUC:         {roc_auc:.4f}")
    if not np.isnan(pr_auc):
        logger.info(f"  PR AUC:          {pr_auc:.4f}")
    if not np.isnan(brier_score):
        logger.info(f"  Brier Score:     {brier_score:.4f}")
    
    logger.info(f"\nAgreement Metrics:")
    if not np.isnan(mcc):
        logger.info(f"  Matthews CC:     {mcc:.4f}")
    if not np.isnan(kappa):
        logger.info(f"  Cohen's Kappa:   {kappa:.4f}")
    
    logger.info("=" * 60)
    
    return metrics


def save_transformer_model(
    trainer: TransformerTrainer,
    feature_cols: List[str],
    metrics: Dict[str, float],
    output_dir: str = "./models"
) -> None:
    """
    Save the trained transformer model and metadata.
    
    Args:
        trainer: Trained TransformerTrainer object
        feature_cols: List of feature column names
        metrics: Evaluation metrics dictionary
        output_dir: Output directory for saving
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    model_path = output_path / "tennis_transformer_model.pt"
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'model_config': {
            'n_features': trainer.model.n_features,
            'd_model': trainer.model.d_model,
        },
        'scaler': trainer.scaler,
        'feature_cols': feature_cols
    }, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save metadata
    metadata = {
        "model_type": "transformer",
        "feature_cols": feature_cols,
        "metrics": metrics,
        "training_history": trainer.history
    }
    
    metadata_path = output_path / "tennis_transformer_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata to {metadata_path}")


def load_transformer_model(
    model_path: str,
    device: Optional[torch.device] = None
) -> TransformerTrainer:
    """
    Load a saved transformer model.
    
    Args:
        model_path: Path to saved model file
        device: Device to load model on
        
    Returns:
        TransformerTrainer object with loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    # Recreate model
    model = TabularTransformer(
        n_features=config['n_features'],
        d_model=config['d_model']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    scaler = checkpoint['scaler']
    feature_cols = checkpoint['feature_cols']
    
    trainer = TransformerTrainer(model, scaler, feature_cols, device)
    
    return trainer
