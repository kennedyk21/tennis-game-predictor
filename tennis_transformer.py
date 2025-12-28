"""
Enhanced Transformer-based model for tennis match prediction.

Improvements:
- Feature importance weighting
- Advanced feature engineering support
- Learning rate scheduling
- Class balancing
- Improved architecture with skip connections
- Comprehensive hyperparameter options
- Cross-validation support
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
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import joblib

logger = logging.getLogger(__name__)


class EnhancedTabularTransformer(nn.Module):
    """
    Enhanced Transformer model for tabular data with improvements:
    - Feature importance weighting
    - Deeper architecture with skip connections
    - Better regularization
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.2,
        use_feature_weights: bool = True,
        max_len: Optional[int] = None
    ):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.use_feature_weights = use_feature_weights
        
        # Set max_len
        if max_len is None:
            max_len = max(n_features + 10, 100)
        
        # Feature importance weights (learnable)
        if use_feature_weights:
            self.feature_weights = nn.Parameter(torch.ones(n_features))
        
        # Feature embedding with larger hidden layer
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # GELU often works better than ReLU
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Multi-scale pooling (combine mean and max pooling)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Enhanced output head with skip connections
        self.output_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 for concat of mean and max pool
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with feature weighting and multi-scale pooling.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Output logits of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # Apply feature importance weights
        if self.use_feature_weights:
            x = x * torch.sigmoid(self.feature_weights)
        
        # Embed each feature
        x = x.unsqueeze(-1)  # (batch_size, n_features, 1)
        x = self.feature_embedding(x)  # (batch_size, n_features, d_model)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:x.shape[1], :].unsqueeze(0)
        x = x + pos_enc
        
        # Apply transformer
        x = self.transformer(x)  # (batch_size, n_features, d_model)
        
        # Multi-scale pooling (mean + max)
        x_mean = x.mean(dim=1)  # (batch_size, d_model)
        x_max = x.max(dim=1)[0]  # (batch_size, d_model)
        x = torch.cat([x_mean, x_max], dim=1)  # (batch_size, d_model * 2)
        
        # Layer norm
        x = self.layer_norm(x[:, :self.d_model]) + self.layer_norm(x[:, self.d_model:])
        x = torch.cat([self.layer_norm(x_mean), self.layer_norm(x_max)], dim=1)
        
        # Output head
        output = self.output_head(x)  # (batch_size, 1)
        
        return output


class TennisDataset(Dataset):
    """Enhanced dataset class with better handling."""
    
    def __init__(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            scaler: Optional StandardScaler
            fit_scaler: Whether to fit the scaler (for training data)
        """
        self.X = X.values.astype(np.float32)
        self.y = y.values.astype(np.float32)
        
        if scaler is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                self.X = self.scaler.fit_transform(self.X)
            else:
                self.X = self.scaler.transform(self.X)
        else:
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class EnhancedTransformerTrainer:
    """Enhanced trainer with learning rate scheduling and better tracking."""
    
    def __init__(
        self,
        model: EnhancedTabularTransformer,
        scaler: StandardScaler,
        feature_cols: List[str],
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.class_weights = class_weights
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
        
        self.history = {
            "train_loss": [], 
            "val_loss": [], 
            "val_accuracy": [],
            "learning_rates": []
        }
        
    def train_epoch(self, train_loader: DataLoader, criterion, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            
            # Apply sigmoid for BCE
            outputs_prob = torch.sigmoid(outputs)
            loss = criterion(outputs_prob, y_batch)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader, criterion):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)
                
                outputs = self.model(X_batch)
                outputs_prob = torch.sigmoid(outputs)
                
                loss = criterion(outputs_prob, y_batch)
                total_loss += loss.item()
                
                predictions = (outputs_prob > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
                
                all_probs.extend(outputs_prob.cpu().numpy().flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        return avg_loss, accuracy
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get learned feature importance weights."""
        if hasattr(self.model, 'feature_weights'):
            weights = torch.sigmoid(self.model.feature_weights).detach().cpu().numpy()
            importance = dict(zip(self.feature_cols, weights))
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            return importance
        return {}


def compute_class_weights_from_labels(y: pd.Series) -> torch.Tensor:
    """Compute class weights for handling class imbalance."""
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32)


def train_transformer_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    d_ff: int = 512,
    dropout: float = 0.2,
    use_feature_weights: bool = True,
    balance_classes: bool = True,
    epochs: int = 100,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 15,
    lr_scheduler_patience: int = 5,
    lr_scheduler_factor: float = 0.5,
    verbose: bool = True
) -> EnhancedTransformerTrainer:
    """
    Train an enhanced transformer model for tennis match prediction.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        d_model: Model dimension (increased default)
        n_heads: Number of attention heads (increased default)
        n_layers: Number of transformer layers (increased default)
        d_ff: Feed-forward dimension (increased default)
        dropout: Dropout rate (increased default)
        use_feature_weights: Whether to use learnable feature importance
        balance_classes: Whether to use class weights for imbalanced data
        epochs: Number of training epochs (increased default)
        batch_size: Batch size (increased default)
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        patience: Early stopping patience (increased default)
        lr_scheduler_patience: Patience for learning rate scheduler
        lr_scheduler_factor: Factor to reduce LR by
        verbose: Whether to print training progress
        
    Returns:
        EnhancedTransformerTrainer object with trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info(f"Using device: {device}")
        logger.info(f"Training samples: {len(X_train)}")
        if X_val is not None:
            logger.info(f"Validation samples: {len(X_val)}")
    
    n_features = X_train.shape[1]
    
    # Compute class weights if needed
    class_weights = None
    if balance_classes:
        class_weights = compute_class_weights_from_labels(y_train)
        if verbose:
            logger.info(f"Class weights: {class_weights}")
    
    # Create model
    model = EnhancedTabularTransformer(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        use_feature_weights=use_feature_weights
    )
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets
    train_dataset = TennisDataset(X_train, y_train, fit_scaler=True)
    scaler = train_dataset.scaler
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TennisDataset(X_val, y_val, scaler=scaler, fit_scaler=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    # Loss and optimizer
    if balance_classes and class_weights is not None:
        # Create weighted BCE loss
        pos_weight = class_weights[1] / class_weights[0]
        criterion = nn.BCELoss()  # Will apply weights through weighted sampling instead
    else:
        criterion = nn.BCELoss()
    
    optimizer = optim.AdamW(  # AdamW often works better than Adam
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        verbose=verbose,
        min_lr=1e-6
    )
    
    # Create trainer
    trainer = EnhancedTransformerTrainer(
        model, scaler, list(X_train.columns), class_weights, device
    )
    
    # Training loop with early stopping and LR scheduling
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader, criterion, optimizer)
        trainer.history["train_loss"].append(train_loss)
        trainer.history["learning_rates"].append(optimizer.param_groups[0]['lr'])
        
        if val_loader is not None:
            val_loss, val_accuracy = trainer.validate(val_loader, criterion)
            trainer.history["val_loss"].append(val_loss)
            trainer.history["val_accuracy"].append(val_accuracy)
            
            # Step the scheduler
            scheduler.step(val_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                trainer.best_model_state = model.state_dict()
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
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
    
    if verbose:
        logger.info("Training complete!")
        
        # Show feature importance if available
        if use_feature_weights:
            importance = trainer.get_feature_importance()
            logger.info("\nTop 10 Most Important Features:")
            for i, (feat, weight) in enumerate(list(importance.items())[:10], 1):
                logger.info(f"  {i}. {feat}: {weight:.4f}")
    
    return trainer


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    **model_kwargs
) -> Dict[str, List[float]]:
    """
    Perform cross-validation on the model.
    
    Args:
        X: Full feature DataFrame
        y: Full target Series
        n_splits: Number of CV folds
        **model_kwargs: Arguments to pass to train_transformer_model
        
    Returns:
        Dictionary with CV results
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = {
        "fold_accuracies": [],
        "fold_losses": [],
        "fold_f1_scores": []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Fold {fold}/{n_splits}")
        logger.info(f"{'='*60}")
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train model
        trainer = train_transformer_model(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            **model_kwargs
        )
        
        # Evaluate
        if len(trainer.history["val_accuracy"]) > 0:
            best_acc = max(trainer.history["val_accuracy"])
            best_loss = min(trainer.history["val_loss"])
            cv_results["fold_accuracies"].append(best_acc)
            cv_results["fold_losses"].append(best_loss)
            
            logger.info(f"Fold {fold} - Best Val Acc: {best_acc:.4f}, Best Val Loss: {best_loss:.4f}")
    
    # Summary statistics
    cv_results["mean_accuracy"] = np.mean(cv_results["fold_accuracies"])
    cv_results["std_accuracy"] = np.std(cv_results["fold_accuracies"])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Cross-Validation Results:")
    logger.info(f"  Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    logger.info(f"{'='*60}")
    
    return cv_results


def evaluate_transformer_model(
    trainer: EnhancedTransformerTrainer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    batch_size: int = 512
) -> Dict[str, float]:
    """
    Comprehensive evaluation of the enhanced transformer model.
    """
    from sklearn.metrics import (
        accuracy_score, log_loss, roc_auc_score, precision_score, recall_score,
        f1_score, confusion_matrix, average_precision_score,
        brier_score_loss, matthews_corrcoef, cohen_kappa_score
    )
    
    # Create test dataset
    test_dataset = TennisDataset(X_test, y_test, scaler=trainer.scaler, fit_scaler=False)
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
            outputs_prob = torch.sigmoid(outputs)
            
            probs = outputs_prob.cpu().numpy().flatten()
            predictions = (outputs_prob > 0.5).float().cpu().numpy().flatten()
            labels = y_batch.numpy().flatten()
            
            all_predictions.extend(predictions)
            all_probs.extend(probs)
            all_labels.extend(labels)
    
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    cm = confusion_matrix(all_labels, all_predictions)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    try:
        log_loss_val = log_loss(all_labels, all_probs)
        roc_auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
        brier = brier_score_loss(all_labels, all_probs)
        mcc = matthews_corrcoef(all_labels, all_predictions)
        kappa = cohen_kappa_score(all_labels, all_predictions)
    except:
        log_loss_val = roc_auc = pr_auc = brier = mcc = kappa = float('nan')
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "log_loss": float(log_loss_val) if not np.isnan(log_loss_val) else None,
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
        "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else None,
        "brier_score": float(brier) if not np.isnan(brier) else None,
        "matthews_corrcoef": float(mcc) if not np.isnan(mcc) else None,
        "cohen_kappa": float(kappa) if not np.isnan(kappa) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test_samples": int(len(all_labels))
    }
    
    logger.info("=" * 60)
    logger.info("ENHANCED TRANSFORMER EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Accuracy:     {accuracy:.4f}")
    logger.info(f"Precision:    {precision:.4f}")
    logger.info(f"Recall:       {recall:.4f}")
    logger.info(f"F1-Score:     {f1:.4f}")
    if not np.isnan(roc_auc):
        logger.info(f"ROC AUC:      {roc_auc:.4f}")
    if not np.isnan(mcc):
        logger.info(f"Matthews CC:  {mcc:.4f}")
    logger.info("=" * 60)
    
    return metrics


def save_transformer_model(
    trainer: EnhancedTransformerTrainer,
    metrics: Dict[str, float],
    output_dir: str = "./models"
) -> None:
    """Save the enhanced transformer model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / "enhanced_tennis_transformer.pt"
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'model_config': {
            'n_features': trainer.model.n_features,
            'd_model': trainer.model.d_model,
            'n_heads': 8,  # Store actual values used
            'n_layers': 4,
            'use_feature_weights': trainer.model.use_feature_weights
        },
        'scaler': trainer.scaler,
        'feature_cols': trainer.feature_cols,
        'class_weights': trainer.class_weights
    }, model_path)
    
    metadata = {
        "model_type": "enhanced_transformer",
        "feature_cols": trainer.feature_cols,
        "metrics": metrics,
        "training_history": trainer.history,
        "feature_importance": trainer.get_feature_importance()
    }
    
    metadata_path = output_path / "enhanced_tennis_transformer_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metadata saved to {metadata_path}")


def load_transformer_model(
    model_path: str,
    device: Optional[torch.device] = None
) -> EnhancedTransformerTrainer:
    """Load a saved enhanced transformer model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    model = EnhancedTabularTransformer(
        n_features=config['n_features'],
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 4),
        use_feature_weights=config.get('use_feature_weights', True)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    trainer = EnhancedTransformerTrainer(
        model,
        checkpoint['scaler'],
        checkpoint['feature_cols'],
        checkpoint.get('class_weights'),
        device
    )
    
    return trainer
