"""
Complete end-to-end pipeline for predicting ATP tennis match winners.

This module loads historical ATP match data, computes ELO ratings and form features,
trains a machine learning model, and saves it for future use.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import logging
import math
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)


def load_raw_matches(
    data_dir: str = "./data/tennis_atp",
    start_year: int = 2000,
    end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Load ATP match CSVs from Jeff Sackmann's tennis_atp dataset
    between start_year and end_year (inclusive) and concatenate into a single DataFrame.

    Assumes files are named like 'atp_matches_YYYY.csv'.

    If end_year is None, uses the current year.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        abs_path = data_path.resolve()
        error_msg = (
            f"Data directory not found: {data_dir}\n"
            f"Absolute path: {abs_path}\n\n"
            f"To fix this:\n"
            f"1. Clone Jeff Sackmann's tennis_atp repository:\n"
            f"   git clone https://github.com/JeffSackmann/tennis_atp.git\n"
            f"2. Update the data_dir parameter to point to the correct location\n"
            f"   For example: data_dir='./tennis_atp' or data_dir='/path/to/tennis_atp'\n"
            f"3. Or download the CSV files manually and place them in: {data_dir}"
        )
        raise FileNotFoundError(error_msg)
    
    if not data_path.is_dir():
        raise NotADirectoryError(f"Path exists but is not a directory: {data_dir}")
    
    if end_year is None:
        from datetime import datetime
        end_year = datetime.now().year
    
    all_dfs = []
    years_loaded = []
    
    csv_files = list(sorted(data_path.glob("atp_matches_*.csv")))
    if not csv_files:
        error_msg = (
            f"No CSV files found matching pattern 'atp_matches_*.csv' in {data_dir}\n"
            f"Absolute path: {data_path.resolve()}\n\n"
            f"Please ensure the directory contains files named like:\n"
            f"  - atp_matches_2000.csv\n"
            f"  - atp_matches_2001.csv\n"
            f"  - etc.\n\n"
            f"These files should come from: https://github.com/JeffSackmann/tennis_atp"
        )
        raise FileNotFoundError(error_msg)
    
    for csv_file in csv_files:
        # Extract year from filename
        try:
            year_str = csv_file.stem.split("_")[-1]
            year = int(year_str)
            
            if start_year <= year <= end_year:
                logger.info(f"Loading {csv_file.name}...")
                df = pd.read_csv(csv_file)
                df["source_file"] = csv_file.name
                all_dfs.append(df)
                years_loaded.append(year)
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse year from filename {csv_file.name}: {e}")
            continue
    
    if not all_dfs:
        available_years = []
        for csv_file in csv_files:
            try:
                year_str = csv_file.stem.split("_")[-1]
                year = int(year_str)
                available_years.append(year)
            except (ValueError, IndexError):
                continue
        
        if available_years:
            error_msg = (
                f"No matching CSV files found in {data_dir} for years {start_year}-{end_year}\n"
                f"Available years in dataset: {min(available_years)}-{max(available_years)}\n"
                f"Please adjust start_year and/or end_year parameters."
            )
        else:
            error_msg = (
                f"No matching CSV files found in {data_dir} for years {start_year}-{end_year}\n"
                f"Could not parse years from any filenames."
            )
        raise ValueError(error_msg)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    logger.info(f"Loaded {len(combined_df)} matches from years {min(years_loaded)}-{max(years_loaded)}")
    
    return combined_df


def preprocess_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning of the raw matches DataFrame.

    - Parse 'tourney_date' as datetime.
    - Ensure key columns are present:
        'tourney_date', 'surface', 'winner_name', 'loser_name',
        'winner_rank', 'loser_rank', 'winner_age', 'loser_age',
        'best_of'
    - Drop rows with missing winner/loser names.
    - Optionally drop rows with missing critical stats like ranks.
    - Add a 'year' column extracted from 'tourney_date'.
    """
    df = df.copy()
    
    # Parse tourney_date
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    
    # Drop rows with invalid dates
    initial_count = len(df)
    df = df.dropna(subset=["tourney_date"])
    logger.info(f"Dropped {initial_count - len(df)} rows with invalid tourney_date")
    
    # Drop rows with missing winner/loser names
    initial_count = len(df)
    df = df.dropna(subset=["winner_name", "loser_name"])
    logger.info(f"Dropped {initial_count - len(df)} rows with missing winner/loser names")
    
    # Drop rows with missing ranks (critical for features)
    initial_count = len(df)
    df = df.dropna(subset=["winner_rank", "loser_rank"])
    logger.info(f"Dropped {initial_count - len(df)} rows with missing ranks")
    
    # Add year column
    df["year"] = df["tourney_date"].dt.year
    
    # Ensure numeric columns are numeric
    numeric_cols = ["winner_rank", "loser_rank", "winner_age", "loser_age", "best_of"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    logger.info(f"Preprocessed dataset: {len(df)} matches remaining")
    logger.info(f"Date range: {df['tourney_date'].min()} to {df['tourney_date'].max()}")
    
    return df


def compute_elo(
    df: pd.DataFrame,
    k_factor: float = 32.0,
) -> pd.DataFrame:
    """
    Compute running ELO ratings for players based on match results.

    Adds the following columns to the DataFrame:
      - winner_elo_pre
      - loser_elo_pre
      - winner_elo_post
      - loser_elo_post

    ELO is computed chronologically by 'tourney_date'.
    """
    df = df.copy()
    
    # Sort chronologically
    df = df.sort_values(by=["tourney_date", "tourney_name", "round", "match_num"], 
                       ascending=[True, True, True, True], 
                       na_position="last").reset_index(drop=True)
    
    # Initialize ELO ratings dictionary
    elo_ratings = {}
    default_elo = 1500.0
    
    # Initialize new columns
    df["winner_elo_pre"] = np.nan
    df["loser_elo_pre"] = np.nan
    df["winner_elo_post"] = np.nan
    df["loser_elo_post"] = np.nan
    
    for idx, row in df.iterrows():
        winner_name = row["winner_name"]
        loser_name = row["loser_name"]
        
        # Get current ELO ratings (default 1500 if not seen before)
        winner_elo = elo_ratings.get(winner_name, default_elo)
        loser_elo = elo_ratings.get(loser_name, default_elo)
        
        # Record pre-match ELO
        df.at[idx, "winner_elo_pre"] = winner_elo
        df.at[idx, "loser_elo_pre"] = loser_elo
        
        # Compute expected scores
        exp_w = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        exp_l = 1 - exp_w
        
        # Update ELO ratings
        winner_elo_new = winner_elo + k_factor * (1 - exp_w)
        loser_elo_new = loser_elo + k_factor * (0 - exp_l)
        
        # Store updated ratings
        elo_ratings[winner_name] = winner_elo_new
        elo_ratings[loser_name] = loser_elo_new
        
        # Record post-match ELO
        df.at[idx, "winner_elo_post"] = winner_elo_new
        df.at[idx, "loser_elo_post"] = loser_elo_new
    
    logger.info(f"Computed ELO ratings for {len(elo_ratings)} unique players")
    
    return df


def add_player_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple player form and surface performance features.

    For each match, compute for winner and loser (using only prior matches):

      - overall_win_pct_to_date
      - surface_win_pct_to_date
      - matches_played_to_date
      - surface_matches_played_to_date

    Then add 'diff' features from the winner's perspective:
      - win_pct_diff = winner_win_pct_to_date - loser_win_pct_to_date
      - surface_win_pct_diff = winner_surface_win_pct_to_date - loser_surface_win_pct_to_date
      - matches_played_diff
      - surface_matches_played_diff

    Uses expanding windows grouped by player and optionally by surface.
    """
    df = df.copy()
    
    # Sort chronologically
    df = df.sort_values(by=["tourney_date", "tourney_name", "round", "match_num"], 
                       ascending=[True, True, True, True], 
                       na_position="last").reset_index(drop=True)
    
    # Initialize counters
    matches_played = {}
    matches_won = {}
    surface_matches_played = {}
    surface_matches_won = {}
    
    # Initialize new columns
    df["winner_win_pct_to_date"] = np.nan
    df["loser_win_pct_to_date"] = np.nan
    df["winner_surface_win_pct_to_date"] = np.nan
    df["loser_surface_win_pct_to_date"] = np.nan
    df["winner_matches_played_to_date"] = np.nan
    df["loser_matches_played_to_date"] = np.nan
    df["winner_surface_matches_played_to_date"] = np.nan
    df["loser_surface_matches_played_to_date"] = np.nan
    
    for idx, row in df.iterrows():
        winner_name = row["winner_name"]
        loser_name = row["loser_name"]
        surface = row.get("surface", "Unknown")
        
        # Get current stats (before this match)
        winner_matches = matches_played.get(winner_name, 0)
        winner_wins = matches_won.get(winner_name, 0)
        loser_matches = matches_played.get(loser_name, 0)
        loser_wins = matches_won.get(loser_name, 0)
        
        winner_surface_matches = surface_matches_played.get((winner_name, surface), 0)
        winner_surface_wins = surface_matches_won.get((winner_name, surface), 0)
        loser_surface_matches = surface_matches_played.get((loser_name, surface), 0)
        loser_surface_wins = surface_matches_won.get((loser_name, surface), 0)
        
        # Compute win percentages (use 0.5 as neutral if no matches played)
        winner_win_pct = winner_wins / winner_matches if winner_matches > 0 else 0.5
        loser_win_pct = loser_wins / loser_matches if loser_matches > 0 else 0.5
        
        winner_surface_win_pct = winner_surface_wins / winner_surface_matches if winner_surface_matches > 0 else 0.5
        loser_surface_win_pct = loser_surface_wins / loser_surface_matches if loser_surface_matches > 0 else 0.5
        
        # Record stats before this match
        df.at[idx, "winner_win_pct_to_date"] = winner_win_pct
        df.at[idx, "loser_win_pct_to_date"] = loser_win_pct
        df.at[idx, "winner_surface_win_pct_to_date"] = winner_surface_win_pct
        df.at[idx, "loser_surface_win_pct_to_date"] = loser_surface_win_pct
        df.at[idx, "winner_matches_played_to_date"] = winner_matches
        df.at[idx, "loser_matches_played_to_date"] = loser_matches
        df.at[idx, "winner_surface_matches_played_to_date"] = winner_surface_matches
        df.at[idx, "loser_surface_matches_played_to_date"] = loser_surface_matches
        
        # Update counters after recording (this match counts for future matches)
        matches_played[winner_name] = winner_matches + 1
        matches_won[winner_name] = winner_wins + 1
        matches_played[loser_name] = loser_matches + 1
        matches_won[loser_name] = loser_wins  # loser didn't win this match
        
        surface_matches_played[(winner_name, surface)] = winner_surface_matches + 1
        surface_matches_won[(winner_name, surface)] = winner_surface_wins + 1
        surface_matches_played[(loser_name, surface)] = loser_surface_matches + 1
        surface_matches_won[(loser_name, surface)] = loser_surface_wins
    
    # Compute diff features from winner's perspective
    df["win_pct_diff"] = df["winner_win_pct_to_date"] - df["loser_win_pct_to_date"]
    df["surface_win_pct_diff"] = df["winner_surface_win_pct_to_date"] - df["loser_surface_win_pct_to_date"]
    df["matches_played_diff"] = df["winner_matches_played_to_date"] - df["loser_matches_played_to_date"]
    df["surface_matches_played_diff"] = df["winner_surface_matches_played_to_date"] - df["loser_surface_matches_played_to_date"]
    
    logger.info("Added player form and surface features")
    
    return df


def build_model_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Given the enriched matches DataFrame (with ELO and form features),
    build a modeling dataset.

    Returns:
      - X: feature DataFrame
      - y: label Series (1 = Player 1 (winner) wins)
      - years: year Series aligned with X and y
      - feature_cols: list of feature column names used
    """
    df = df.copy()
    
    # Compute feature differences from winner's perspective
    df["elo_diff"] = df["winner_elo_pre"] - df["loser_elo_pre"]
    df["rank_diff"] = df["loser_rank"] - df["winner_rank"]  # lower rank = stronger player
    df["age_diff"] = df["winner_age"] - df["loser_age"]
    
    # Feature columns
    feature_cols = [
        "elo_diff",
        "rank_diff",
        "age_diff",
        "win_pct_diff",
        "surface_win_pct_diff",
        "matches_played_diff",
        "surface_matches_played_diff",
        "best_of"
    ]
    
    # Check which features exist
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Create feature DataFrame
    X = df[available_features].copy()
    
    # Label: always 1 since winner is Player 1
    y = pd.Series(1, index=X.index, name="y")
    
    # Get years before filtering
    years = df["year"].copy()
    
    # Drop rows with any NaN in features
    initial_count = len(X)
    mask = ~X[available_features].isna().any(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    years = years[mask].reset_index(drop=True)
    
    logger.info(f"Built modeling dataset: {len(X)} matches (dropped {initial_count - len(X)} with missing features)")
    
    return X, y, years, available_features


def train_test_split_by_year(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    test_start_year: int = 2018
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into train and test sets based on year.

    All matches with 'year' < test_start_year go to train,
    and matches with 'year' >= test_start_year go to test.
    """
    train_mask = years < test_start_year
    test_mask = years >= test_start_year
    
    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)
    
    logger.info(f"Train set: {len(X_train)} matches (years < {test_start_year})")
    logger.info(f"Test set: {len(X_test)} matches (years >= {test_start_year})")
    
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier on the training data.

    Use reasonable hyperparameters:
      - n_estimators ~ 300
      - max_depth ~ 6-10 (or None, but limit overfitting)
      - class_weight=None (class balance is near 50/50)
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight=None
    )
    
    logger.info("Training RandomForestClassifier...")
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluate the model on the test set.

    Compute:
      - accuracy
      - log_loss
      - roc_auc

    Return a dict with these metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        "accuracy": float(accuracy),
        "log_loss": float(loss),
        "roc_auc": float(roc_auc)
    }
    
    logger.info(f"Test set accuracy: {accuracy:.4f}")
    logger.info(f"Test set log loss: {loss:.4f}")
    logger.info(f"Test set ROC AUC: {roc_auc:.4f}")
    
    return metrics


def save_model_and_metadata(
    model: RandomForestClassifier,
    feature_cols: List[str],
    metrics: dict,
    output_dir: str = "./models"
) -> None:
    """
    Save the trained model and associated metadata to disk.

    - Save model as 'tennis_match_model.pkl' via joblib.
    - Save metadata JSON as 'tennis_match_model_metadata.json'
      containing:
        - feature_cols
        - metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path / "tennis_match_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save metadata
    metadata = {
        "feature_cols": feature_cols,
        "metrics": metrics
    }
    
    metadata_path = output_path / "tennis_match_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def run_pipeline(
    data_dir: str = "./data/tennis_atp",
    start_year: int = 2000,
    end_year: Optional[int] = None,
    test_start_year: int = 2018
) -> None:
    """
    End-to-end pipeline:
      1. Load raw matches
      2. Preprocess
      3. Compute ELO
      4. Add player form features
      5. Build model dataset
      6. Split train/test by year
      7. Train model
      8. Evaluate
      9. Save model and metadata
    """
    logger.info("=" * 60)
    logger.info("Starting tennis match prediction pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load raw matches
    logger.info("\n[1/9] Loading raw matches...")
    df = load_raw_matches(data_dir=data_dir, start_year=start_year, end_year=end_year)
    
    # Step 2: Preprocess
    logger.info("\n[2/9] Preprocessing matches...")
    df = preprocess_matches(df)
    
    # Step 3: Compute ELO
    logger.info("\n[3/9] Computing ELO ratings...")
    df = compute_elo(df)
    
    # Step 4: Add form features
    logger.info("\n[4/9] Adding player form and surface features...")
    df = add_player_form_features(df)
    
    # Step 5: Build model dataset
    logger.info("\n[5/9] Building modeling dataset...")
    X, y, years, feature_cols = build_model_dataset(df)
    
    # Step 6: Split train/test
    logger.info("\n[6/9] Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split_by_year(
        X, y, years, test_start_year=test_start_year
    )
    
    # Step 7: Train model
    logger.info("\n[7/9] Training model...")
    model = train_model(X_train, y_train)
    
    # Step 8: Evaluate
    logger.info("\n[8/9] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 9: Save model and metadata
    logger.info("\n[9/9] Saving model and metadata...")
    save_model_and_metadata(model, feature_cols, metrics)
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_pipeline(
        data_dir="./data/tennis_atp",
        start_year=2000,
        end_year=None,
        test_start_year=2018,
    )

