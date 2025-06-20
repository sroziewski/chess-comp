#!/usr/bin/env python
"""
Script to predict themes for chess puzzles in the test set.

This script:
1. Loads training data from /raid/sroziewski/chess/training_data_02_01.csv
2. Loads test data from /raid/sroziewski/chess/testing_data_cropped.csv
3. Extracts features from 'FEN' and 'Moves' columns
4. Trains a multi-label classification model to predict themes
5. Predicts themes for the test set with confidence scores
6. Filters predictions based on confidence threshold
7. Saves the results to /raid/sroziewski/chess/testing_data_with_themes_cropped.csv

Usage:
    python predict_themes.py [--train-file TRAIN_FILE] [--test-file TEST_FILE] 
                            [--output-file OUTPUT_FILE] [--confidence-threshold THRESHOLD]
                            [--n-jobs N_JOBS] [--use-gpu] [--log-file LOG_FILE]

Arguments:
    --train-file: Path to the training data file (default: /raid/sroziewski/chess/training_data_02_01.csv)
    --test-file: Path to the test data file (default: /raid/sroziewski/chess/testing_data_cropped.csv)
    --output-file: Path to the output file (default: /raid/sroziewski/chess/testing_data_with_themes_cropped.csv)
    --confidence-threshold: Confidence threshold for theme prediction (default: 0.7)
    --n-jobs: Number of jobs to run in parallel (always 1, parameter is ignored)
    --use-gpu: Use GPU for training (always enabled, parameter is ignored)
    --log-file: Path to the log file (default: predict_themes_{timestamp}.log)
"""

import os
import pandas as pd
import numpy as np
import chess
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import concurrent.futures
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
import lightgbm as lgb

# Set up logging
def setup_logging(log_file=None):
    """Set up logging configuration."""
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"predict_themes_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

# Feature extraction functions
def extract_position_features(fen):
    """
    Extract basic features from a FEN string.

    Parameters
    ----------
    fen : str
        FEN string representing a chess position

    Returns
    -------
    dict
        Dictionary of extracted features
    """
    try:
        board = chess.Board(fen)
        features = {}

        # Piece counts
        pieces = board.piece_map()
        piece_counts = {
            'white_pawns': 0, 'white_knights': 0, 'white_bishops': 0, 
            'white_rooks': 0, 'white_queens': 0, 'white_king': 0,
            'black_pawns': 0, 'black_knights': 0, 'black_bishops': 0, 
            'black_rooks': 0, 'black_queens': 0, 'black_king': 0
        }

        for square, piece in pieces.items():
            color = 'white' if piece.color == chess.WHITE else 'black'
            piece_type = ['pawns', 'knights', 'bishops', 'rooks', 'queens', 'king'][piece.piece_type - 1]
            piece_counts[f"{color}_{piece_type}"] += 1

        features.update(piece_counts)

        # Material balance
        material_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }

        white_material = sum(material_values[p.piece_type] for s, p in pieces.items() if p.color == chess.WHITE)
        black_material = sum(material_values[p.piece_type] for s, p in pieces.items() if p.color == chess.BLACK)

        features['material_balance'] = white_material - black_material
        features['total_material'] = white_material + black_material

        # Check and attack features
        features['is_check'] = int(board.is_check())

        # Count attackers and defenders for each square
        attack_defend_counts = {'white_attackers': 0, 'black_attackers': 0}

        for square in chess.SQUARES:
            white_attackers = len(board.attackers(chess.WHITE, square))
            black_attackers = len(board.attackers(chess.BLACK, square))

            attack_defend_counts['white_attackers'] += white_attackers
            attack_defend_counts['black_attackers'] += black_attackers

        features.update(attack_defend_counts)

        # King safety
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)

        if white_king_square is not None:
            white_king_attackers = len(board.attackers(chess.BLACK, white_king_square))
            features['white_king_attackers'] = white_king_attackers
        else:
            features['white_king_attackers'] = 0

        if black_king_square is not None:
            black_king_attackers = len(board.attackers(chess.WHITE, black_king_square))
            features['black_king_attackers'] = black_king_attackers
        else:
            features['black_king_attackers'] = 0

        # Game phase estimation
        if features['total_material'] < 30:
            features['is_endgame'] = 1
            features['is_middlegame'] = 0
            features['is_opening'] = 0
        elif features['total_material'] > 62:
            features['is_endgame'] = 0
            features['is_middlegame'] = 0
            features['is_opening'] = 1
        else:
            features['is_endgame'] = 0
            features['is_middlegame'] = 1
            features['is_opening'] = 0

        return features
    except Exception as e:
        # Return empty features if there's an error
        return {}

def extract_move_features(moves_str, fen):
    """
    Extract basic features from a moves string and starting FEN.

    Parameters
    ----------
    moves_str : str
        String of moves in UCI format
    fen : str
        Starting FEN position

    Returns
    -------
    dict
        Dictionary of extracted features
    """
    try:
        features = {}

        # If moves string is empty, return empty features
        if not moves_str or pd.isna(moves_str):
            return features

        # Parse moves
        moves = moves_str.split()
        features['num_moves'] = len(moves)

        # Initialize board from FEN
        board = chess.Board(fen)

        # Track move characteristics
        checks = 0
        captures = 0
        promotions = 0

        # Process each move
        for move_str in moves:
            try:
                # Parse move
                move = chess.Move.from_uci(move_str)

                # Check if move is a capture
                if board.is_capture(move):
                    captures += 1

                # Check if move is a promotion
                if move.promotion is not None:
                    promotions += 1

                # Make the move
                board.push(move)

                # Check if move gives check
                if board.is_check():
                    checks += 1

            except Exception:
                # Skip invalid moves
                continue

        features['pct_checks'] = checks / max(1, len(moves))
        features['pct_captures'] = captures / max(1, len(moves))
        features['pct_promotions'] = promotions / max(1, len(moves))

        return features
    except Exception as e:
        # Return empty features if there's an error
        return {}

def extract_move_features_from_tuple(moves_fen_tuple):
    """
    Extract move features from a tuple of (moves_str, fen).

    Parameters
    ----------
    moves_fen_tuple : tuple
        Tuple containing (moves_str, fen)

    Returns
    -------
    dict
        Dictionary of extracted features
    """
    moves_str, fen = moves_fen_tuple
    return extract_move_features(moves_str, fen)

def extract_features(df, n_jobs=-1):
    """
    Extract features from a DataFrame with 'FEN' and 'Moves' columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'FEN' and 'Moves' columns
    n_jobs : int, optional
        Number of jobs to run in parallel, by default -1 (use all available cores)
        Note: This parameter is ignored, and the function always uses 1 thread.

    Returns
    -------
    pandas.DataFrame
        DataFrame with extracted features
    """
    logger = logging.getLogger()
    logger.info("Extracting features from FEN and Moves")

    # Force single-threaded execution
    n_jobs = 1
    logger.info("Using 1 worker process for feature extraction (single-threaded mode)")

    # Extract position features
    logger.info("Extracting position features")
    position_features = []

    # Create a list of (moves_str, fen) tuples for move features
    moves_fen_tuples = list(zip(df['Moves'], df['FEN']))

    # Use a single ProcessPoolExecutor for both feature extraction tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Extract position features
        for features in tqdm(executor.map(extract_position_features, df['FEN']), 
                            total=len(df), desc="Position features"):
            position_features.append(features)

        # Extract move features
        logger.info("Extracting move features")
        move_features = []

        for features in tqdm(executor.map(extract_move_features_from_tuple, moves_fen_tuples), 
                            total=len(df), desc="Move features"):
            move_features.append(features)

    # Combine features
    logger.info("Combining features")
    all_features = []

    for i in range(len(df)):
        combined = {}
        combined.update(position_features[i])
        combined.update(move_features[i])
        all_features.append(combined)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)

    # Fill missing values
    features_df = features_df.fillna(0)

    logger.info(f"Extracted {features_df.shape[1]} features")
    return features_df

# Theme prediction functions
def prepare_theme_data(df):
    """
    Prepare theme data for multi-label classification.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'Themes' column

    Returns
    -------
    tuple
        (MultiLabelBinarizer, list of theme lists)
    """
    logger = logging.getLogger()
    logger.info("Preparing theme data")

    # Parse themes
    theme_lists = []
    for themes_str in df['Themes']:
        if pd.isna(themes_str) or not themes_str.strip():
            theme_lists.append([])
        else:
            theme_lists.append(themes_str.split())

    # Count theme frequencies
    theme_counter = Counter()
    for themes in theme_lists:
        theme_counter.update(themes)

    logger.info(f"Found {len(theme_counter)} unique themes")

    # Keep only themes that appear at least 10 times
    min_theme_freq = 10
    common_themes = [theme for theme, count in theme_counter.items() if count >= min_theme_freq]
    logger.info(f"Keeping {len(common_themes)} themes that appear at least {min_theme_freq} times")

    # Filter theme lists to include only common themes
    filtered_theme_lists = []
    for themes in theme_lists:
        filtered_theme_lists.append([theme for theme in themes if theme in common_themes])

    # Create MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=common_themes)
    mlb.fit([common_themes])

    return mlb, filtered_theme_lists

def train_theme_models(X, y_binary, theme_names, n_jobs=-1, min_auc=0.7, use_gpu=False):
    """
    Train a binary classifier for each theme.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix
    y_binary : numpy.ndarray
        Binary matrix of theme labels
    theme_names : list
        List of theme names
    n_jobs : int, optional
        Number of jobs to run in parallel, by default -1 (use all available cores)
        Note: This parameter is ignored, and the function always uses 1 thread.
    min_auc : float, optional
        Minimum AUC score for a model to be considered good enough, by default 0.7
    use_gpu : bool, optional
        Whether to use GPU for training, by default False
        Note: This parameter is ignored, and the function always uses GPU.

    Returns
    -------
    dict
        Dictionary of trained models, one for each theme
    dict
        Dictionary of model metrics, one for each theme
    """
    logger = logging.getLogger()
    logger.info("Training theme models")

    # Force single-threaded execution
    n_jobs = 1
    logger.info("Using 1 thread for model training (single-threaded mode)")

    # Force GPU usage
    use_gpu = True
    logger.info("GPU training enabled (forced)")

    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # Train a model for each theme
    models = {}
    theme_metrics = {}

    for i, theme in enumerate(theme_names):
        logger.info(f"Training model for theme '{theme}' ({i+1}/{len(theme_names)})")

        # Get binary labels for this theme
        y_train_theme = y_train[:, i]
        y_val_theme = y_val[:, i]

        # Skip themes with too few positive examples
        if sum(y_train_theme) < 10:
            logger.info(f"Skipping theme '{theme}' with only {sum(y_train_theme)} positive examples")
            continue

        # Create and train model
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 7,
            'num_leaves': 31,
            'min_child_samples': 20,
            'n_jobs': n_jobs,
            'random_state': 42
        }

        # Add GPU parameters if GPU is enabled
        if use_gpu:
            logger.info(f"Using GPU for training model for theme '{theme}'")
            params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train, 
            y_train_theme,
            eval_set=[(X_val, y_val_theme)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )

        # Evaluate model
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate metrics
        f1 = f1_score(y_val_theme, y_pred)
        auc = roc_auc_score(y_val_theme, y_pred_proba)

        # Only store model if AUC is above threshold
        if auc >= min_auc:
            models[theme] = model
            theme_metrics[theme] = {'f1': f1, 'auc': auc}
            logger.info(f"Theme '{theme}': F1={f1:.4f}, AUC={auc:.4f} - Model accepted")
        else:
            logger.info(f"Theme '{theme}': F1={f1:.4f}, AUC={auc:.4f} - Model rejected (AUC < {min_auc})")

    # Log overall metrics for accepted models
    if theme_metrics:
        mean_f1 = np.mean([metrics['f1'] for metrics in theme_metrics.values()])
        mean_auc = np.mean([metrics['auc'] for metrics in theme_metrics.values()])
        logger.info(f"Mean F1 for accepted models: {mean_f1:.4f}, Mean AUC: {mean_auc:.4f}")
    else:
        logger.warning("No models met the minimum AUC threshold")

    return models, theme_metrics

def predict_themes(models, X, theme_names, confidence_threshold=0.7):
    """
    Predict themes for a feature matrix.

    Parameters
    ----------
    models : dict
        Dictionary of trained models, one for each theme
    X : pandas.DataFrame
        Feature matrix
    theme_names : list
        List of theme names
    confidence_threshold : float, optional
        Confidence threshold for theme prediction, by default 0.7

    Returns
    -------
    tuple
        (list of theme lists, list of confidence dictionaries)
    """
    logger = logging.getLogger()
    logger.info("Predicting themes")

    # Initialize predictions and confidences
    theme_predictions = []
    confidence_scores = []

    # Make predictions for each sample
    for i in tqdm(range(len(X)), desc="Predicting"):
        sample = X.iloc[i:i+1]
        predicted_themes = []
        confidence_dict = {}

        # Predict each theme
        for theme in theme_names:
            if theme in models:
                # Get probability of positive class
                proba = models[theme].predict_proba(sample)[0, 1]

                # If probability is above threshold, predict the theme
                if proba >= confidence_threshold:
                    predicted_themes.append(theme)

                # Store confidence score
                confidence_dict[theme] = proba

        theme_predictions.append(predicted_themes)
        confidence_scores.append(confidence_dict)

    return theme_predictions, confidence_scores

def main():
    """Main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict themes for chess puzzles in the test set.')
    parser.add_argument('--train-file', type=str, default='train_probe.csv',
                        help='Path to the training data file')
    parser.add_argument('--test-file', type=str, default='testing_data_cropped.csv',
                        help='Path to the test data file')
    parser.add_argument('--output-file', type=str, default='testing_data_with_themes_cropped.csv',
                        help='Path to the output file')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help='Confidence threshold for theme prediction')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of jobs to run in parallel (always 1, parameter is ignored)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for training (always enabled, parameter is ignored)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to the log file')

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_file)
    logger.info("Starting theme prediction")

    # Force single-threaded execution
    args.n_jobs = 1
    logger.info("Using 1 worker process for all tasks (single-threaded mode)")

    try:
        # Load training data
        logger.info(f"Loading training data from {args.train_file}")
        train_df = pd.read_csv(args.train_file)
        logger.info(f"Loaded {len(train_df)} training samples")

        # Load test data
        logger.info(f"Loading test data from {args.test_file}")
        test_df = pd.read_csv(args.test_file)
        logger.info(f"Loaded {len(test_df)} test samples")

        # Extract features from training data
        logger.info("Extracting features from training data")
        train_features = extract_features(train_df, n_jobs=args.n_jobs)

        # Prepare theme data
        logger.info("Preparing theme data")
        mlb, theme_lists = prepare_theme_data(train_df)
        theme_names = mlb.classes_

        # Convert theme lists to binary matrix
        logger.info("Converting theme lists to binary matrix")
        y_binary = mlb.transform(theme_lists)

        # Train theme models
        logger.info("Training theme models")
        models, theme_metrics = train_theme_models(train_features, y_binary, theme_names, n_jobs=args.n_jobs, use_gpu=True)

        # Extract features from test data
        logger.info("Extracting features from test data")
        test_features = extract_features(test_df, n_jobs=args.n_jobs)

        # Predict themes for test data
        logger.info("Predicting themes for test data")
        theme_predictions, confidence_scores = predict_themes(
            models, test_features, theme_names, confidence_threshold=args.confidence_threshold
        )

        # Convert theme predictions to strings
        logger.info("Converting theme predictions to strings")
        theme_strings = [' '.join(themes) for themes in theme_predictions]

        # Add theme predictions to test data
        logger.info("Adding theme predictions to test data")
        test_df_with_themes = test_df.copy()
        test_df_with_themes['Themes'] = theme_strings

        # Save results
        logger.info(f"Saving results to {args.output_file}")
        test_df_with_themes.to_csv(args.output_file, index=False)

        # Log statistics
        logger.info(f"Predicted themes for {len(test_df)} test samples")
        logger.info(f"Mean number of themes per sample: {np.mean([len(themes) for themes in theme_predictions]):.2f}")
        logger.info(f"Samples with no predicted themes: {sum(len(themes) == 0 for themes in theme_predictions)}")

        logger.info("Theme prediction completed successfully")

    except Exception as e:
        logger.error(f"Error during theme prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
