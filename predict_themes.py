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
    --n-jobs: Number of jobs to run in parallel (default: -1, use half of available cores)
    --use-gpu: Use GPU for training if available
    --log-file: Path to the log file (default: predict_themes_{timestamp}.log)
"""

import os
import pandas as pd
import numpy as np
import chess
import logging
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import concurrent.futures
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
import lightgbm as lgb
from chess_puzzle_rating.utils.progress import track_progress

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
    start_time = time.time()
    # Set an overall timeout for the entire function
    max_total_time = 15.0  # 15 seconds max for the entire function

    try:
        # Function to check if we've exceeded the total time limit
        def check_timeout():
            if time.time() - start_time > max_total_time:
                return True
            return False
        # Initialize board from FEN
        board_init_start = time.time()
        board = chess.Board(fen)
        board_init_time = time.time() - board_init_start

        features = {
            'fen_length': len(fen),
            'board_init_time': board_init_time
        }

        # Check for timeout after board initialization
        if check_timeout():
            return {
                'fen_length': len(fen),
                'board_init_time': board_init_time,
                'total_position_time': time.time() - start_time,
                'overall_timeout': True,
                'timeout_location': 'after_board_init'
            }

        # Piece counts
        piece_count_start = time.time()
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
        features['piece_count_time'] = time.time() - piece_count_start

        # Check for timeout after piece counting
        if check_timeout():
            features['overall_timeout'] = True
            features['timeout_location'] = 'after_piece_count'
            features['total_position_time'] = time.time() - start_time
            return features

        # Material balance
        material_start = time.time()
        material_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }

        white_material = sum(material_values[p.piece_type] for s, p in pieces.items() if p.color == chess.WHITE)
        black_material = sum(material_values[p.piece_type] for s, p in pieces.items() if p.color == chess.BLACK)

        features['material_balance'] = white_material - black_material
        features['total_material'] = white_material + black_material
        features['material_calc_time'] = time.time() - material_start

        # Check for timeout after material calculation
        if check_timeout():
            features['overall_timeout'] = True
            features['timeout_location'] = 'after_material_calc'
            features['total_position_time'] = time.time() - start_time
            return features

        # Check and attack features
        check_start = time.time()
        features['is_check'] = int(board.is_check())
        features['check_calc_time'] = time.time() - check_start

        # Check for timeout after check calculation
        if check_timeout():
            features['overall_timeout'] = True
            features['timeout_location'] = 'after_check_calc'
            features['total_position_time'] = time.time() - start_time
            return features

        # Count attackers and defenders for important squares only (not all 64)
        attack_start = time.time()
        attack_defend_counts = {'white_attackers': 0, 'black_attackers': 0}

        # Define important squares (center squares and piece locations)
        important_squares = set()
        # Add center squares
        for square in [chess.E4, chess.D4, chess.E5, chess.D5]:
            important_squares.add(square)
        # Add squares with pieces
        for square in pieces.keys():
            important_squares.add(square)

        # Set a timeout for the entire attackers calculation
        max_attack_calc_time = 5.0  # 5 seconds max
        squares_processed = 0

        for square in important_squares:
            # Check if we've exceeded the timeout
            if time.time() - attack_start > max_attack_calc_time:
                features['attack_calc_timeout'] = True
                features['attack_calc_squares_processed'] = squares_processed
                break

            try:
                white_attackers = len(board.attackers(chess.WHITE, square))
                black_attackers = len(board.attackers(chess.BLACK, square))

                attack_defend_counts['white_attackers'] += white_attackers
                attack_defend_counts['black_attackers'] += black_attackers
                squares_processed += 1
            except Exception as e:
                features[f'attack_calc_error_{squares_processed}'] = str(e)[:50]
                continue

        features.update(attack_defend_counts)
        features['attack_calc_time'] = time.time() - attack_start
        features['attack_calc_squares_total'] = len(important_squares)
        features['attack_calc_squares_processed'] = squares_processed

        # Check for timeout after attackers calculation
        if check_timeout():
            features['overall_timeout'] = True
            features['timeout_location'] = 'after_attack_calc'
            features['total_position_time'] = time.time() - start_time
            return features

        # King safety with timeout
        king_safety_start = time.time()
        max_king_safety_time = 3.0  # 3 seconds max

        try:
            # Get king squares
            white_king_square = board.king(chess.WHITE)
            black_king_square = board.king(chess.BLACK)

            # Process white king safety
            if white_king_square is not None and time.time() - king_safety_start < max_king_safety_time:
                try:
                    white_king_attackers = len(board.attackers(chess.BLACK, white_king_square))
                    features['white_king_attackers'] = white_king_attackers
                except Exception as e:
                    features['white_king_error'] = str(e)[:50]
                    features['white_king_attackers'] = 0
            else:
                features['white_king_attackers'] = 0
                if white_king_square is None:
                    features['white_king_missing'] = True
                elif time.time() - king_safety_start >= max_king_safety_time:
                    features['white_king_timeout'] = True

            # Process black king safety
            if black_king_square is not None and time.time() - king_safety_start < max_king_safety_time:
                try:
                    black_king_attackers = len(board.attackers(chess.WHITE, black_king_square))
                    features['black_king_attackers'] = black_king_attackers
                except Exception as e:
                    features['black_king_error'] = str(e)[:50]
                    features['black_king_attackers'] = 0
            else:
                features['black_king_attackers'] = 0
                if black_king_square is None:
                    features['black_king_missing'] = True
                elif time.time() - king_safety_start >= max_king_safety_time:
                    features['black_king_timeout'] = True

        except Exception as e:
            features['king_safety_error'] = str(e)[:50]
            features['white_king_attackers'] = 0
            features['black_king_attackers'] = 0

        features['king_safety_time'] = time.time() - king_safety_start
        if time.time() - king_safety_start >= max_king_safety_time:
            features['king_safety_timeout'] = True

        # Check for timeout after king safety calculation
        if check_timeout():
            features['overall_timeout'] = True
            features['timeout_location'] = 'after_king_safety'
            features['total_position_time'] = time.time() - start_time
            return features

        # Game phase estimation
        phase_start = time.time()
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
        features['phase_calc_time'] = time.time() - phase_start

        # Check for timeout after phase calculation
        if check_timeout():
            features['overall_timeout'] = True
            features['timeout_location'] = 'after_phase_calc'
            features['total_position_time'] = time.time() - start_time
            return features

        # Total processing time
        features['total_position_time'] = time.time() - start_time

        # Final check to ensure we're not exceeding the timeout
        if time.time() - start_time > max_total_time:
            features['overall_timeout'] = True
            features['timeout_location'] = 'final_check'

        return features
    except Exception as e:
        # Return basic error information
        return {
            'error': str(e),
            'fen_length': len(fen) if isinstance(fen, str) else 0,
            'total_position_time': time.time() - start_time
        }

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
    start_time = time.time()
    # Set an overall timeout for the entire function
    max_total_time = 10.0  # 10 seconds max for the entire function

    # Function to check if we've exceeded the total time limit
    def check_timeout():
        if time.time() - start_time > max_total_time:
            return True
        return False

    try:
        features = {
            'moves_str_length': len(moves_str) if isinstance(moves_str, str) else 0,
            'fen_length': len(fen) if isinstance(fen, str) else 0
        }

        # If moves string is empty, return basic features
        if not moves_str or pd.isna(moves_str):
            features['total_move_time'] = time.time() - start_time
            features['empty_moves'] = True
            return features

        # Parse moves
        parse_start = time.time()
        moves = moves_str.split()
        features['num_moves'] = len(moves)
        features['parse_time'] = time.time() - parse_start

        # Initialize board from FEN
        board_init_start = time.time()
        board = chess.Board(fen)
        features['board_init_time'] = time.time() - board_init_start

        # Check for timeout after board initialization
        if check_timeout():
            features['overall_timeout'] = True
            features['timeout_location'] = 'after_board_init'
            features['total_move_time'] = time.time() - start_time
            return features

        # Track move characteristics
        checks = 0
        captures = 0
        promotions = 0
        invalid_moves = 0

        # Process each move
        move_processing_start = time.time()
        max_move_processing_time = 5.0  # 5 seconds max for move processing

        for i, move_str in enumerate(moves):
            # Check for timeout during move processing
            current_time = time.time()
            if current_time - start_time > max_total_time:
                features['overall_timeout'] = True
                features['timeout_location'] = 'during_move_processing'
                features['moves_processed'] = i
                features['total_moves'] = len(moves)
                features['total_move_time'] = current_time - start_time
                break

            # Check if we've spent too much time on move processing
            if current_time - move_processing_start > max_move_processing_time:
                features['move_processing_timeout'] = True
                features['moves_processed'] = i
                features['total_moves'] = len(moves)
                break

            try:
                # Parse move
                move_parse_start = time.time()
                move = chess.Move.from_uci(move_str)

                # Check if move is a capture
                capture_check_start = time.time()
                is_capture = board.is_capture(move)
                if is_capture:
                    captures += 1

                # Check if move is a promotion
                promotion_check_start = time.time()
                if move.promotion is not None:
                    promotions += 1

                # Make the move
                push_start = time.time()
                board.push(move)

                # Check if move gives check
                check_start = time.time()
                if board.is_check():
                    checks += 1

                # Track timing for the most expensive operations
                if i == 0:  # Only track detailed timing for first move to reduce overhead
                    features['first_move_parse_time'] = time.time() - move_parse_start
                    features['first_move_capture_check_time'] = time.time() - capture_check_start
                    features['first_move_promotion_check_time'] = time.time() - promotion_check_start
                    features['first_move_push_time'] = time.time() - push_start
                    features['first_move_check_time'] = time.time() - check_start

            except Exception as move_error:
                # Track invalid moves
                invalid_moves += 1
                if invalid_moves <= 3:  # Limit the number of errors we log to avoid bloat
                    features[f'move_error_{invalid_moves}'] = str(move_error)[:100]  # Truncate long error messages
                continue

        features['move_processing_time'] = time.time() - move_processing_start
        features['invalid_moves'] = invalid_moves

        # Calculate percentages
        stats_start = time.time()
        features['pct_checks'] = checks / max(1, len(moves))
        features['pct_captures'] = captures / max(1, len(moves))
        features['pct_promotions'] = promotions / max(1, len(moves))
        features['stats_calc_time'] = time.time() - stats_start

        # Total processing time
        features['total_move_time'] = time.time() - start_time

        # Final check to ensure we're not exceeding the timeout
        if time.time() - start_time > max_total_time:
            features['overall_timeout'] = True
            features['timeout_location'] = 'final_check'

        return features
    except Exception as e:
        # Return basic error information
        return {
            'error': str(e)[:100],  # Truncate long error messages
            'moves_str_length': len(moves_str) if isinstance(moves_str, str) else 0,
            'fen_length': len(fen) if isinstance(fen, str) else 0,
            'total_move_time': time.time() - start_time
        }

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
    tuple_start_time = time.time()
    # Set an overall timeout for the entire function
    max_tuple_time = 12.0  # 12 seconds max (slightly longer than extract_move_features)

    try:
        # Check if we can unpack the tuple
        if not isinstance(moves_fen_tuple, tuple) or len(moves_fen_tuple) != 2:
            return {
                'tuple_error': 'Invalid tuple format',
                'tuple_wrapper_time': time.time() - tuple_start_time
            }

        moves_str, fen = moves_fen_tuple

        # Call extract_move_features with a timeout
        features = extract_move_features(moves_str, fen)

        # Check if we've exceeded our time limit
        if time.time() - tuple_start_time > max_tuple_time:
            features['tuple_timeout'] = True

        # Add wrapper timing information
        features['tuple_wrapper_time'] = time.time() - tuple_start_time
        return features
    except Exception as e:
        # Return basic error information if the tuple unpacking fails
        return {
            'tuple_error': str(e)[:100],  # Truncate long error messages
            'tuple_wrapper_time': time.time() - tuple_start_time
        }

def extract_features(df, n_jobs=-1):
    """
    Extract features from a DataFrame with 'FEN' and 'Moves' columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'FEN' and 'Moves' columns
    n_jobs : int, optional
        Number of jobs to run in parallel, by default -1 (use all available cores)

    Returns
    -------
    pandas.DataFrame
        DataFrame with extracted features
    """
    logger = logging.getLogger()
    logger.info("Extracting features from FEN and Moves")
    logger.info(f"Input data shape: {df.shape}")

    # Log some sample data to help with debugging
    if len(df) > 0:
        logger.info(f"Sample FEN: {df['FEN'].iloc[0]}")
        logger.info(f"Sample Moves: {df['Moves'].iloc[0]}")

    # Determine the number of workers to use
    if n_jobs <= 0:
        # Use a reasonable number of processes (half of available CPUs)
        n_jobs = max(1, (os.cpu_count() or 4) // 2)

    logger.info(f"Using {n_jobs} worker processes for feature extraction")

    # Extract position features
    logger.info("Extracting position features")
    position_features = []

    # Create batches for better logging
    batch_size = max(1, len(df) // 10)  # Create ~10 batches
    logger.info(f"Processing in batches of ~{batch_size} samples")

    # Create a list of (moves_str, fen) tuples for move features
    logger.info("Preparing move-FEN tuples for parallel processing")
    moves_fen_tuples = list(zip(df['Moves'], df['FEN']))
    logger.info(f"Created {len(moves_fen_tuples)} move-FEN tuples")

    # Use a single ProcessPoolExecutor for both feature extraction tasks
    logger.info("Starting parallel feature extraction")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Extract position features
        logger.info("Submitting position feature extraction tasks")
        position_map = executor.map(extract_position_features, df['FEN'])

        logger.info("Processing position features")
        position_start_time = time.time()
        last_progress_time = time.time()

        # Add a maximum stall time before we consider the process hung
        max_stall_time = 60.0  # 60 seconds max stall time

        for i, features in enumerate(track_progress(position_map, total=len(df), description="Position features", logger=logger)):
            # Log if we're spending too much time on a single item (potential stall)
            current_time = time.time()
            time_since_last = current_time - last_progress_time

            # Check for stalls
            if time_since_last > 10.0:  # If more than 10 seconds since last progress
                logger.warning(f"Position feature extraction stalled for {time_since_last:.2f}s on item {i}")

                # If we've been stalled for too long, create a default feature set and continue
                if time_since_last > max_stall_time:
                    logger.error(f"Position feature extraction hung for {time_since_last:.2f}s on item {i}, using default features")
                    # Create default features for this item
                    features = {
                        'error': 'Process hung, timed out after 60 seconds',
                        'process_hung': True,
                        'fen_length': len(df['FEN'].iloc[i]) if i < len(df) else 0
                    }

                if isinstance(features, dict):
                    # Log any timeout or error information
                    timeout_info = {k: v for k, v in features.items() if 'timeout' in k or 'error' in k or 'hung' in k}
                    if timeout_info:
                        logger.warning(f"Timeout/error information: {timeout_info}")

                    # Log the FEN that caused the stall
                    if i < len(df):
                        logger.warning(f"Problematic FEN: {df['FEN'].iloc[i][:100]}...")

            last_progress_time = current_time
            position_features.append(features)

            # Log progress at regular intervals
            if (i + 1) % batch_size == 0 or i == len(df) - 1:
                elapsed = time.time() - position_start_time
                logger.info(f"Processed {i+1}/{len(df)} position features in {elapsed:.2f}s ({(i+1)/elapsed:.2f} samples/s)")

                # Log feature counts to help identify potential issues
                feature_counts = sum(len(f) for f in position_features[-batch_size:])
                logger.info(f"Last batch extracted {feature_counts} position features (avg: {feature_counts/min(batch_size, i+1):.1f} per sample)")

        # Extract move features
        logger.info("Extracting move features")
        move_features = []

        logger.info("Submitting move feature extraction tasks")
        move_map = executor.map(extract_move_features_from_tuple, moves_fen_tuples)

        logger.info("Processing move features")
        move_start_time = time.time()
        last_move_progress_time = time.time()

        # Add a maximum stall time before we consider the process hung
        max_move_stall_time = 60.0  # 60 seconds max stall time

        for i, features in enumerate(track_progress(move_map, total=len(df), description="Move features", logger=logger)):
            # Log if we're spending too much time on a single item (potential stall)
            current_time = time.time()
            time_since_last = current_time - last_move_progress_time

            # Check for stalls
            if time_since_last > 10.0:  # If more than 10 seconds since last progress
                logger.warning(f"Move feature extraction stalled for {time_since_last:.2f}s on item {i}")

                # If we've been stalled for too long, create a default feature set and continue
                if time_since_last > max_move_stall_time:
                    logger.error(f"Move feature extraction hung for {time_since_last:.2f}s on item {i}, using default features")
                    # Create default features for this item
                    features = {
                        'error': 'Process hung, timed out after 60 seconds',
                        'process_hung': True,
                        'moves_str_length': len(df['Moves'].iloc[i]) if i < len(df) else 0,
                        'fen_length': len(df['FEN'].iloc[i]) if i < len(df) else 0
                    }

                if isinstance(features, dict):
                    # Log any timeout or error information
                    timeout_info = {k: v for k, v in features.items() if 'timeout' in k or 'error' in k or 'hung' in k}
                    if timeout_info:
                        logger.warning(f"Timeout/error information: {timeout_info}")

                    # Log the moves and FEN that caused the stall
                    if i < len(df):
                        logger.warning(f"Problematic Moves: {df['Moves'].iloc[i][:100]}...")
                        logger.warning(f"Problematic FEN: {df['FEN'].iloc[i][:100]}...")

            last_move_progress_time = current_time
            move_features.append(features)

            # Log progress at regular intervals
            if (i + 1) % batch_size == 0 or i == len(df) - 1:
                elapsed = time.time() - move_start_time
                logger.info(f"Processed {i+1}/{len(df)} move features in {elapsed:.2f}s ({(i+1)/elapsed:.2f} samples/s)")

                # Log feature counts to help identify potential issues
                feature_counts = sum(len(f) for f in move_features[-batch_size:])
                logger.info(f"Last batch extracted {feature_counts} move features (avg: {feature_counts/min(batch_size, i+1):.1f} per sample)")

    # Analyze timing data from position features
    logger.info("Analyzing position feature extraction timing")
    position_times = [f.get('total_position_time', 0) for f in position_features if isinstance(f, dict)]
    if position_times:
        avg_position_time = sum(position_times) / len(position_times)
        max_position_time = max(position_times)
        min_position_time = min(position_times)
        logger.info(f"Position feature extraction timing: avg={avg_position_time:.4f}s, min={min_position_time:.4f}s, max={max_position_time:.4f}s")

        # Identify slow samples
        slow_threshold = avg_position_time * 5  # 5x average time is considered slow
        slow_indices = [i for i, t in enumerate(position_times) if t > slow_threshold]
        if slow_indices:
            logger.info(f"Found {len(slow_indices)} slow position feature extractions (>5x avg time)")
            for i in slow_indices[:5]:  # Log details for up to 5 slow samples
                logger.info(f"Slow position sample {i}: {position_times[i]:.4f}s, FEN: {df['FEN'].iloc[i][:50]}...")

        # Analyze sub-timings if available
        sub_timings = {
            'board_init_time': [],
            'piece_count_time': [],
            'material_calc_time': [],
            'check_calc_time': [],
            'attack_calc_time': [],
            'king_safety_time': [],
            'phase_calc_time': []
        }

        for f in position_features:
            if isinstance(f, dict):
                for key in sub_timings:
                    if key in f:
                        sub_timings[key].append(f[key])

        for key, times in sub_timings.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                logger.info(f"  {key}: avg={avg_time:.4f}s, max={max_time:.4f}s")

    # Analyze timing data from move features
    logger.info("Analyzing move feature extraction timing")
    move_times = [f.get('total_move_time', 0) for f in move_features if isinstance(f, dict)]
    if move_times:
        avg_move_time = sum(move_times) / len(move_times)
        max_move_time = max(move_times)
        min_move_time = min(move_times)
        logger.info(f"Move feature extraction timing: avg={avg_move_time:.4f}s, min={min_move_time:.4f}s, max={max_move_time:.4f}s")

        # Identify slow samples
        slow_threshold = avg_move_time * 5  # 5x average time is considered slow
        slow_indices = [i for i, t in enumerate(move_times) if t > slow_threshold]
        if slow_indices:
            logger.info(f"Found {len(slow_indices)} slow move feature extractions (>5x avg time)")
            for i in slow_indices[:5]:  # Log details for up to 5 slow samples
                logger.info(f"Slow move sample {i}: {move_times[i]:.4f}s, Moves length: {move_features[i].get('moves_str_length', 'N/A')}")

        # Count errors
        error_count = sum(1 for f in move_features if isinstance(f, dict) and 'error' in f)
        if error_count > 0:
            logger.info(f"Found {error_count} errors in move feature extraction")
            # Log a few examples
            for i, f in enumerate(move_features):
                if isinstance(f, dict) and 'error' in f:
                    logger.info(f"Move error in sample {i}: {f['error']}")
                    if i >= 4:  # Log at most 5 errors
                        break

    # Combine features
    logger.info("Combining features")
    logger.info(f"Position features: {len(position_features)}, Move features: {len(move_features)}")
    all_features = []

    combine_start_time = time.time()
    for i in range(len(df)):
        combined = {}
        combined.update(position_features[i])
        combined.update(move_features[i])
        all_features.append(combined)

        # Log progress at regular intervals
        if (i + 1) % batch_size == 0 or i == len(df) - 1:
            elapsed = time.time() - combine_start_time
            logger.info(f"Combined {i+1}/{len(df)} feature sets in {elapsed:.2f}s ({(i+1)/elapsed:.2f} samples/s)")

    # Convert to DataFrame
    logger.info("Converting to DataFrame")
    df_start_time = time.time()
    features_df = pd.DataFrame(all_features)
    logger.info(f"Converted to DataFrame in {time.time() - df_start_time:.2f}s")

    # Fill missing values
    logger.info("Filling missing values")
    fill_start_time = time.time()
    features_df = features_df.fillna(0)
    logger.info(f"Filled missing values in {time.time() - fill_start_time:.2f}s")

    # Remove timing, error, and timeout columns to keep the feature set clean
    columns_to_remove = [col for col in features_df.columns if any(term in col for term in 
                        ['_time', 'time_', 'error', 'timeout', 'hung', 'wrapper_'])]
    if columns_to_remove:
        logger.info(f"Removing {len(columns_to_remove)} timing, error, and diagnostic columns from final features")
        features_df = features_df.drop(columns=columns_to_remove)

    logger.info(f"Extracted {features_df.shape[1]} features for {len(df)} samples")
    logger.info(f"Total feature extraction time: {time.time() - position_start_time:.2f}s")
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
    min_auc : float, optional
        Minimum AUC score for a model to be considered good enough, by default 0.7
    use_gpu : bool, optional
        Whether to use GPU for training, by default False

    Returns
    -------
    dict
        Dictionary of trained models, one for each theme
    dict
        Dictionary of model metrics, one for each theme
    """
    logger = logging.getLogger()
    logger.info("Training theme models")

    # Determine the number of workers to use
    if n_jobs <= 0:
        # Use a reasonable number of threads (half of available CPUs)
        n_jobs = max(1, (os.cpu_count() or 4) // 2)

    logger.info(f"Using {n_jobs} threads for model training")

    # Enable GPU if requested
    if use_gpu:
        logger.info("GPU training enabled")

    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # Train a model for each theme
    models = {}
    theme_metrics = {}

    # Add progress bar for theme model training
    for i, theme in enumerate(track_progress(theme_names, description="Training theme models", logger=logger)):
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
    for i in track_progress(range(len(X)), description="Predicting", logger=logger):
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
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of jobs to run in parallel (default: -1, use half of available cores)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for training if available')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to the log file')

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_file)
    logger.info("Starting theme prediction")

    # Determine the number of workers to use
    if args.n_jobs <= 0:
        # Use a reasonable number of processes (half of available CPUs)
        args.n_jobs = max(1, (os.cpu_count() or 4) // 2)

    logger.info(f"Using {args.n_jobs} worker processes for parallel tasks")

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
        models, theme_metrics = train_theme_models(train_features, y_binary, theme_names, n_jobs=args.n_jobs, use_gpu=args.use_gpu)

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
