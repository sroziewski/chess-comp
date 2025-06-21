import os
import pandas as pd
import numpy as np
import argparse
import logging
import time
from chess_puzzle_rating.utils.progress import setup_logging, log_time, track_progress

def get_custom_logger(log_file=None):
    """
    Set up logging configuration.

    Parameters
    ----------
    log_file : str, optional
        Path to log file, by default None

    Returns
    -------
    logging.Logger
        Configured logger
    """
    if log_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = f"combine_features_{timestamp}.log"

    return setup_logging(log_file_name=log_file)

@log_time(name="convert_uci_to_pgn")
def convert_uci_to_pgn(uci_move, fen):
    """
    Convert a UCI move to PGN notation.

    Parameters
    ----------
    uci_move : str
        Move in UCI format (e.g., "e2e4")
    fen : str
        FEN string representing the position before the move

    Returns
    -------
    str
        Move in PGN format (e.g., "e4")
    """
    try:
        import chess

        # Set up the board
        board = chess.Board(fen)

        # Parse the UCI move
        move = chess.Move.from_uci(uci_move)

        # Get the SAN (Standard Algebraic Notation) representation
        san = board.san(move)

        return san

    except Exception as e:
        logging.error(f"Error converting UCI move {uci_move} to PGN: {str(e)}")
        return uci_move  # Return the original UCI move if conversion fails

@log_time(name="add_pgn_notation")
def add_pgn_notation(df, stockfish_df):
    """
    Add PGN notation for UCI moves in the Stockfish features.

    Parameters
    ----------
    df : pandas.DataFrame
        Original DataFrame with FEN column
    stockfish_df : pandas.DataFrame
        DataFrame with Stockfish features including UCI moves

    Returns
    -------
    pandas.DataFrame
        DataFrame with added PGN notation columns
    """
    logger = logging.getLogger()
    logger.info("Converting UCI moves to PGN notation")

    # Create a copy of the Stockfish features DataFrame
    result_df = stockfish_df.copy()

    # Add FEN column from original DataFrame
    result_df['FEN'] = df['FEN']

    # Convert top move UCI to PGN
    if 'engine_top_move_uci' in result_df.columns:
        logger.info("Converting top move UCI to PGN")
        pgn_moves = []

        for idx, row in track_progress(result_df.iterrows(), total=len(result_df), description="Converting top moves", logger=logger):
            if pd.notna(row['engine_top_move_uci']) and row['engine_top_move_uci'] != 0:
                pgn_move = convert_uci_to_pgn(row['engine_top_move_uci'], row['FEN'])
                pgn_moves.append(pgn_move)
            else:
                pgn_moves.append(None)

        result_df['engine_top_move_pgn'] = pgn_moves

    # Convert second move UCI to PGN if it exists
    if 'engine_second_move_uci' in result_df.columns:
        logger.info("Converting second move UCI to PGN")
        pgn_moves = []

        for idx, row in track_progress(result_df.iterrows(), total=len(result_df), description="Converting second moves", logger=logger):
            if pd.notna(row['engine_second_move_uci']) and row['engine_second_move_uci'] != 0:
                # Make the first move to get the new position
                try:
                    board = chess.Board(row['FEN'])
                    first_move = chess.Move.from_uci(row['engine_top_move_uci'])
                    board.push(first_move)

                    # Convert the second move from the new position
                    second_move = chess.Move.from_uci(row['engine_second_move_uci'])
                    pgn_move = board.san(second_move)
                    pgn_moves.append(pgn_move)
                except Exception as e:
                    logger.error(f"Error converting second move: {str(e)}")
                    pgn_moves.append(None)
            else:
                pgn_moves.append(None)

        result_df['engine_second_move_pgn'] = pgn_moves

    # Remove the FEN column as it was only needed for conversion
    result_df = result_df.drop(columns=['FEN'])

    return result_df

@log_time(name="combine_features")
def combine_features(original_df, stockfish_df, convert_to_pgn=True):
    """
    Combine original features with Stockfish features.

    Parameters
    ----------
    original_df : pandas.DataFrame
        Original DataFrame with features
    stockfish_df : pandas.DataFrame
        DataFrame with Stockfish features
    convert_to_pgn : bool, optional
        Whether to convert UCI moves to PGN notation, by default True

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame
    """
    logger = logging.getLogger()
    logger.info("Combining original features with Stockfish features")

    # Make sure the indices match
    if not original_df.index.equals(stockfish_df.index):
        logger.warning("Indices don't match. Resetting indices and using PuzzleId for merging.")

        # Check if PuzzleId exists in both DataFrames
        if 'PuzzleId' in original_df.columns and 'PuzzleId' in stockfish_df.columns:
            # Reset indices and use PuzzleId for merging
            original_df = original_df.reset_index()
            stockfish_df = stockfish_df.reset_index()

            # Convert UCI moves to PGN if requested
            if convert_to_pgn:
                stockfish_df = add_pgn_notation(original_df, stockfish_df)

            # Merge on PuzzleId
            combined_df = pd.merge(original_df, stockfish_df, on='PuzzleId', how='left')
        else:
            logger.error("PuzzleId column not found in both DataFrames. Cannot merge.")
            return None
    else:
        # Convert UCI moves to PGN if requested
        if convert_to_pgn:
            stockfish_df = add_pgn_notation(original_df, stockfish_df)

        # Combine the DataFrames
        combined_df = pd.concat([original_df, stockfish_df], axis=1)

    # Fill missing values
    combined_df = combined_df.fillna(0)

    logger.info(f"Combined DataFrame has {combined_df.shape[1]} columns")

    return combined_df

def main():
    """
    Main function to combine original features with Stockfish features.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Combine original features with Stockfish features")
    parser.add_argument("--original_train", type=str, required=True, help="Path to original training data CSV file")
    parser.add_argument("--original_test", type=str, required=True, help="Path to original test data CSV file")
    parser.add_argument("--stockfish_train", type=str, required=True, help="Path to Stockfish training features CSV file")
    parser.add_argument("--stockfish_test", type=str, required=True, help="Path to Stockfish test features CSV file")
    parser.add_argument("--output_train", type=str, default="combined_features_train.csv", help="Output file for combined training features")
    parser.add_argument("--output_test", type=str, default="combined_features_test.csv", help="Output file for combined test features")
    parser.add_argument("--no_pgn", action="store_true", help="Don't convert UCI moves to PGN notation")

    args = parser.parse_args()

    # Set up logging
    logger = get_custom_logger()

    logger.info("Starting feature combination")

    try:
        # Load original training data
        logger.info(f"Loading original training data from {args.original_train}")
        start_time = time.time()
        original_train_df = pd.read_csv(args.original_train)
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(original_train_df)} original training samples with {original_train_df.shape[1]} features in {load_time:.2f}s ({len(original_train_df)/load_time:.1f} samples/s)")

        # Load original test data
        logger.info(f"Loading original test data from {args.original_test}")
        start_time = time.time()
        original_test_df = pd.read_csv(args.original_test)
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(original_test_df)} original test samples with {original_test_df.shape[1]} features in {load_time:.2f}s ({len(original_test_df)/load_time:.1f} samples/s)")

        # Load Stockfish training features
        logger.info(f"Loading Stockfish training features from {args.stockfish_train}")
        start_time = time.time()
        stockfish_train_df = pd.read_csv(args.stockfish_train, index_col=0)
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(stockfish_train_df)} Stockfish training samples with {stockfish_train_df.shape[1]} features in {load_time:.2f}s ({len(stockfish_train_df)/load_time:.1f} samples/s)")

        # Load Stockfish test features
        logger.info(f"Loading Stockfish test features from {args.stockfish_test}")
        start_time = time.time()
        stockfish_test_df = pd.read_csv(args.stockfish_test, index_col=0)
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(stockfish_test_df)} Stockfish test samples with {stockfish_test_df.shape[1]} features in {load_time:.2f}s ({len(stockfish_test_df)/load_time:.1f} samples/s)")

        # Combine training features
        logger.info("Combining training features")
        combined_train_df = combine_features(original_train_df, stockfish_train_df, not args.no_pgn)

        # Save combined training features
        logger.info(f"Saving combined training features to {args.output_train}")
        start_time = time.time()
        combined_train_df.to_csv(args.output_train, index=False)
        save_time = time.time() - start_time
        logger.info(f"Saved {len(combined_train_df)} combined training samples in {save_time:.2f}s ({len(combined_train_df)/save_time:.1f} samples/s)")

        # Combine test features
        logger.info("Combining test features")
        combined_test_df = combine_features(original_test_df, stockfish_test_df, not args.no_pgn)

        # Save combined test features
        logger.info(f"Saving combined test features to {args.output_test}")
        start_time = time.time()
        combined_test_df.to_csv(args.output_test, index=False)
        save_time = time.time() - start_time
        logger.info(f"Saved {len(combined_test_df)} combined test samples in {save_time:.2f}s ({len(combined_test_df)/save_time:.1f} samples/s)")

        # Log overall statistics
        total_samples = len(original_train_df) + len(original_test_df)
        total_features = combined_train_df.shape[1]
        logger.info(f"Processed a total of {total_samples} samples")
        logger.info(f"Combined features: {total_features} columns per sample")
        logger.info("Feature combination completed successfully")

    except Exception as e:
        logger.error(f"Error during feature combination: {str(e)}")
        raise

if __name__ == "__main__":
    main()
