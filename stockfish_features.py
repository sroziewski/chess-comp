import os
import pandas as pd
import numpy as np
import chess
import chess.engine
import logging
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
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
        log_file = f"stockfish_features_{timestamp}.log"

    return setup_logging(log_file_name=log_file)

@log_time(name="analyze_position", log_interval=100000)
def analyze_position(fen, engine_path, depth=20, time_limit=1.0):
    """
    Analyze a chess position using Stockfish.

    Parameters
    ----------
    fen : str
        FEN string representing the chess position
    engine_path : str
        Path to the Stockfish engine executable
    depth : int, optional
        Maximum depth for analysis, by default 20
    time_limit : float, optional
        Time limit for analysis in seconds, by default 1.0

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    engine = None
    try:
        # Initialize the engine with a timeout
        engine = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=10.0)

        # Set a timeout for the engine operations
        engine.configure({"Threads": 1})  # Limit threads to avoid resource issues

        # Set up the board
        board = chess.Board(fen)

        # Set up analysis options with a strict time limit
        limit = chess.engine.Limit(depth=depth, time=time_limit)

        # Analyze the position with a timeout
        info = engine.analyse(board, limit, timeout=max(time_limit * 2, 5.0))

        # Extract features
        result = {
            'engine_cp_score': None,
            'engine_mate_score': None,
            'engine_top_move_uci': None,
            'engine_pv_length': 0,
            'engine_analysis_depth': info.get('depth', 0),
            'engine_multipv_count': len(info.get('multipv', [])) if 'multipv' in info and isinstance(info.get('multipv'), (list, tuple)) else 1,
            'engine_nodes': info.get('nodes', 0),
            'engine_nps': info.get('nps', 0),
            'engine_time': info.get('time', 0),
            'engine_tbhits': info.get('tbhits', 0),
            'engine_hashfull': info.get('hashfull', 0),
            'engine_selective_depth': info.get('seldepth', 0)
        }

        # Extract score
        if 'score' in info:
            score = info['score'].relative
            if score.is_mate():
                result['engine_mate_score'] = score.mate()
            else:
                result['engine_cp_score'] = score.score()

        # Extract PV (Principal Variation)
        if 'pv' in info:
            pv = info['pv']
            # Ensure pv is a list or tuple before using len()
            if isinstance(pv, (list, tuple)):
                result['engine_pv_length'] = len(pv)
                if pv:
                    result['engine_top_move_uci'] = pv[0].uci()

                    # Additional PV-based features
                    if len(pv) > 1:
                        result['engine_second_move_uci'] = pv[1].uci()
                    if len(pv) > 2:
                        result['engine_third_move_uci'] = pv[2].uci()

                    # Calculate move characteristics for top move
                    top_move = pv[0]
                    result['engine_top_move_is_capture'] = int(board.is_capture(top_move))
                    result['engine_top_move_is_check'] = int(board.gives_check(top_move))
                    result['engine_top_move_is_promotion'] = int(top_move.promotion is not None)

                    # Piece type of the moving piece
                    from_square = top_move.from_square
                    piece = board.piece_at(from_square)
                    if piece:
                        result[f'engine_top_move_piece_{piece.piece_type}'] = 1
                        result['engine_top_move_piece_color'] = int(piece.color)

        return result

    except chess.engine.EngineTerminatedError as e:
        logging.error(f"Engine terminated unexpectedly while analyzing position {fen}: {str(e)}")
        return {
            'engine_cp_score': None,
            'engine_mate_score': None,
            'engine_top_move_uci': None,
            'engine_pv_length': 0,
            'engine_analysis_depth': 0,
            'error': f"Engine terminated: {str(e)}"
        }
    except chess.engine.EngineError as e:
        logging.error(f"Engine error while analyzing position {fen}: {str(e)}")
        return {
            'engine_cp_score': None,
            'engine_mate_score': None,
            'engine_top_move_uci': None,
            'engine_pv_length': 0,
            'engine_analysis_depth': 0,
            'error': f"Engine error: {str(e)}"
        }
    except TimeoutError as e:
        logging.error(f"Timeout while analyzing position {fen}: {str(e)}")
        return {
            'engine_cp_score': None,
            'engine_mate_score': None,
            'engine_top_move_uci': None,
            'engine_pv_length': 0,
            'engine_analysis_depth': 0,
            'error': f"Timeout: {str(e)}"
        }
    except Exception as e:
        logging.error(f"Error analyzing position {fen}: {str(e)}")
        return {
            'engine_cp_score': None,
            'engine_mate_score': None,
            'engine_top_move_uci': None,
            'engine_pv_length': 0,
            'engine_analysis_depth': 0,
            'error': str(e)
        }
    finally:
        # Always ensure the engine is properly closed
        if engine is not None:
            try:
                engine.quit()
            except Exception as e:
                logging.error(f"Error closing engine: {str(e)}")
                # If normal quit fails, try to terminate the process
                try:
                    engine.close()
                except:
                    pass

def process_position(args):
    """
    Process a single position for parallel execution.

    Parameters
    ----------
    args : tuple
        Tuple containing (idx, fen, engine_path, depth, time_limit)

    Returns
    -------
    tuple
        Tuple containing (idx, features)
    """
    idx, fen, engine_path, depth, time_limit = args
    features = analyze_position(fen, engine_path, depth, time_limit)
    return idx, features

@log_time(name="extract_stockfish_features")
def extract_stockfish_features(df, engine_path, depth=20, time_limit=1.0, n_jobs=1, batch_size=1000, save_intermediate=True):
    """
    Extract Stockfish features for all positions in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess positions with a 'FEN' column
    engine_path : str
        Path to the Stockfish engine executable
    depth : int, optional
        Maximum depth for analysis, by default 20
    time_limit : float, optional
        Time limit for analysis in seconds, by default 1.0
    n_jobs : int, optional
        Number of parallel jobs, by default 1
    batch_size : int, optional
        Number of positions to process in each batch, by default 1000
    save_intermediate : bool, optional
        Whether to save intermediate results after each batch, by default True

    Returns
    -------
    pandas.DataFrame
        DataFrame with Stockfish features
    """
    logger = logging.getLogger()
    logger.info(f"Extracting Stockfish features for {len(df)} positions")
    logger.info(f"Using Stockfish engine at {engine_path}")
    logger.info(f"Analysis parameters: depth={depth}, time_limit={time_limit}s, n_jobs={n_jobs}, batch_size={batch_size}")

    # Prepare arguments for parallel processing
    args_list = [(idx, row['FEN'], engine_path, depth, time_limit) for idx, row in df.iterrows()]

    # Calculate number of batches
    num_batches = (len(args_list) + batch_size - 1) // batch_size
    logger.info(f"Processing data in {num_batches} batches of up to {batch_size} positions each")

    # Process positions in batches
    all_features_dict = {}

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(args_list))
        batch_args = args_list[batch_start:batch_end]

        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} with {len(batch_args)} positions")

        # Process batch
        batch_features_dict = {}

        if n_jobs > 1:
            logger.info(f"Using parallel processing with {n_jobs} workers")
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                for idx, features in track_progress(
                    executor.map(process_position, batch_args), 
                    total=len(batch_args), 
                    description=f"Analyzing batch {batch_idx + 1}/{num_batches} (parallel)",
                    logger=logger
                ):
                    batch_features_dict[idx] = features
        else:
            logger.info("Using sequential processing")
            for args in track_progress(
                batch_args, 
                description=f"Analyzing batch {batch_idx + 1}/{num_batches}",
                logger=logger
            ):
                idx, features = process_position(args)
                batch_features_dict[idx] = features

        # Add batch results to all results
        all_features_dict.update(batch_features_dict)

        # Save intermediate results if requested
        if save_intermediate and batch_idx < num_batches - 1:
            intermediate_df = pd.DataFrame.from_dict(all_features_dict, orient='index')
            intermediate_df = intermediate_df.fillna(0)
            intermediate_file = f"stockfish_features_intermediate_batch_{batch_idx + 1}.csv"
            logger.info(f"Saving intermediate results to {intermediate_file}")
            intermediate_df.to_csv(intermediate_file)
            logger.info(f"Saved intermediate results with {intermediate_df.shape[1]} features for {len(intermediate_df)} positions")

    # Convert all results to DataFrame
    features_df = pd.DataFrame.from_dict(all_features_dict, orient='index')

    # Fill missing values
    features_df = features_df.fillna(0)

    logger.info(f"Extracted {features_df.shape[1]} Stockfish features for {len(features_df)} positions")

    return features_df

def main():
    """
    Main function to extract Stockfish features from chess positions.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract Stockfish features from chess positions")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data CSV file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test data CSV file")
    parser.add_argument("--engine_path", type=str, required=True, help="Path to Stockfish engine executable")
    parser.add_argument("--output_train", type=str, default="stockfish_features_train.csv", help="Output file for training features")
    parser.add_argument("--output_test", type=str, default="stockfish_features_test.csv", help="Output file for test features")
    parser.add_argument("--depth", type=int, default=20, help="Maximum depth for analysis")
    parser.add_argument("--time_limit", type=float, default=1.0, help="Time limit for analysis in seconds")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs (default: 1, recommended max: CPU count)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of positions to process in each batch")
    parser.add_argument("--no_save_intermediate", action="store_true", help="Don't save intermediate results after each batch")

    args = parser.parse_args()

    # Limit the number of parallel jobs to avoid overwhelming the system
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    if args.n_jobs > cpu_count:
        print(f"Warning: Requested {args.n_jobs} parallel jobs, but only {cpu_count} CPUs available.")
        print(f"Consider reducing n_jobs to {cpu_count} or less to avoid performance issues.")

    # Ensure n_jobs is at least 1
    if args.n_jobs < 1:
        args.n_jobs = 1
        print("Warning: n_jobs must be at least 1. Setting n_jobs=1.")

    # Set up logging
    logger = get_custom_logger()

    logger.info("Starting Stockfish feature extraction")

    try:
        # Load training data
        logger.info(f"Loading training data from {args.train_file}")
        start_time = time.time()
        train_df = pd.read_csv(args.train_file)
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(train_df)} training samples in {load_time:.2f}s ({len(train_df)/load_time:.1f} samples/s)")

        # Load test data
        logger.info(f"Loading test data from {args.test_file}")
        start_time = time.time()
        test_df = pd.read_csv(args.test_file)
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(test_df)} test samples in {load_time:.2f}s ({len(test_df)/load_time:.1f} samples/s)")

        # Extract Stockfish features for training data
        logger.info("Extracting Stockfish features for training data")
        train_features = extract_stockfish_features(
            train_df, 
            args.engine_path, 
            depth=args.depth, 
            time_limit=args.time_limit, 
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            save_intermediate=not args.no_save_intermediate
        )

        # Save training features
        logger.info(f"Saving training features to {args.output_train}")
        start_time = time.time()
        train_features.to_csv(args.output_train)
        save_time = time.time() - start_time
        logger.info(f"Saved {len(train_features)} training samples in {save_time:.2f}s ({len(train_features)/save_time:.1f} samples/s)")

        # Extract Stockfish features for test data
        logger.info("Extracting Stockfish features for test data")
        test_features = extract_stockfish_features(
            test_df, 
            args.engine_path, 
            depth=args.depth, 
            time_limit=args.time_limit, 
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            save_intermediate=not args.no_save_intermediate
        )

        # Save test features
        logger.info(f"Saving test features to {args.output_test}")
        start_time = time.time()
        test_features.to_csv(args.output_test)
        save_time = time.time() - start_time
        logger.info(f"Saved {len(test_features)} test samples in {save_time:.2f}s ({len(test_features)/save_time:.1f} samples/s)")

        # Log overall statistics
        total_samples = len(train_df) + len(test_df)
        logger.info(f"Processed a total of {total_samples} samples")
        logger.info(f"Extracted {train_features.shape[1]} features per sample")
        logger.info("Stockfish feature extraction completed successfully")

    except Exception as e:
        logger.error(f"Error during Stockfish feature extraction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
