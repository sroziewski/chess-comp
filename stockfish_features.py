import os
import pandas as pd
import numpy as np
import chess
import chess.engine
import logging
import argparse
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def setup_logging(log_file=None):
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
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

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
    try:
        # Initialize the engine
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        
        # Set up the board
        board = chess.Board(fen)
        
        # Set up analysis options
        limit = chess.engine.Limit(depth=depth, time=time_limit)
        
        # Analyze the position
        info = engine.analyse(board, limit)
        
        # Extract features
        result = {
            'engine_cp_score': None,
            'engine_mate_score': None,
            'engine_top_move_uci': None,
            'engine_pv_length': 0,
            'engine_analysis_depth': info.get('depth', 0),
            'engine_multipv_count': len(info.get('multipv', [])) if 'multipv' in info else 1,
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
        
        # Close the engine
        engine.quit()
        
        return result
    
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

def extract_stockfish_features(df, engine_path, depth=20, time_limit=1.0, n_jobs=1):
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
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with Stockfish features
    """
    logger = logging.getLogger()
    logger.info(f"Extracting Stockfish features for {len(df)} positions")
    logger.info(f"Using Stockfish engine at {engine_path}")
    logger.info(f"Analysis parameters: depth={depth}, time_limit={time_limit}s, n_jobs={n_jobs}")
    
    # Prepare arguments for parallel processing
    args_list = [(idx, row['FEN'], engine_path, depth, time_limit) for idx, row in df.iterrows()]
    
    # Process positions
    features_dict = {}
    
    if n_jobs > 1:
        logger.info(f"Using parallel processing with {n_jobs} workers")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for idx, features in tqdm(executor.map(process_position, args_list), total=len(args_list), desc="Analyzing positions"):
                features_dict[idx] = features
    else:
        logger.info("Using sequential processing")
        for args in tqdm(args_list, desc="Analyzing positions"):
            idx, features = process_position(args)
            features_dict[idx] = features
    
    # Convert to DataFrame
    features_df = pd.DataFrame.from_dict(features_dict, orient='index')
    
    # Fill missing values
    features_df = features_df.fillna(0)
    
    logger.info(f"Extracted {features_df.shape[1]} Stockfish features")
    
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
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    logger.info("Starting Stockfish feature extraction")
    
    try:
        # Load training data
        logger.info(f"Loading training data from {args.train_file}")
        train_df = pd.read_csv(args.train_file)
        logger.info(f"Loaded {len(train_df)} training samples")
        
        # Load test data
        logger.info(f"Loading test data from {args.test_file}")
        test_df = pd.read_csv(args.test_file)
        logger.info(f"Loaded {len(test_df)} test samples")
        
        # Extract Stockfish features for training data
        logger.info("Extracting Stockfish features for training data")
        train_features = extract_stockfish_features(
            train_df, 
            args.engine_path, 
            depth=args.depth, 
            time_limit=args.time_limit, 
            n_jobs=args.n_jobs
        )
        
        # Save training features
        logger.info(f"Saving training features to {args.output_train}")
        train_features.to_csv(args.output_train)
        
        # Extract Stockfish features for test data
        logger.info("Extracting Stockfish features for test data")
        test_features = extract_stockfish_features(
            test_df, 
            args.engine_path, 
            depth=args.depth, 
            time_limit=args.time_limit, 
            n_jobs=args.n_jobs
        )
        
        # Save test features
        logger.info(f"Saving test features to {args.output_test}")
        test_features.to_csv(args.output_test)
        
        logger.info("Stockfish feature extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Error during Stockfish feature extraction: {str(e)}")
        raise

if __name__ == "__main__":
    main()