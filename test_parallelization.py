import pandas as pd
import time
import os
from chess_puzzle_rating.features.position_features import extract_fen_features
from chess_puzzle_rating.utils.config import get_config

def test_parallelization():
    # Print the current configuration
    config = get_config()
    parallel_config = config.get('performance', {}).get('parallel', {})
    print(f"Using {parallel_config.get('n_workers')} worker processes with {parallel_config.get('max_threads_per_worker')} threads per worker")
    
    # Create a small test dataset
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # After 1. e4
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # After 1. e4 e5
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",  # After 1. e4 e5 2. Nf3
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # After 1. e4 e5 2. Nf3 Nc6
    ]
    
    # Create a DataFrame with the test FENs
    df = pd.DataFrame({'FEN': test_fens})
    
    # Time the feature extraction
    start_time = time.time()
    features_df = extract_fen_features(df)
    end_time = time.time()
    
    # Print the results
    print(f"Feature extraction completed in {end_time - start_time:.2f} seconds")
    print(f"Generated {features_df.shape[1]} features for {features_df.shape[0]} positions")
    
    # Print some sample features
    print("\nSample features:")
    sample_cols = ['material_balance', 'white_center_control', 'black_center_control', 'white_king_safety', 'black_king_safety']
    print(features_df[sample_cols].head())
    
    return features_df

if __name__ == "__main__":
    test_parallelization()