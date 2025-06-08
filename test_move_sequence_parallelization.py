import pandas as pd
import time
import os
from chess_puzzle_rating.features.move_features import analyze_move_sequence
from chess_puzzle_rating.utils.config import get_config

def test_move_sequence_parallelization():
    # Print the current configuration
    config = get_config()
    parallel_config = config.get('performance', {}).get('parallel', {})
    n_workers = parallel_config.get('n_workers')
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    print(f"Using {n_workers} worker processes with {parallel_config.get('max_threads_per_worker')} threads per worker")
    
    # Create a small test dataset
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # After 1. e4
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # After 1. e4 e5
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",  # After 1. e4 e5 2. Nf3
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # After 1. e4 e5 2. Nf3 Nc6
    ]
    
    # Create test moves
    test_moves = [
        "e4 e5 Nf3 Nc6 Bc4 Bc5",  # Italian Game
        "e4 c5 Nf3 d6 d4 cxd4",  # Sicilian Defense
        "d4 Nf6 c4 g6 Nc3 Bg7",  # King's Indian Defense
        "e4 e6 d4 d5 Nc3 Bb4",  # French Defense
        "e4 c6 d4 d5 Nc3 dxe4",  # Caro-Kann Defense
    ]
    
    # Create a DataFrame with the test FENs and moves
    df = pd.DataFrame({
        'FEN': test_fens,
        'Moves': test_moves
    })
    
    # Duplicate the data to create a larger dataset for testing
    large_df = pd.concat([df] * 1000, ignore_index=True)
    print(f"Created test dataset with {len(large_df)} rows")
    
    # Time the move sequence analysis
    start_time = time.time()
    features_df = analyze_move_sequence(large_df)
    end_time = time.time()
    
    # Print the results
    print(f"Move sequence analysis completed in {end_time - start_time:.2f} seconds")
    print(f"Generated {features_df.shape[1]} features for {features_df.shape[0]} positions")
    
    # Print some sample features
    print("\nSample features:")
    sample_cols = ['move_forcing_factor', 'material_sacrifice_ratio', 'move_depth_complexity']
    if all(col in features_df.columns for col in sample_cols):
        print(features_df[sample_cols].head())
    else:
        available_cols = list(features_df.columns)[:5]
        print(f"Sample columns not found. Available columns: {available_cols}")
    
    return features_df

if __name__ == "__main__":
    test_move_sequence_parallelization()