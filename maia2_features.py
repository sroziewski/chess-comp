import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import math # For entropy
import os
import concurrent.futures
import time

try:
    from maia2 import model, inference
    from chess_puzzle_rating.utils.config import get_config
except ImportError as e:
    print(f"Required library not found: {e}")
    print("Please install required libraries: pip install maia2 chess_puzzle_rating")
    exit()

"""
This script generates features using Maia-2 chess models for a dataset of chess positions.
It processes each position with multiple ELO levels for both rapid and blitz models.
The script uses parallel processing to speed up computation.

The parallelization approach:
1. Each worker process initializes its own copy of the models and environment
2. Rows are processed in parallel using ProcessPoolExecutor
3. Results are collected and combined into the final DataFrame
4. Progress is tracked using tqdm

Progress bars and timing information are provided for:
1. Loading the dataset
2. Initializing columns for the output DataFrame
3. Processing rows in parallel
4. Updating the DataFrame with results
5. Saving the augmented dataset

The number of worker processes is configured in config.yaml under performance.parallel.n_workers
"""

# --- Configuration ---
INPUT_CSV_PATH = 'final_dataset_engine.csv'
OUTPUT_CSV_PATH = 'final_dataset_with_maia2_advanced_probs.csv'
ELO_LEVELS = list(range(850, 2451, 100)) # Reduced step for example brevity
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Get configuration for parallelization
config = get_config()
performance_config = config.get('performance', {})
parallel_config = performance_config.get('parallel', {})

# Determine the number of processes to use
N_WORKERS = parallel_config.get('n_workers', None)
if N_WORKERS is None:
    N_WORKERS = os.cpu_count() or 4  # Default to 4 if os.cpu_count() returns None

def calculate_entropy(probabilities_dict):
    entropy = 0.0
    if not probabilities_dict: # Handle empty dict
        return 0.0
    for p in probabilities_dict.values():
        if p > 1e-9: # p > 0, avoid log(0)
            entropy -= p * math.log2(p)
    return entropy

# Global variables to store models and environment in each worker process
_maia_rapid_model = None
_maia_blitz_model = None
_prepared_env = None

def initialize_worker():
    """
    Initialize the models and environment in each worker process.
    This function is called once per worker process.
    """
    global _maia_rapid_model, _maia_blitz_model, _prepared_env

    # Only initialize if not already initialized
    if _maia_rapid_model is None or _maia_blitz_model is None or _prepared_env is None:
        try:
            # Initialize models
            _maia_rapid_model = model.from_pretrained(type="rapid", device=DEVICE)
            _maia_blitz_model = model.from_pretrained(type="blitz", device=DEVICE)

            # Initialize environment
            _prepared_env = inference.prepare()
        except Exception as e:
            print(f"Error initializing worker: {e}")
            raise

def process_row(row_data):
    """
    Process a single row of data to generate Maia2 features.

    Parameters:
    -----------
    row_data : tuple
        Tuple containing (index, row) where row is a pandas Series

    Returns:
    --------
    dict
        Dictionary containing the index and all generated features
    """
    global _maia_rapid_model, _maia_blitz_model, _prepared_env

    # Initialize worker if needed
    if _maia_rapid_model is None or _maia_blitz_model is None or _prepared_env is None:
        initialize_worker()

    index, row = row_data
    result = {'index': index}

    fen = str(row['FEN'])
    solution_moves_str = str(row['Moves'])

    # Skip invalid rows
    if pd.isna(fen) or pd.isna(solution_moves_str) or not solution_moves_str.strip():
        return result

    try:
        first_solution_move = solution_moves_str.split(' ')[0].strip()
        if not first_solution_move:
            return result
    except IndexError:
        return result

    for elo_self in ELO_LEVELS:
        elo_oppo = elo_self

        # --- Rapid Model Inference ---
        try:
            move_probs_rapid, win_prob_val_rapid = inference.inference_each(
                _maia_rapid_model, _prepared_env, fen, elo_self, elo_oppo
            )
            success_prob_rapid = move_probs_rapid.get(first_solution_move, 0.0)
            entropy_rapid = calculate_entropy(move_probs_rapid)

            result[f'maia2_success_prob_rapid_{elo_self}'] = success_prob_rapid
            result[f'maia2_win_prob_rapid_{elo_self}'] = win_prob_val_rapid
            result[f'maia2_entropy_rapid_{elo_self}'] = entropy_rapid
        except Exception:
            result[f'maia2_success_prob_rapid_{elo_self}'] = 0.0
            result[f'maia2_win_prob_rapid_{elo_self}'] = 0.0
            result[f'maia2_entropy_rapid_{elo_self}'] = 0.0

        # --- Blitz Model Inference ---
        try:
            move_probs_blitz, win_prob_val_blitz = inference.inference_each(
                _maia_blitz_model, _prepared_env, fen, elo_self, elo_oppo
            )
            success_prob_blitz = move_probs_blitz.get(first_solution_move, 0.0)
            entropy_blitz = calculate_entropy(move_probs_blitz)

            result[f'maia2_success_prob_blitz_{elo_self}'] = success_prob_blitz
            result[f'maia2_win_prob_blitz_{elo_self}'] = win_prob_val_blitz
            result[f'maia2_entropy_blitz_{elo_self}'] = entropy_blitz
        except Exception:
            result[f'maia2_success_prob_blitz_{elo_self}'] = 0.0
            result[f'maia2_win_prob_blitz_{elo_self}'] = 0.0
            result[f'maia2_entropy_blitz_{elo_self}'] = 0.0

    return result

def main():
    print(f"Using device: {DEVICE}")
    print(f"Using {N_WORKERS} worker processes for parallel processing")

    start_time = time.time()

    print("Models and inference environment will be initialized in each worker process")

    print(f"Loading dataset from {INPUT_CSV_PATH}...")
    load_start_time = time.time()
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        load_time = time.time() - load_start_time
        print(f"Dataset loaded in {load_time:.2f} seconds. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_CSV_PATH} not found.")
        return

    if 'FEN' not in df.columns or 'Moves' not in df.columns:
        print("Error: Dataset must contain 'FEN' and 'Moves' columns.")
        return

    # Initialize new columns for the output DataFrame
    print("Initializing new columns for the output DataFrame...")
    for elo in tqdm(ELO_LEVELS, desc="Initializing columns"):
        df[f'maia2_success_prob_rapid_{elo}'] = np.nan
        df[f'maia2_win_prob_rapid_{elo}'] = np.nan
        df[f'maia2_entropy_rapid_{elo}'] = np.nan
        df[f'maia2_success_prob_blitz_{elo}'] = np.nan
        df[f'maia2_win_prob_blitz_{elo}'] = np.nan
        df[f'maia2_entropy_blitz_{elo}'] = np.nan

    print(f"Will generate features for {len(ELO_LEVELS)} ELO levels: {ELO_LEVELS}")

    # Prepare the list of (index, row) tuples
    row_data = list(df.iterrows())

    # Calculate chunk size for better load balancing
    chunk_size = max(1, len(row_data) // (N_WORKERS * 4))

    print(f"Processing {len(row_data)} rows with chunk size {chunk_size}...")

    # Process rows in parallel
    results = []
    processing_start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # Process results as they complete
        for result in tqdm(
            executor.map(process_row, row_data, chunksize=chunk_size),
            total=len(row_data),
            desc="Processing Puzzles"
        ):
            results.append(result)

    processing_time = time.time() - processing_start_time
    print(f"Processing completed in {processing_time:.2f} seconds")

    # Update the DataFrame with the results from parallel processing
    print("Updating DataFrame with results...")
    for result in tqdm(results, desc="Updating DataFrame"):
        if 'index' in result:
            idx = result.pop('index')
            for key, value in result.items():
                if key in df.columns:
                    df.loc[idx, key] = value

    print(f"\nSaving augmented dataset to {OUTPUT_CSV_PATH}...")
    save_start_time = time.time()
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    save_time = time.time() - save_start_time
    print(f"Dataset saved in {save_time:.2f} seconds.")

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()
