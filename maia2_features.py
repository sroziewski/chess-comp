import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import math # For entropy

try:
    from maia2 import model, inference
except ImportError:
    print("Maia2 library not found. Please install it: pip install maia2")
    exit()

# --- Configuration ---
INPUT_CSV_PATH = 'final_dataset.csv'
OUTPUT_CSV_PATH = 'final_dataset_with_maia2_advanced_probs.csv'
ELO_LEVELS = list(range(850, 2451, 200)) # Reduced step for example brevity
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_entropy(probabilities_dict):
    entropy = 0.0
    if not probabilities_dict: # Handle empty dict
        return 0.0
    for p in probabilities_dict.values():
        if p > 1e-9: # p > 0, avoid log(0)
            entropy -= p * math.log2(p)
    return entropy

def main():
    print(f"Using device: {DEVICE}")
    print("Loading Maia-2 models...")
    try:
        maia_rapid_model = model.from_pretrained(type="rapid", device=DEVICE)
        maia_blitz_model = model.from_pretrained(type="blitz", device=DEVICE)
    except Exception as e:
        print(f"Error loading Maia-2 models: {e}")
        return

    print("Preparing Maia-2 inference environment...")
    prepared_env = inference.prepare()
    print("Preparation complete.")

    print(f"Loading dataset from {INPUT_CSV_PATH}...")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_CSV_PATH} not found.")
        return

    if 'FEN' not in df.columns or 'Moves' not in df.columns:
        print("Error: Dataset must contain 'FEN' and 'Moves' columns.")
        return

    # Initialize new columns
    for elo in ELO_LEVELS:
        df[f'maia2_success_prob_rapid_{elo}'] = np.nan
        df[f'maia2_win_prob_rapid_{elo}'] = np.nan
        df[f'maia2_entropy_rapid_{elo}'] = np.nan
        df[f'maia2_success_prob_blitz_{elo}'] = np.nan
        df[f'maia2_win_prob_blitz_{elo}'] = np.nan
        df[f'maia2_entropy_blitz_{elo}'] = np.nan

    print(f"Will generate features for {len(ELO_LEVELS)} ELO levels: {ELO_LEVELS}")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Puzzles"):
        fen = str(row['FEN'])
        solution_moves_str = str(row['Moves'])

        if pd.isna(fen) or pd.isna(solution_moves_str) or not solution_moves_str.strip():
            continue
        try:
            first_solution_move = solution_moves_str.split(' ')[0].strip()
            if not first_solution_move: continue
        except IndexError:
            continue

        for elo_self in ELO_LEVELS:
            elo_oppo = elo_self

            # --- Rapid Model Inference ---
            try:
                move_probs_rapid, win_prob_val_rapid = inference.inference_each(
                    maia_rapid_model, prepared_env, fen, elo_self, elo_oppo
                )
                success_prob_rapid = move_probs_rapid.get(first_solution_move, 0.0)
                entropy_rapid = calculate_entropy(move_probs_rapid)

                df.loc[index, f'maia2_success_prob_rapid_{elo_self}'] = success_prob_rapid
                df.loc[index, f'maia2_win_prob_rapid_{elo_self}'] = win_prob_val_rapid
                df.loc[index, f'maia2_entropy_rapid_{elo_self}'] = entropy_rapid
            except Exception: # Broad except for brevity, log specific errors in practice
                df.loc[index, f'maia2_success_prob_rapid_{elo_self}'] = 0.0
                df.loc[index, f'maia2_win_prob_rapid_{elo_self}'] = 0.0 # Or some neutral value like 0.5 for win prob
                df.loc[index, f'maia2_entropy_rapid_{elo_self}'] = 0.0

            # --- Blitz Model Inference ---
            try:
                move_probs_blitz, win_prob_val_blitz = inference.inference_each(
                    maia_blitz_model, prepared_env, fen, elo_self, elo_oppo
                )
                success_prob_blitz = move_probs_blitz.get(first_solution_move, 0.0)
                entropy_blitz = calculate_entropy(move_probs_blitz)

                df.loc[index, f'maia2_success_prob_blitz_{elo_self}'] = success_prob_blitz
                df.loc[index, f'maia2_win_prob_blitz_{elo_self}'] = win_prob_val_blitz
                df.loc[index, f'maia2_entropy_blitz_{elo_self}'] = entropy_blitz
            except Exception:
                df.loc[index, f'maia2_success_prob_blitz_{elo_self}'] = 0.0
                df.loc[index, f'maia2_win_prob_blitz_{elo_self}'] = 0.0
                df.loc[index, f'maia2_entropy_blitz_{elo_self}'] = 0.0

    print(f"\nSaving augmented dataset to {OUTPUT_CSV_PATH}...")
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Dataset saved successfully.")

if __name__ == '__main__':
    main()