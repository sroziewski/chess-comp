import pandas as pd
import numpy as np
from tqdm import tqdm

# Configure tqdm to work with pandas apply
tqdm.pandas()


def get_nth_puzzle_move(moves_string, n):
    """
    Extracts the Nth move (1-indexed) for the side whose turn it is
    from a space-separated string of moves.
    Assumes moves alternate between player and opponent.
    N=1 is the first player move, N=2 is the second player move, etc.
    """
    if pd.isna(moves_string) or not isinstance(moves_string, str):
        return None

    moves_list = moves_string.strip().split(' ')
    # Player moves are at indices 0, 2, 4, ...
    # So the Nth player move is at index (N-1)*2
    player_move_index = (n - 1) * 2

    if player_move_index < len(moves_list):
        return moves_list[player_move_index]
    else:
        return None  # Not enough moves in the sequence for the Nth player move


def add_engine_move_matches(df):
    """
    Adds columns for matching engine's 1st, 2nd, and 3rd predicted moves
    with the puzzle's corresponding solution moves for the active player.
    - 'is_engine_first_move_match'
    - 'is_engine_second_move_match'
    - 'is_engine_third_move_match'
    """

    # To store the values for the new columns
    match_values = {
        1: [],  # For first_move_match
        2: [],  # For second_move_match
        3: []  # For third_move_match
    }

    # Corresponding engine move columns
    engine_move_cols = {
        1: 'engine_top_move_uci',  # Assuming this is the engine's 1st best move
        2: 'engine_second_move_uci',
        3: 'engine_third_move_uci'
    }

    print("Processing rows to determine engine move matches (1st, 2nd, 3rd):")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        puzzle_moves_str = row.get('Moves')

        for n in [1, 2, 3]:  # For 1st, 2nd, and 3rd move
            puzzle_nth_move = get_nth_puzzle_move(puzzle_moves_str, n)
            engine_nth_move = row.get(engine_move_cols[n])

            current_match = 0  # Default to no match or not applicable

            if pd.notna(puzzle_nth_move) and pd.notna(engine_nth_move):
                if puzzle_nth_move == engine_nth_move:
                    current_match = 1

            match_values[n].append(current_match)

    df['is_engine_first_move_match'] = match_values[1]
    df['is_engine_second_move_match'] = match_values[2]
    df['is_engine_third_move_match'] = match_values[3]

    return df


# --- Main Script ---
if __name__ == "__main__":
    input_csv_path = 'final_dataset_engine.csv'  # <--- CHANGE THIS TO YOUR ACTUAL FILENAME
    output_csv_path = 'final_dataset_engine_with_matches.csv'  # <--- Desired output filename

    print(f"Loading dataset from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
        print("Dataset loaded successfully.")
        print(f"Shape: {df.shape}")
        # print(f"Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: File not found at {input_csv_path}")
        exit()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()

    # Ensure necessary columns exist
    required_cols = ['Moves', 'engine_top_move_uci', 'engine_second_move_uci', 'engine_third_move_uci']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print("Please ensure your CSV contains these columns from engine analysis.")
        exit()

    # Add the new features
    df_processed = add_engine_move_matches(df.copy())  # Use .copy() to avoid SettingWithCopyWarning

    # Display some info about the new columns
    print("\n--- New Column Info ---")
    for n in [1, 2, 3]:
        col_name = f"is_engine_{['first', 'second', 'third'][n - 1]}_move_match"
        print(f"\nValue counts for '{col_name}':")
        print(df_processed[col_name].value_counts(dropna=False))

        num_matches = df_processed[col_name].sum()
        num_puzzles_with_nth_move_defined_by_puzzle = df_processed.apply(
            lambda row: pd.notna(get_nth_puzzle_move(row.get('Moves'), n)), axis=1
        ).sum()
        engine_col_for_nth_move = ['engine_top_move_uci', 'engine_second_move_uci', 'engine_third_move_uci'][n - 1]
        num_puzzles_with_nth_move_defined_by_engine = df_processed[engine_col_for_nth_move].notna().sum()

        print(f"Number of puzzles where a {n}-th puzzle move exists: {num_puzzles_with_nth_move_defined_by_puzzle}")
        print(
            f"Number of puzzles where an engine's {n}-th move exists ({engine_col_for_nth_move}): {num_puzzles_with_nth_move_defined_by_engine}")
        print(f"Number of puzzles where puzzle's {n}-th move matched engine's {n}-th move: {num_matches}")

        if num_puzzles_with_nth_move_defined_by_puzzle > 0 and num_puzzles_with_nth_move_defined_by_engine > 0:
            relevant_puzzles_for_match_rate = df_processed[
                df_processed.apply(lambda row: pd.notna(get_nth_puzzle_move(row.get('Moves'), n)), axis=1) &
                df_processed[engine_col_for_nth_move].notna()
                ]
            if not relevant_puzzles_for_match_rate.empty:
                match_rate = relevant_puzzles_for_match_rate[col_name].mean() * 100
                print(f"Match rate for {n}-th move among puzzles where both are defined: {match_rate:.2f}%")

    # Check for duplicate rows
    duplicate_count = df_processed.duplicated().sum()
    if duplicate_count > 0:
        print(f"\nFound {duplicate_count} duplicate rows. Removing duplicates...")
        df_processed = df_processed.drop_duplicates()
        print(f"After removing duplicates, shape: {df_processed.shape}")

    # Save the updated DataFrame
    print(f"\nSaving updated dataset to: {output_csv_path}")
    try:
        df_processed.to_csv(output_csv_path, index=False)
        print("Successfully saved the updated dataset.")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    print("\nScript finished.")
