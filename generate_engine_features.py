import chess
import chess.engine
import pandas as pd
import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import logging # For better logging from workers

# --- Configuration ---
# IMPORTANT: Update this path to your UCI engine executable
ENGINE_PATH = "/usr/games/stockfish" # Example for Linux/macOS
# ENGINE_PATH = "C:/path/to/stockfish.exe" # Example for Windows

# If your engine uses a specific NNUE file not loaded by default, you might need to set UCI options
# For Stockfish, it usually finds its default nnue file.
# UCI_OPTIONS = {"EvalFile": "/path/to/your.nnue"} # Example

# Analysis settings
ANALYSIS_TIME_LIMIT_SECONDS = 0.1  # Time per FEN analysis (adjust as needed)
# OR
# ANALYSIS_DEPTH = 10 # Depth per FEN analysis

OUTPUT_CSV_FILE = "/raid/sroziewski/chess/engine_features.csv"
TRAIN_FILE_INPUT = '/raid/sroziewski/chess/training_data_02_01.csv' # To get FENs from train
TEST_FILE_INPUT = '/raid/sroziewski/chess/testing_data_cropped.csv'   # To get FENs from test


# Number of parallel processes.
NUM_PROCESSES = cpu_count()
# NUM_PROCESSES = 4 # Or set a fixed number to test, e.g., half your cores

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')

# --- Engine Analysis Function (Worker Task) ---
def analyze_fen_task_v2(fen_data_tuple, engine_exe_path, time_limit_sec):
    """
    Worker function to analyze a single FEN.
    Each worker initializes its own engine.
    fen_data_tuple: (unique_id, fen_string) - unique_id can be index or PuzzleId for tracking
    engine_exe_path: Path to the engine executable
    time_limit_sec: Time limit for analysis
    """
    unique_id, fen = fen_data_tuple
    # logger.debug(f"Worker analyzing FEN: {fen}") # Too verbose for many tasks

    engine = None
    # Default features in case of any error
    features = {
        'FEN': fen, 'engine_cp_score': None, 'engine_mate_score': None,
        'engine_top_move_uci': None, 'engine_pv_length': 0, 'engine_analysis_depth': 0,
        'engine_multipv_count': 1, 'engine_nodes': 0, 'engine_nps': 0,
        'engine_time': 0, 'engine_tbhits': 0, 'engine_hashfull': 0,
        'engine_selective_depth': 0, 'engine_second_move_uci': None, 'engine_third_move_uci': None,
        'engine_top_move_is_capture': 0, 'engine_top_move_is_check': 0, 'engine_top_move_is_promotion': 0,
        'engine_top_move_piece_color': None
    }

    try:
        board = chess.Board(fen) # Validate FEN early

        # Initialize engine for this specific task/process
        engine = chess.engine.SimpleEngine.popen_uci(engine_exe_path)

        # Perform analysis
        # If using depth: limit = chess.engine.Limit(depth=analysis_depth_limit)
        limit = chess.engine.Limit(time=time_limit_sec)
        info = engine.analyse(board, limit)

        # Extract basic engine info
        features['engine_analysis_depth'] = info.get('depth', 0)
        features['engine_multipv_count'] = len(info.get('multipv', [])) if 'multipv' in info and isinstance(info.get('multipv'), (list, tuple)) else 1
        features['engine_nodes'] = info.get('nodes', 0)
        features['engine_nps'] = info.get('nps', 0)
        features['engine_time'] = info.get('time', 0)
        features['engine_tbhits'] = info.get('tbhits', 0)
        features['engine_hashfull'] = info.get('hashfull', 0)
        features['engine_selective_depth'] = info.get('seldepth', 0)

        # Extract score
        score_obj = info.get("score")
        if score_obj is not None:
            pov_score = score_obj.pov(board.turn)
            if pov_score.is_mate():
                features['engine_mate_score'] = pov_score.mate()
                # Use a large fixed value for mate for cp_score, sign indicates who is mating
                features['engine_cp_score'] = 32000 * np.sign(pov_score.mate()) if pov_score.mate() != 0 else 0
            else:
                features['engine_cp_score'] = pov_score.score()

        # Extract principal variation (PV)
        pv = info.get("pv")
        if pv and isinstance(pv, (list, tuple)):
            features['engine_pv_length'] = len(pv)
            if pv:
                features['engine_top_move_uci'] = pv[0].uci()

                # Additional PV-based features
                if len(pv) > 1:
                    features['engine_second_move_uci'] = pv[1].uci()
                if len(pv) > 2:
                    features['engine_third_move_uci'] = pv[2].uci()

                # Calculate move characteristics for top move
                top_move = pv[0]
                features['engine_top_move_is_capture'] = int(board.is_capture(top_move))
                features['engine_top_move_is_check'] = int(board.gives_check(top_move))
                features['engine_top_move_is_promotion'] = int(top_move.promotion is not None)

                # Piece type of the moving piece
                from_square = top_move.from_square
                piece = board.piece_at(from_square)
                if piece:
                    features[f'engine_top_move_piece_{piece.piece_type}'] = 1
                    features['engine_top_move_piece_color'] = int(piece.color)

    except ValueError:  # Invalid FEN
        logger.warning(f"Invalid FEN encountered and skipped: {fen}")
        features['error'] = "Invalid FEN"
    except chess.engine.EngineError as e:  # More specific engine errors
        logger.error(
            f"EngineError during analysis for FEN {fen} (ID: {unique_id}): {e}. Engine path: {engine_exe_path}")
        features['error'] = f"Engine error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error during analysis for FEN {fen} (ID: {unique_id}): {e}", exc_info=True)
        features['error'] = str(e)
    finally:
        if engine:
            try:
                engine.quit()
            except chess.engine.EngineTerminatedError:
                pass  # Engine might have already terminated if there was a severe error
            except Exception as e:
                logger.error(f"Error quitting engine for FEN {fen}: {e}")
    return features


def main_parallel_v2():
    logger.info(f"Loading FENs...")
    try:
        train_df = pd.read_csv(TRAIN_FILE_INPUT, usecols=['PuzzleId', 'FEN'])
        test_df = pd.read_csv(TEST_FILE_INPUT, usecols=['PuzzleId', 'FEN'])
    except FileNotFoundError:
        logger.error(f"{TRAIN_FILE_INPUT} or {TEST_FILE_INPUT} not found.")
        return

    combined_fens_df = pd.concat([train_df, test_df], ignore_index=True)
    # Use FEN as the primary key for uniqueness for analysis
    # Keep PuzzleId for potential reference, but FEN is what defines the board state
    unique_fens_for_analysis = combined_fens_df.drop_duplicates(subset=['FEN'])[['FEN']].copy()
    unique_fens_for_analysis['unique_id'] = unique_fens_for_analysis.index # Add a simple unique ID for tracking

    tasks_to_process = [(row['unique_id'], row['FEN']) for _, row in unique_fens_for_analysis.iterrows()]
    total_tasks = len(tasks_to_process)
    logger.info(f"Found {len(combined_fens_df)} total FENs, {total_tasks} unique FENs to analyze.")

    if total_tasks == 0:
        logger.info("No FENs to process.")
        return

    actual_num_processes = min(NUM_PROCESSES, total_tasks) # Don't start more processes than tasks
    logger.info(f"Starting parallel analysis with {actual_num_processes} processes for {total_tasks} FENs.")
    start_time = time.time()

    all_engine_features = []

    # `partial` helps fix arguments for the worker function
    # These arguments are the same for all tasks dispatched to the workers
    worker_with_fixed_args = partial(analyze_fen_task_v2,
                                     engine_exe_path=ENGINE_PATH,
                                     time_limit_sec=ANALYSIS_TIME_LIMIT_SECONDS)

    chunksize = max(1, min(256, (total_tasks // actual_num_processes // 4) + 1))
    logger.info(f"Using chunksize: {chunksize} for Pool.imap_unordered")

    try:
        with Pool(processes=actual_num_processes) as pool:
            # Using imap_unordered to get results as they are completed.
            # This is generally good for long-running tasks with many items.
            results_iterator = pool.imap_unordered(worker_with_fixed_args, tasks_to_process, chunksize=chunksize)

            processed_count = 0
            for i, result in enumerate(results_iterator):
                all_engine_features.append(result)
                processed_count += 1

                # Progress logging
                if (i + 1) % (chunksize * actual_num_processes // 4 or 100) == 0 or (i + 1) == total_tasks:
                    current_time_iter = time.time()
                    elapsed_iter = current_time_iter - start_time
                    avg_time_per_task_iter = elapsed_iter / (i + 1) if (i + 1) > 0 else 0
                    logger.info(f"Processed {i + 1}/{total_tasks} FENs. "
                                f"Elapsed: {elapsed_iter:.2f}s. Avg time/FEN: {avg_time_per_task_iter:.3f}s.")

        logger.info(f"Parallel analysis complete. Total time: {(time.time() - start_time):.2f}s")

    except Exception as e:
        logger.error(f"An error occurred during the parallel processing pool: {e}", exc_info=True)
        return  # Exit if the pool itself fails critically

    if not all_engine_features:
        logger.warning("No engine features were generated. Check logs for errors.")
        return

    engine_features_df = pd.DataFrame(all_engine_features)
    if 'FEN' not in engine_features_df.columns and total_tasks > 0:
        engine_features_df['FEN'] = [task[1] for task in tasks_to_process[:len(engine_features_df)]]

    engine_features_df.to_csv(OUTPUT_CSV_FILE, index=False)
    logger.info(f"Engine features saved to {OUTPUT_CSV_FILE}")

    # Print some stats
    logger.info("\n--- Feature Summary ---")
    logger.info(f"Total FENs submitted for processing: {total_tasks}")
    logger.info(f"Total results received: {len(engine_features_df)}")
    if not engine_features_df.empty:
        # Basic score stats
        logger.info(f"CP Score non-null: {engine_features_df['engine_cp_score'].notna().sum()}")
        logger.info(f"Mate Score non-null: {engine_features_df['engine_mate_score'].notna().sum()}")
        logger.info(f"Median CP Score: {engine_features_df['engine_cp_score'].median()}")

        # Engine analysis stats
        mean_depth_val = engine_features_df[
            'engine_analysis_depth'].mean() if 'engine_analysis_depth' in engine_features_df and engine_features_df[
            'engine_analysis_depth'].notna().any() else 'N/A'
        logger.info(
            f"Mean Analysis Depth: {mean_depth_val if isinstance(mean_depth_val, str) else f'{mean_depth_val:.2f}'}")

        # PV stats
        logger.info(f"Average PV Length: {engine_features_df['engine_pv_length'].mean():.2f}")
        logger.info(f"Top Moves Available: {engine_features_df['engine_top_move_uci'].notna().sum()}")

        # Move characteristic stats
        if 'engine_top_move_is_capture' in engine_features_df:
            capture_pct = engine_features_df['engine_top_move_is_capture'].mean() * 100
            check_pct = engine_features_df['engine_top_move_is_check'].mean() * 100
            promotion_pct = engine_features_df['engine_top_move_is_promotion'].mean() * 100
            logger.info(f"Top Move Characteristics: Captures {capture_pct:.1f}%, Checks {check_pct:.1f}%, Promotions {promotion_pct:.1f}%")

        # Engine performance stats
        if 'engine_nodes' in engine_features_df and engine_features_df['engine_nodes'].notna().any():
            avg_nodes = engine_features_df['engine_nodes'].mean()
            avg_nps = engine_features_df['engine_nps'].mean()
            logger.info(f"Average Nodes: {avg_nodes:.0f}, Average NPS: {avg_nps:.0f}")

        # Error stats
        if 'error' in engine_features_df.columns:
            error_count = engine_features_df['error'].notna().sum()
            if error_count > 0:
                logger.warning(f"Errors encountered: {error_count} ({error_count/len(engine_features_df)*100:.1f}%)")
    else:
        logger.warning("engine_features_df is empty, no stats to display.")


if __name__ == "__main__":
    if not os.path.exists(ENGINE_PATH) or not os.path.isfile(ENGINE_PATH):
        logger.critical(f"CRITICAL ERROR: Chess engine not found at '{ENGINE_PATH}'.")
        logger.critical("Please update the ENGINE_PATH variable in the script.")
    else:
        logger.info(f"Using Engine: {ENGINE_PATH}")
        logger.info(f"Analysis time limit per FEN: {ANALYSIS_TIME_LIMIT_SECONDS}s")
        main_parallel_v2()
