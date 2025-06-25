"""
Training script for LightGBM model with autoencoder features from train_stacking.py.

This script loads features from final_dataset.csv, uses the autoencoder trained by train_stacking.py,
and applies only the LightGBM model from train_lgbm_pt_ae_no_engine_direct.py for prediction.
Unlike train_lgbm_with_stacking_ae.py, this script uses a simple 80/20 train/test split instead of cross-validation.
"""

import os
# Set the boost_compute directory to /raid/sroziewski/.boost_compute
os.environ['BOOST_COMPUTE_DEFAULT_TEMP_PATH'] = '/raid/sroziewski/.boost_compute'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA # Added for potential future use, not active
import lightgbm as lgb
import gc
import warnings
import time
import datetime
import logging
import torch

# --- Configuration Import ---
try:
    from chess_puzzle_rating.utils.config import get_config
except ImportError:
    print("ERROR: chess_puzzle_rating.utils.config not found. Please ensure it's in PYTHONPATH or install the package.")
    # Fallback dummy config if needed for basic script execution without full framework
    def get_config():
        print("Warning: Using fallback dummy config.")
        return {
            'logging': {'log_dir': 'logs', 'log_level': 'INFO', 'log_to_console': True, 'log_to_file': True, 'log_file_name': 'fallback_lgbm.log'},
            'data_paths': {'submission_file': 'submission_fallback.txt'},
            'training': {'random_state': 42, 'lgbm_early_stopping_rounds': 50},
            'autoencoder': {'model_save_dir': 'models_fallback', 'batch_size_per_gpu': 64}, # sequence_length and embedding_dim might be needed by ProbAutoencoder
            'gpu': {'target_num_gpus': 1},
            'lgbm_params': {
                'objective': 'regression', 'metric': 'rmse', 'n_estimators': 1000,
                'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
                'bagging_freq': 1, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
                'boosting_type': 'gbdt',
            },
            'dashboards': {'enabled': True, 'output_dir': 'dashboards_fallback'}
        }

# --- Progress Tracking Import ---
try:
    from chess_puzzle_rating.utils.progress import (
        setup_logging, get_logger, log_time, ProgressTracker,
        track_progress, record_metric, create_performance_dashboard
    )
except ImportError:
    print("ERROR: chess_puzzle_rating.utils.progress not found. Using basic logging.")
    # Fallback basic logging setup
    def setup_logging(log_dir, log_level, log_to_console, log_to_file, log_file_name):
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("chess_puzzle_rating_fallback")
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, log_file_name))
            fh.setLevel(log_level)
            logger.addHandler(fh)
        return logger
    def record_metric(name, value, category): pass # Dummy
    def create_performance_dashboard(output_dir, dashboard_name):
        print(f"Warning: Dashboard creation skipped due to missing chess_puzzle_rating.utils.progress and/or dependencies.")
        return None


# --- Autoencoder Functions ---
try:
    from train_lgbm_pt_ae_no_engine_direct import (
        ProbAutoencoder, # extract_embeddings_from_autoencoder, # Not used
        get_success_prob_features_with_trained_ae,
        train_autoencoder
    )
except ImportError:
    print("ERROR: train_lgbm_pt_ae_no_engine_direct not found. Autoencoder features will not be available.")
    # Dummy ProbAutoencoder and function if not found
    class ProbAutoencoder(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = torch.nn.Linear(1,1)
        def forward(self, x): return x, x
    def get_success_prob_features_with_trained_ae(*args, **kwargs): return pd.DataFrame()
    def train_autoencoder(*args, **kwargs): return None


# --- Progress Bar Import ---
from tqdm.auto import tqdm
tqdm.pandas()

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Initialize Logging ---
config = get_config()
logging_config = config.get('logging', {})
logger = setup_logging(
    log_dir=logging_config.get('log_dir', 'logs'),
    log_level=getattr(logging, logging_config.get('log_level', 'INFO').upper(), logging.INFO),
    log_to_console=logging_config.get('log_to_console', True),
    log_to_file=logging_config.get('log_to_file', True),
    log_file_name=logging_config.get('log_file_name', f"lgbm_no_cv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
)

# --- Data Paths ---
SUBMISSION_FILE = config['data_paths']['submission_file']
if not SUBMISSION_FILE.endswith('_lgbm_with_stacking_ae_no_cv.txt'):
    SUBMISSION_FILE = SUBMISSION_FILE.replace('.txt', '_lgbm_with_stacking_ae_no_cv.txt')

logger.info("Starting LightGBM training with autoencoder features (no cross-validation).")
logger.info(f"Submission file: {SUBMISSION_FILE}")

# --- Training Parameters ---
RANDOM_STATE = config['training']['random_state']
LGBM_EARLY_STOPPING_ROUNDS = config['training']['lgbm_early_stopping_rounds']

# --- PyTorch Autoencoder Config ---
MODEL_SAVE_DIR = config['autoencoder']['model_save_dir']
AE_MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "autoencoders")
LGBM_MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "lightgbm_no_cv_with_stacking_ae")

# --- PyTorch Autoencoder Training Config ---
AE_EPOCHS = config['autoencoder'].get('epochs', 50)
AE_LEARNING_RATE = config['autoencoder'].get('learning_rate', 0.001)
AE_VALID_SPLIT = config['autoencoder'].get('valid_split', 0.2)
AE_EARLY_STOPPING_PATIENCE = config['autoencoder'].get('early_stopping_patience', 10)
FORCE_RETRAIN_AES = config['autoencoder'].get('force_retrain', False)
PROB_SEQ_LENGTH = config['autoencoder'].get('sequence_length', 20)
EMBEDDING_DIM_PROB = config['autoencoder'].get('embedding_dim', 32)

os.makedirs(AE_MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LGBM_MODEL_SAVE_DIR, exist_ok=True)

# --- GPU Configuration ---
TARGET_NUM_GPUS = config['gpu']['target_num_gpus']
if torch.cuda.is_available():
    AVAILABLE_GPUS = torch.cuda.device_count()
    logger.info(f"Found {AVAILABLE_GPUS} CUDA GPUs.")
    GPUS_TO_USE = min(AVAILABLE_GPUS, TARGET_NUM_GPUS)
    if GPUS_TO_USE > 0:
        DEVICE_IDS = list(range(GPUS_TO_USE))
        PRIMARY_DEVICE_STR = f'cuda:{DEVICE_IDS[0]}'
        logger.info(f"Will use {GPUS_TO_USE} GPUs: {DEVICE_IDS} for PyTorch. Primary: {PRIMARY_DEVICE_STR}")
        AE_BATCH_SIZE_PER_GPU = config['autoencoder']['batch_size_per_gpu']
        AE_TOTAL_BATCH_SIZE = AE_BATCH_SIZE_PER_GPU * GPUS_TO_USE
        logger.info(f"PyTorch Autoencoder Total Batch Size for Inference: {AE_TOTAL_BATCH_SIZE}")
    else:
        DEVICE_IDS = None
        PRIMARY_DEVICE_STR = 'cpu'
        logger.info("CUDA available but 0 GPUs selected. Using CPU.")
        AE_TOTAL_BATCH_SIZE = config['autoencoder']['batch_size_per_gpu']
else:
    AVAILABLE_GPUS = 0
    GPUS_TO_USE = 0
    DEVICE_IDS = None
    PRIMARY_DEVICE_STR = 'cpu'
    logger.info("No CUDA GPUs. Using CPU for PyTorch.")
    AE_TOTAL_BATCH_SIZE = config['autoencoder']['batch_size_per_gpu']
DEVICE = torch.device(PRIMARY_DEVICE_STR)


# --- Main Script ---
if __name__ == '__main__':
    overall_start_time = time.time()
    logger.info(f"Using PyTorch on device: {DEVICE} with {GPUS_TO_USE} GPU(s) for DataParallel if > 1.")
    record_metric("gpu_count", GPUS_TO_USE, "hardware")
    record_metric("device_type", "cuda" if DEVICE.type == "cuda" else "cpu", "hardware")

    logger.info("Step 1: Loading data from final_dataset_engine_with_matches.csv...")
    data_load_start = time.time()
    try:
        combined_df = pd.read_csv('final_dataset_engine_with_matches.csv')
    except FileNotFoundError:
        logger.error("final_dataset_engine_with_matches.csv not found. Exiting.")
        exit(1)
    logger.info(f"Initial combined_df shape: {combined_df.shape}")

    if 'is_train' not in combined_df.columns:
        logger.error("'is_train' column not found in combined_df. Cannot split. Exiting.")
        exit(1)
    train_df_orig = combined_df[combined_df['is_train'] == 1].copy()
    test_df_orig = combined_df[combined_df['is_train'] == 0].copy()

    if train_df_orig.empty:
        logger.error("No training data found (train_df_orig is empty after filtering 'is_train' == 1). Exiting.")
        exit(1)

    test_puzzle_ids = test_df_orig['PuzzleId'] if 'PuzzleId' in test_df_orig.columns and not test_df_orig.empty else pd.Series([])
    if 'Rating' not in test_df_orig.columns and not test_df_orig.empty: test_df_orig['Rating'] = np.nan

    data_load_time = time.time() - data_load_start
    logger.info(f"Data loaded in {data_load_time:.2f} seconds. Train shape: {train_df_orig.shape}, Test shape: {test_df_orig.shape}")
    record_metric("data_load_time", data_load_time, "performance")
    # ... (other initial record_metric calls)

    logger.info("Step 2: Loading or training autoencoders...")
    prob_cols_all = [col for col in combined_df.columns if 'success_prob_' in col]
    rapid_prob_cols_all = sorted([col for col in prob_cols_all if 'rapid' in col], key=lambda x: int(x.split('_')[-1]))
    blitz_prob_cols_all = sorted([col for col in prob_cols_all if 'blitz' in col], key=lambda x: int(x.split('_')[-1]))
    trained_rapid_ae, trained_blitz_ae = None, None

    for ae_type, prob_cols_subset, save_path_suffix, model_var_name in [
        ("RapidProbAE", rapid_prob_cols_all, "rapid_ae_best.pth", "trained_rapid_ae"),
        ("BlitzProbAE", blitz_prob_cols_all, "blitz_ae_best.pth", "trained_blitz_ae")
    ]:
        save_path = os.path.join(AE_MODEL_SAVE_DIR, save_path_suffix)
        current_ae_model = None

        if os.path.exists(save_path) and not FORCE_RETRAIN_AES:
            logger.info(f"Loading pre-trained {ae_type} model from {save_path}...")
            try:
                current_ae_model = ProbAutoencoder().to(DEVICE)
                current_ae_model.load_state_dict(torch.load(save_path, map_location=DEVICE))
                current_ae_model.eval()
                logger.info(f"{ae_type} model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading {ae_type} model from {save_path}: {e}")
                current_ae_model = None

        # If model wasn't loaded successfully and we have the required data, train a new one
        if current_ae_model is None and prob_cols_subset and len(prob_cols_subset) == PROB_SEQ_LENGTH:
            logger.info(f"Pre-trained {ae_type} model not found or couldn't be loaded. Training a new one...")
            # Ensure we use .dropna() on the correct subset of columns before .values
            seq_data_all_for_current_ae = combined_df[prob_cols_subset].dropna().values.astype(np.float32)
            if seq_data_all_for_current_ae.shape[0] > AE_TOTAL_BATCH_SIZE:
                from torch.utils.data import TensorDataset, DataLoader
                X_tr, X_val = train_test_split(seq_data_all_for_current_ae, test_size=AE_VALID_SPLIT,
                                               random_state=RANDOM_STATE)
                loader_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr)), batch_size=AE_TOTAL_BATCH_SIZE,
                                       shuffle=True, num_workers=2, pin_memory=(DEVICE.type == 'cuda'))
                loader_val = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=AE_TOTAL_BATCH_SIZE,
                                        shuffle=False, num_workers=2, pin_memory=(DEVICE.type == 'cuda'))
                base_ae_model = ProbAutoencoder().to(DEVICE)
                current_ae_model = train_autoencoder(base_ae_model, loader_tr, loader_val, AE_EPOCHS, AE_LEARNING_RATE,
                                                     AE_EARLY_STOPPING_PATIENCE, save_path, ae_type)
                if current_ae_model:
                    logger.info(f"{ae_type} model trained and saved successfully.")
                else:
                    logger.error(f"Failed to train {ae_type} model.")
            else:
                logger.warning(f"Not enough {ae_type.split('ProbAE')[0]} data for AE training (found {seq_data_all_for_current_ae.shape[0]} sequences).")
        elif current_ae_model is None:
            logger.warning(f"{ae_type.split('ProbAE')[0]} prob columns not found/wrong length for AE.")

        if model_var_name == "trained_rapid_ae": trained_rapid_ae = current_ae_model
        elif model_var_name == "trained_blitz_ae": trained_blitz_ae = current_ae_model


    logger.info("Step 3: Extracting autoencoder features...")
    feature_engineering_start = time.time()
    if trained_rapid_ae or trained_blitz_ae:
        success_prob_df_subset = combined_df[prob_cols_all].copy() if prob_cols_all else pd.DataFrame()
        if not success_prob_df_subset.empty:
            success_prob_features_df = get_success_prob_features_with_trained_ae(
                success_prob_df_subset, trained_rapid_ae, trained_blitz_ae, DEVICE,
                batch_size=AE_TOTAL_BATCH_SIZE
            )
            combined_df = pd.concat([combined_df.reset_index(drop=True), success_prob_features_df.reset_index(drop=True)], axis=1)
            del success_prob_features_df; gc.collect()
        else: logger.warning("No success_prob columns for AE.")
    else: logger.warning("No AEs available for feature extraction.")
    logger.info(f"Feature engineering completed in {time.time() - feature_engineering_start:.2f}s. Shape: {combined_df.shape}")

    logger.info("Step 4: Preparing data for LightGBM model...")
    target_col = 'Rating'
    if target_col not in combined_df.columns:
        logger.error(f"Target column '{target_col}' not found. Exiting.")
        exit(1)

    cols_to_drop = ['PuzzleId', 'FEN', 'Moves', 'Rating', 'is_train', 'Themes', 'GameUrl', 'OpeningTags', 'idx', 'Popularity', 'NbPlays', 'engine_top_move_uci', 'engine_second_move_uci', 'engine_third_move_uci']
    feature_columns = [col for col in combined_df.columns if col not in cols_to_drop and pd.api.types.is_numeric_dtype(combined_df[col])]
    non_numeric_to_drop = [col for col in combined_df.columns if col not in cols_to_drop and col not in feature_columns]
    if non_numeric_to_drop:
        logger.warning(f"Dropping non-numeric features (or features already in cols_to_drop): {non_numeric_to_drop}")

    logger.info(f"Using {len(feature_columns)} features for LightGBM model.")

    train_processed_df = combined_df[combined_df['is_train'] == 1].copy()
    test_processed_df = combined_df[combined_df['is_train'] == 0].copy()

    if train_processed_df.empty:
        logger.error("train_processed_df is empty after feature selection and is_train filtering. Cannot train. Exiting.")
        exit(1)

    # --- CRITICAL DIAGNOSTICS FOR y_train ---
    logger.info("--- Target Variable (y_train) Diagnostics ---")
    y_train_series = train_processed_df[target_col].astype(np.float32)
    logger.info(f"y_train_series shape: {y_train_series.shape}")
    logger.info(f"y_train_series NaNs: {y_train_series.isna().sum()}")
    y_train_nona = y_train_series.dropna()

    if y_train_nona.empty:
        logger.error("CRITICAL: Target variable y_train is all NaN or empty after dropping NaNs. Cannot train. Exiting.")
        exit(1)

    num_unique_targets = y_train_nona.nunique()
    logger.info(f"y_train (non-NaN) unique values count: {num_unique_targets}")
    logger.info(f"y_train (non-NaN) unique values (first 10): {np.unique(y_train_nona)[:10]}")
    logger.info(f"y_train (non-NaN) min: {y_train_nona.min()}, max: {y_train_nona.max()}, mean: {y_train_nona.mean()}, std: {y_train_nona.std()}")
    logger.info(f"y_train (non-NaN) value counts (top 5): \n{y_train_nona.value_counts().nlargest(5)}")

    if num_unique_targets == 1:
        logger.error(f"CRITICAL: Target variable y_train (after dropping NaNs) has only ONE unique value: {y_train_nona.unique()[0]}.")
        logger.error("This means the model will learn to predict this constant value, resulting in an RMSE of 0 and perfect R2, which is misleading.")
        logger.error("The issue is with the 'Rating' column in your input data ('final_dataset.csv' for training samples).")
        logger.error("Please investigate and fix the data generation process for 'final_dataset.csv' to ensure 'Rating' has diverse values for training.")
        logger.error("Exiting script to prevent misleading results and wasted computation.")
        exit(1) # EXIT SCRIPT HERE
    elif num_unique_targets < 10: # Increased threshold for warning
        logger.warning(f"Target variable y_train has very few unique values ({num_unique_targets}). This might lead to trivial or misleading results for a regression task.")
    # --- END CRITICAL DIAGNOSTICS FOR y_train ---

    # Split the training data into train (80%) and test (20%) sets
    X = train_processed_df[feature_columns].astype(np.float32)
    y = y_train_series.to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    
    logger.info(f"Split training data into train (80%) and test (20%) sets:")
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Convert boolean columns to int
    for col in X_train.columns:
        if X_train[col].dtype == 'bool':
            X_train[col] = X_train[col].astype(int)
            X_test[col] = X_test[col].astype(int)
    
    # Fill NaN values
    X_train = X_train.fillna(-999.0)
    X_test = X_test.fillna(-999.0)
    
    # Prepare submission test data
    X_submission = test_processed_df[feature_columns].astype(np.float32) if not test_processed_df.empty else pd.DataFrame(columns=feature_columns).astype(np.float32)
    
    # Convert boolean columns to int in submission data
    for col in X_submission.columns:
        if X_submission[col].dtype == 'bool':
            X_submission[col] = X_submission[col].astype(int)
    
    # Fill NaN values in submission data
    X_submission = X_submission.fillna(-999.0)

    logger.info("Step 5: Training LightGBM model...")
    lgb_params = config['lgbm_params'].copy()
    lgb_params['seed'] = RANDOM_STATE
    logger.info(f"LightGBM Parameters: {lgb_params}")
    lgbm_training_start = time.time()

    # Train a single LightGBM model on the training set
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse',
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=100), lgb.log_evaluation(period=100)]
    )
    
    # Save the model
    model_path = os.path.join(LGBM_MODEL_SAVE_DIR, f"lgbm_model_iter_{model.best_iteration_}.txt")
    model.booster_.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

    # Get feature importances
    feature_importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    logger.info(f"Feature Importances (Top 20):\n{feature_importances.head(20)}")
    logger.info(f"LGBM training completed in {time.time() - lgbm_training_start:.2f}s")

    logger.info("Step 6: Evaluating model on test set...")
    # Make predictions on the test set
    test_preds = model.predict(X_test, num_iteration=model.best_iteration_)
    
    # Calculate metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    logger.info(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    record_metric("test_rmse", test_rmse, "model_performance")
    record_metric("test_mae", test_mae, "model_performance")
    record_metric("test_r2", test_r2, "model_performance")

    # Optional: Analyze RMSE by rating range
    try:
        rating_ranges = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000, float('inf'))]
        logger.info("RMSE by rating range:")
        for low, high in rating_ranges:
            mask = (y_test >= low) & (y_test < high)
            if mask.sum() > 0:
                range_rmse = np.sqrt(mean_squared_error(y_test[mask], test_preds[mask]))
                logger.info(f"  {low}-{high}: {range_rmse:.4f} (n={mask.sum()})")
                record_metric(f"rmse_{low}_{high}", range_rmse, "model_performance_by_range")
    except Exception as e:
        logger.warning(f"Error calculating RMSE by rating range: {e}")

    logger.info("Step 7: Generating submission file...")
    if X_submission.empty:
        logger.warning("Submission data is empty. Submission file will be empty.")
        submission_df = pd.DataFrame({'PuzzleId': [], 'Rating': []})
    else:
        # Make predictions on the submission data
        submission_preds = model.predict(X_submission, num_iteration=model.best_iteration_)
        final_predictions = np.round(submission_preds).astype(int)
        submission_df = pd.DataFrame({'PuzzleId': test_puzzle_ids, 'Rating': final_predictions})

    if not SUBMISSION_FILE.lower().endswith('.txt'): SUBMISSION_FILE = os.path.splitext(SUBMISSION_FILE)[0] + ".txt"
    try:
        with open(SUBMISSION_FILE, 'w') as f:
            if not submission_df.empty:
                for pred_rating in submission_df['Rating']: f.write(f"{pred_rating}\n")
                logger.info(f"Submission file '{SUBMISSION_FILE}' created. Head:\n{submission_df.head()}")
            else:
                f.write("")
                logger.info(f"Submission file '{SUBMISSION_FILE}' created (empty).")
    except Exception as e: logger.error(f"Error writing submission file: {e}")

    total_execution_time = time.time() - overall_start_time
    logger.info(f"Total execution time: {total_execution_time:.2f} seconds")
    record_metric("total_execution_time", total_execution_time, "performance")

    dashboard_config = config.get('dashboards', {})
    if dashboard_config.get('enabled', True):
        try:
            logger.info("Generating performance dashboard...")
            # Ensure create_performance_dashboard is defined and dependencies are met
            # This part might need matplotlib, seaborn.
            output_dir = dashboard_config.get('output_dir', 'dashboards')
            dashboard_name = f"lgbm_no_cv_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dashboard_path = create_performance_dashboard(output_dir=output_dir, dashboard_name=dashboard_name)
            if dashboard_path:
                logger.info(f"Performance dashboard created at: {dashboard_path}")
            else:
                logger.warning("Performance dashboard generation skipped or failed (check logs/dependencies).")
        except NameError: # If create_performance_dashboard is not defined due to import errors
             logger.error(f"Error creating performance dashboard: create_performance_dashboard function not available.")
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {str(e)}. Ensure visualization libraries (matplotlib, seaborn) are installed.")

    logger.info("Training script finished.")