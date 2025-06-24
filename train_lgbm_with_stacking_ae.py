"""
Training script for LightGBM model with autoencoder features from train_stacking.py.

This script loads features from final_merged_features.csv, uses the autoencoder trained by train_stacking.py,
and applies only the LightGBM model from train_lgbm_pt_ae_no_engine_direct.py for prediction.
"""

import os
# Set the boost_compute directory to /raid/sroziewski/.boost_compute
os.environ['BOOST_COMPUTE_DEFAULT_TEMP_PATH'] = '/raid/sroziewski/.boost_compute'

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
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
from chess_puzzle_rating.utils.config import get_config

# --- Progress Tracking Import ---
from chess_puzzle_rating.utils.progress import (
    setup_logging, get_logger, log_time, ProgressTracker,
    track_progress, record_metric, create_performance_dashboard
)

# --- Autoencoder Functions ---
from train_lgbm_pt_ae_no_engine_direct import (
    ProbAutoencoder, extract_embeddings_from_autoencoder, # Assuming extract_embeddings_from_autoencoder is not used here but kept for consistency
    get_success_prob_features_with_trained_ae
)

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
    log_level=getattr(logging, logging_config.get('log_level', 'INFO')),
    log_to_console=logging_config.get('log_to_console', True),
    log_to_file=logging_config.get('log_to_file', True),
    log_file_name=logging_config.get('log_file_name', f"lgbm_with_stacking_ae_diag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
)

# --- Data Paths ---
SUBMISSION_FILE = config['data_paths']['submission_file']
if not SUBMISSION_FILE.endswith('_lgbm_with_stacking_ae.txt'):
    SUBMISSION_FILE = SUBMISSION_FILE.replace('.txt', '_lgbm_with_stacking_ae.txt')

logger.info("Starting LightGBM training with autoencoder features from train_stacking.py")
logger.info(f"Submission file: {SUBMISSION_FILE}")

# --- Training Parameters ---
N_SPLITS_LGBM = config['training']['n_splits_lgbm']
RANDOM_STATE = config['training']['random_state']
LGBM_EARLY_STOPPING_ROUNDS = config['training']['lgbm_early_stopping_rounds']

# --- PyTorch Autoencoder Config ---
MODEL_SAVE_DIR = config['autoencoder']['model_save_dir']
AE_MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "autoencoders")
LGBM_MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "lightgbm_folds_with_stacking_ae")

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

# --- PyTorch Autoencoder Definition ---
# These are likely used by ProbAutoencoder if it's instantiated without arguments and relies on config internally
# PROB_SEQ_LENGTH = config['autoencoder']['sequence_length']
# EMBEDDING_DIM_PROB = config['autoencoder']['embedding_dim']

# --- Main Script ---
if __name__ == '__main__':
    # Record overall start time
    overall_start_time = time.time()

    logger.info(f"Using PyTorch on device: {DEVICE} with {GPUS_TO_USE} GPU(s) for DataParallel if > 1.")

    # Record GPU configuration
    record_metric("gpu_count", GPUS_TO_USE, "hardware")
    record_metric("device_type", "cuda" if DEVICE.type == "cuda" else "cpu", "hardware")

    # Step 1: Load data from final_merged_features.csv
    logger.info("Step 1: Loading data from final_dataset.csv..")
    data_load_start = time.time()

    logger.info("Loading final_dataset.csv..")
    try:
        combined_df = pd.read_csv('final_dataset.csv')
    except FileNotFoundError:
        logger.error("final_dataset.csv not found. Exiting.")
        exit()

    logger.info(f"Initial combined_df shape: {combined_df.shape}")
    logger.info(f"Initial combined_df columns: {combined_df.columns.tolist()}")
    logger.info(f"Data types in combined_df: \n{combined_df.dtypes.value_counts()}")


    # Split into train and test sets
    if 'is_train' not in combined_df.columns:
        logger.error("'is_train' column not found in combined_df. Cannot split. Exiting.")
        exit()

    train_df_orig = combined_df[combined_df['is_train'] == 1].copy()
    test_df_orig = combined_df[combined_df['is_train'] == 0].copy()

    if train_df_orig.empty:
        logger.error("No training data found (train_df_orig is empty). Exiting.")
        exit()
    if test_df_orig.empty:
        logger.warning("No test data found (test_df_orig is empty). Submission will be empty if this proceeds.")
        # Potentially exit if test data is required

    if 'PuzzleId' not in test_df_orig.columns and not test_df_orig.empty:
        logger.warning("'PuzzleId' not found in test_df_orig. Submission might be problematic.")
        test_puzzle_ids = pd.Series(range(len(test_df_orig))) # Placeholder if missing
    elif not test_df_orig.empty:
        test_puzzle_ids = test_df_orig['PuzzleId']
    else: # test_df_orig is empty
        test_puzzle_ids = pd.Series([])


    if 'Rating' not in test_df_orig.columns and not test_df_orig.empty:
        test_df_orig['Rating'] = np.nan # This is fine, target is not needed for test predictions

    data_load_time = time.time() - data_load_start
    logger.info(f"Data loaded in {data_load_time:.2f} seconds")
    logger.info(f"Training data shape (train_df_orig): {train_df_orig.shape}, Test data shape (test_df_orig): {test_df_orig.shape}")

    # Record data metrics
    record_metric("data_load_time", data_load_time, "performance")
    record_metric("train_rows_orig", train_df_orig.shape[0], "data_stats")
    record_metric("train_columns_orig", train_df_orig.shape[1], "data_stats")
    record_metric("test_rows_orig", test_df_orig.shape[0], "data_stats")
    record_metric("test_columns_orig", test_df_orig.shape[1], "data_stats")

    # Step 2: Load autoencoders trained by train_stacking.py
    logger.info("Step 2: Loading autoencoders trained by train_stacking.py...")

    prob_cols_all = [col for col in combined_df.columns if 'success_prob_' in col]
    # rapid_prob_cols_all = sorted([col for col in prob_cols_all if 'rapid' in col], key=lambda x: int(x.split('_')[-1]))
    # blitz_prob_cols_all = sorted([col for col in prob_cols_all if 'blitz' in col], key=lambda x: int(x.split('_')[-1]))

    trained_rapid_ae, trained_blitz_ae = None, None
    rapid_ae_save_path = os.path.join(AE_MODEL_SAVE_DIR, "rapid_ae_best.pth")
    blitz_ae_save_path = os.path.join(AE_MODEL_SAVE_DIR, "blitz_ae_best.pth")

    # Load pre-trained autoencoders
    for ae_type, save_path, model_var_name in [
        ("RapidProbAE", rapid_ae_save_path, "trained_rapid_ae"),
        ("BlitzProbAE", blitz_ae_save_path, "trained_blitz_ae")
    ]:
        if os.path.exists(save_path):
            logger.info(f"Loading pre-trained {ae_type} model from {save_path}...")
            # Assuming ProbAutoencoder takes PROB_SEQ_LENGTH and EMBEDDING_DIM_PROB from config if needed
            # Or that these are hardcoded/defaulted in its definition
            # If ProbAutoencoder requires these explicitly:
            # current_ae_model = ProbAutoencoder(PROB_SEQ_LENGTH, EMBEDDING_DIM_PROB).to(DEVICE)
            try:
                current_ae_model = ProbAutoencoder().to(DEVICE) # Ensure this matches your AE definition
                current_ae_model.load_state_dict(torch.load(save_path, map_location=DEVICE))
                current_ae_model.eval()
                logger.info(f"{ae_type} model loaded successfully.")

                if model_var_name == "trained_rapid_ae":
                    trained_rapid_ae = current_ae_model
                elif model_var_name == "trained_blitz_ae":
                    trained_blitz_ae = current_ae_model
            except Exception as e:
                logger.error(f"Error loading {ae_type} model from {save_path}: {e}")
        else:
            logger.warning(f"Pre-trained {ae_type} model not found at {save_path}. Autoencoder features will not be available.")

    # Step 3: Extract autoencoder features
    logger.info("Step 3: Extracting autoencoder features...")
    feature_engineering_start = time.time()

    if trained_rapid_ae or trained_blitz_ae:
        logger.info("Extracting success probability features with autoencoder...")
        success_prob_df_subset = combined_df[prob_cols_all].copy() if prob_cols_all else pd.DataFrame() # use .copy()
        if not success_prob_df_subset.empty:
            logger.info(f"Shape of success_prob_df_subset for AE: {success_prob_df_subset.shape}")
            success_prob_features_df = get_success_prob_features_with_trained_ae(
                success_prob_df_subset, trained_rapid_ae, trained_blitz_ae, DEVICE, # Pass AE_TOTAL_BATCH_SIZE if your function expects it
                batch_size=AE_TOTAL_BATCH_SIZE
            )
            logger.info(f"Shape of generated AE features: {success_prob_features_df.shape}")
            # Ensure index alignment for concat
            combined_df = pd.concat([combined_df.reset_index(drop=True), success_prob_features_df.reset_index(drop=True)], axis=1)
            del success_prob_features_df
            gc.collect()
        else:
            logger.warning("No success_prob columns found to pass to AE feature extractor.")
    else:
        logger.warning("No AEs available. Using only basic aggregates for success_prob (if configured).")
        # Example: if basic aggregates are fallback (not shown in original script as primary path if AE fails)
        # if 'prob_all_mean' not in combined_df.columns and prob_cols_all:
        #     combined_df['prob_all_mean'] = combined_df[prob_cols_all].mean(axis=1)
        #     combined_df['prob_all_std'] = combined_df[prob_cols_all].std(axis=1)

    feature_engineering_time = time.time() - feature_engineering_start
    logger.info(f"Feature engineering completed in {feature_engineering_time:.2f} seconds")
    logger.info(f"Shape of combined_df after AE features: {combined_df.shape}")
    record_metric("feature_engineering_time", feature_engineering_time, "performance")

    # Step 4: Prepare data for LightGBM model
    logger.info("Step 4: Preparing data for LightGBM model...")
    target_col = 'Rating'

    if target_col not in combined_df.columns:
        logger.error(f"Target column '{target_col}' not found in combined_df after feature engineering. Exiting.")
        exit()

    # Columns to exclude from features
    cols_to_drop = [
        'PuzzleId', 'FEN', 'Moves', 'Rating', 'is_train',
        'Themes', 'GameUrl', 'OpeningTags', 'idx',
        'Popularity', 'NbPlays' # Explicitly add Popularity and NbPlays as requested
    ]
    # Remove original cols if log-transformed versions exist and are preferred
    if 'Popularity_log' in combined_df.columns and 'Popularity' not in cols_to_drop:
         logger.info("Popularity_log exists, Popularity (original) will be dropped if not already.")
    if 'NbPlays_log' in combined_df.columns and 'NbPlays' not in cols_to_drop:
        logger.info("NbPlays_log exists, NbPlays (original) will be dropped if not already.")


    feature_columns = []
    for col in combined_df.columns:
        if col not in cols_to_drop:
            if pd.api.types.is_numeric_dtype(combined_df[col]):
                feature_columns.append(col)
            else:
                try:
                    # Attempt to convert to numeric if possible (e.g. object dtype that is actually numeric)
                    pd.to_numeric(combined_df[col])
                    feature_columns.append(col)
                    logger.info(f"Column {col} (dtype: {combined_df[col].dtype}) converted to numeric and kept.")
                except ValueError:
                    logger.warning(f"Dropping non-numeric feature: {col} (dtype: {combined_df[col].dtype})")

    logger.info(f"Identified {len(feature_columns)} potential feature columns.")
    logger.info(f"Feature columns: {feature_columns[:20]} ... (showing first 20)") # Log some feature names

    # Separate train and test based on the 'is_train' column from the *final* combined_df
    train_processed_df = combined_df[combined_df['is_train'] == 1].copy()
    test_processed_df = combined_df[combined_df['is_train'] == 0].copy()

    logger.info(f"Shape of train_processed_df: {train_processed_df.shape}")
    logger.info(f"Shape of test_processed_df: {test_processed_df.shape}")

    if train_processed_df.empty:
        logger.error("train_processed_df is empty after feature selection. Cannot train. Exiting.")
        exit()

    X_train = train_processed_df[feature_columns].astype(np.float32)
    y_train = train_processed_df[target_col].astype(np.float32)

    if not test_processed_df.empty:
        X_test = test_processed_df[feature_columns].astype(np.float32)
    else:
        X_test = pd.DataFrame(columns=feature_columns).astype(np.float32) # Empty dataframe with correct columns

    # --- CRITICAL DIAGNOSTICS FOR y_train ---
    logger.info("--- Target Variable (y_train) Diagnostics ---")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_train NaNs: {y_train.isna().sum()}")
    y_train_nona = y_train.dropna()
    if not y_train_nona.empty:
        logger.info(f"y_train (non-NaN) unique values count: {y_train_nona.nunique()}")
        logger.info(f"y_train (non-NaN) unique values (first 10): {np.unique(y_train_nona)[:10]}")
        logger.info(f"y_train (non-NaN) min: {y_train_nona.min()}, max: {y_train_nona.max()}, mean: {y_train_nona.mean()}, std: {y_train_nona.std()}")
        logger.info(f"y_train (non-NaN) value counts (top 5): \n{y_train_nona.value_counts().nlargest(5)}")
        if y_train_nona.nunique() < 5: # Threshold for concern
            logger.warning("Target variable y_train has very few unique values. This might lead to trivial or misleading results.")
            if y_train_nona.nunique() == 1:
                logger.error("CRITICAL: Target variable y_train is CONSTANT. Model will predict this constant. RMSE will be 0. THIS IS LIKELY THE CAUSE OF THE ISSUE.")
                # Consider exiting if this is detected: exit()
    else:
        logger.error("CRITICAL: y_train is all NaN or empty after dropping NaNs. Cannot train.")
        exit()
    # --- END CRITICAL DIAGNOSTICS FOR y_train ---

    # --- Feature Matrix (X_train) Diagnostics ---
    logger.info("--- Feature Matrix (X_train) Diagnostics ---")
    logger.info(f"X_train shape before processing: {X_train.shape}")
    logger.info(f"X_train NaNs before fill: {X_train.isna().sum().sum()} (total NaNs)")
    # Optional: Log NaN counts per column if high
    # nan_counts_per_col = X_train.isna().sum()
    # logger.info(f"X_train NaNs per column (top 5 with NaNs): \n{nan_counts_per_col[nan_counts_per_col > 0].sort_values(ascending=False).head()}")

    # Convert boolean columns to int
    for col in X_train.columns:
        if X_train[col].dtype == 'bool':
            logger.info(f"Converting boolean column {col} to int.")
            X_train[col] = X_train[col].astype(int)
            if col in X_test.columns:
                X_test[col] = X_test[col].astype(int)

    X_train = X_train.fillna(-999.0)
    if not X_test.empty:
        X_test = X_test.fillna(-999.0)

    logger.info(f"X_train NaNs after fill: {X_train.isna().sum().sum()}")
    if (X_train == -999.0).all().all():
        logger.error("CRITICAL: All values in X_train became -999.0 after fillna. All features might have been NaN. Check input data and feature engineering.")
    # --- END Feature Matrix (X_train) Diagnostics ---

    logger.info(f"Final X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}")
    if X_train.shape[1] == 0:
        logger.error("CRITICAL: No features remaining in X_train. Cannot train. Check cols_to_drop and feature selection logic.")
        exit()

    # Create an explicit train/validation split for final validation metrics
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, shuffle=True # Ensure shuffle if data might be ordered
    )
    logger.info(f"Shapes for final validation: X_train_final: {X_train_final.shape}, X_val_final: {X_val_final.shape}")


    # Step 5: Train LightGBM model
    logger.info("Step 5: Training LightGBM model...")
    kf = KFold(n_splits=N_SPLITS_LGBM, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(X_train.shape[0])

    if X_test.empty:
        logger.warning("X_test is empty, test predictions will not be generated meaningfully.")
        test_preds_lgbm = np.array([]) # Empty array
    else:
        test_preds_lgbm = np.zeros(X_test.shape[0])


    # Load LightGBM parameters from configuration
    lgb_params = config['lgbm_params'].copy()
    logger.info(f"LightGBM Parameters: {lgb_params}")


    # Record training start time
    lgbm_training_start = time.time()
    logger.info(f"Starting LightGBM KFold training with {N_SPLITS_LGBM} folds")

    feature_importances = pd.DataFrame() # To store feature importances

    kfold_splits = list(kf.split(X_train, y_train))
    for fold, (train_idx, val_idx) in enumerate(
            tqdm(kfold_splits, total=N_SPLITS_LGBM, desc="LGBM KFold Training")):

        fold_start_time = time.time()
        logger.info(f"--- Training fold {fold + 1}/{N_SPLITS_LGBM} ---")

        lgb_params['seed'] = RANDOM_STATE + fold # Vary seed per fold
        X_tr_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        logger.info(f"Fold {fold + 1} train shape: {X_tr_fold.shape}, validation shape: {X_val_fold.shape}")
        logger.info(f"Fold {fold + 1} y_tr_fold unique values (first 5): {np.unique(y_tr_fold)[:5]}, mean: {y_tr_fold.mean():.2f}")
        logger.info(f"Fold {fold + 1} y_val_fold unique values (first 5): {np.unique(y_val_fold)[:5]}, mean: {y_val_fold.mean():.2f}")


        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr_fold, y_tr_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='rmse', # Default for LGBMRegressor is 'rmse' or 'l2'
            callbacks=[
                lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=100), # More verbose early stopping
                lgb.log_evaluation(period=100) # Log eval results periodically
            ]
        )

        model_fold_path = os.path.join(LGBM_MODEL_SAVE_DIR,
                                       f"lgbm_fold_{fold + 1}_best_iter_{model.best_iteration_}.txt")
        model.booster_.save_model(model_fold_path)
        logger.info(f"Fold {fold + 1} LGBM model saved: {model_fold_path} (Best iter: {model.best_iteration_}, Score: {model.best_score_['valid_0']['rmse']:.4f})")

        # Make predictions
        oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration_)
        if not X_test.empty:
            test_preds_lgbm += model.predict(X_test, num_iteration=model.best_iteration_) / N_SPLITS_LGBM

        # Feature importances
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_train.columns
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = fold + 1
        feature_importances = pd.concat([feature_importances, fold_importance_df], axis=0)


        # Calculate fold metrics
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, oof_preds[val_idx]))
        fold_time = time.time() - fold_start_time

        logger.info(f"Fold {fold + 1} completed in {fold_time:.2f} seconds. Val RMSE: {fold_rmse:.4f}")
        record_metric(f"fold_{fold + 1}_rmse", fold_rmse, "fold_performance")
        record_metric(f"fold_{fold + 1}_time", fold_time, "performance")
        record_metric(f"fold_{fold + 1}_best_iteration", model.best_iteration_ if model.best_iteration_ else -1, "model_stats")


    # Log overall feature importances
    mean_importances = feature_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)
    logger.info(f"Mean Feature Importances (Top 20):\n{mean_importances.head(20)}")


    # Record total training time
    lgbm_training_time = time.time() - lgbm_training_start
    logger.info(f"LightGBM KFold training completed in {lgbm_training_time:.2f} seconds")
    record_metric("lgbm_training_time", lgbm_training_time, "performance")

    # Calculate overall out-of-fold RMSE
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    logger.info(f"Overall OOF RMSE (LGBM with Stacking AE): {oof_rmse:.4f}")
    record_metric("oof_rmse", oof_rmse, "model_performance")

    # Log OOF predictions distribution
    logger.info(f"OOF Predictions: Min={np.min(oof_preds):.2f}, Max={np.max(oof_preds):.2f}, Mean={np.mean(oof_preds):.2f}, Std={np.std(oof_preds):.2f}")


    # Step 6: Evaluate model on validation set (if y_train was not constant)
    logger.info("Step 6: Evaluating model on explicit validation set...")

    logger.info("Training model on explicit validation split (X_train_final, y_train_final) for final metrics...")
    val_model_params = lgb_params.copy()
    val_model_params['seed'] = RANDOM_STATE # Consistent seed for this model
    val_model = lgb.LGBMRegressor(**val_model_params)
    val_model.fit(
        X_train_final, y_train_final,
        eval_set=[(X_val_final, y_val_final)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=100),
            lgb.log_evaluation(period=100)
        ]
    )

    val_preds = val_model.predict(X_val_final, num_iteration=val_model.best_iteration_)
    val_rmse = np.sqrt(mean_squared_error(y_val_final, val_preds))
    logger.info(f"Validation RMSE on explicit 80/20 split: {val_rmse:.4f} (Best iter: {val_model.best_iteration_}, Score: {val_model.best_score_['valid_0']['rmse']:.4f})")
    record_metric("validation_rmse", val_rmse, "model_performance")

    val_mae = mean_absolute_error(y_val_final, val_preds)
    val_r2 = r2_score(y_val_final, val_preds)
    logger.info(f"Validation MAE: {val_mae:.4f}")
    logger.info(f"Validation R² Score: {val_r2:.4f}")
    logger.info(f"Validation Predictions: Min={np.min(val_preds):.2f}, Max={np.max(val_preds):.2f}, Mean={np.mean(val_preds):.2f}, Std={np.std(val_preds):.2f}")


    record_metric("validation_mae", val_mae, "model_performance")
    record_metric("validation_r2", val_r2, "model_performance")

    # Calculate RMSE by rating range
    logger.info("RMSE by rating range on explicit validation set:")
    rating_ranges = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000, float('inf'))]
    range_results = []

    for low, high in rating_ranges:
        mask = (y_val_final >= low) & (y_val_final < high)
        if np.sum(mask) > 0:
            range_rmse = np.sqrt(mean_squared_error(y_val_final[mask], val_preds[mask]))
            range_count = np.sum(mask)
            range_name = f"{low}-{high if high != float('inf') else 'inf'}"
            logger.info(f"  {range_name}: {range_rmse:.4f} (n={range_count})")
            record_metric(f"rmse_{range_name}", range_rmse, "range_performance")
            record_metric(f"count_{range_name}", range_count, "range_performance")
            range_results.append({'range': range_name, 'rmse': range_rmse, 'count': range_count})

    # Step 7: Generate submission file
    logger.info("Step 7: Generating submission file...")
    if X_test.empty or test_preds_lgbm.size == 0:
        logger.warning("Test data or predictions are empty. Submission file will be empty or not generated.")
        submission_df = pd.DataFrame({'PuzzleId': [], 'Rating': []})
    else:
        final_predictions = np.round(test_preds_lgbm).astype(int)
        submission_df = pd.DataFrame({'PuzzleId': test_puzzle_ids, 'Rating': final_predictions})

    submission_file_path = SUBMISSION_FILE
    if not submission_file_path.lower().endswith('.txt'):
        submission_file_path = os.path.splitext(submission_file_path)[0] + ".txt"

    try:
        with open(submission_file_path, 'w') as f:
            if not submission_df.empty:
                for pred_rating in submission_df['Rating']:
                    f.write(f"{pred_rating}\n")
                logger.info(f"Submission file '{submission_file_path}' created with {len(submission_df)} predictions.")
                logger.info(f"First 5 predictions:\n{submission_df.head().to_string()}")
            else:
                f.write("") # Create an empty file
                logger.info(f"Submission file '{submission_file_path}' created (empty).")

    except Exception as e:
        logger.error(f"Error writing submission file: {e}")


    # Calculate total execution time
    total_execution_time = time.time() - overall_start_time
    logger.info(f"Total execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")

    # Record final metrics
    record_metric("total_execution_time", total_execution_time, "performance")
    record_metric("final_oof_rmse", oof_rmse, "model_performance")
    record_metric("final_validation_rmse", val_rmse, "model_performance")
    record_metric("final_validation_mae", val_mae, "model_performance")
    record_metric("final_validation_r2", val_r2, "model_performance")

    # Generate performance dashboard
    dashboard_config = config.get('dashboards', {})
    if dashboard_config.get('enabled', True):
        try:
            logger.info("Generating performance dashboard...")
            output_dir = dashboard_config.get('output_dir', 'dashboards')
            dashboard_name = f"lgbm_stack_ae_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dashboard_path = create_performance_dashboard(output_dir=output_dir, dashboard_name=dashboard_name)
            logger.info(f"Performance dashboard created at: {dashboard_path}")
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {str(e)}")

    logger.info("Training script finished.")
    if y_train_nona.nunique() == 1:
         logger.critical("SCRIPT FINISHED, BUT THE TARGET VARIABLE WAS CONSTANT. RESULTS ARE LIKELY INVALID. PLEASE CHECK THE INPUT DATA 'final_merged_features.csv'.")