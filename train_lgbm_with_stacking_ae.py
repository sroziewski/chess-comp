"""
Training script for LightGBM model with autoencoder features from train_stacking.py.

This script loads features from final_dataset.csv, uses the autoencoder trained by train_stacking.py,
and applies only the LightGBM model from train_lgbm_pt_ae_no_engine_direct.py for prediction.
"""

import os
# Set the boost_compute directory to /raid/sroziewski/.boost_compute
os.environ['BOOST_COMPUTE_DEFAULT_TEMP_PATH'] = '/raid/sroziewski/.boost_compute'

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    ProbAutoencoder, extract_embeddings_from_autoencoder,
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
    log_file_name=logging_config.get('log_file_name', f"lgbm_with_stacking_ae_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
PROB_SEQ_LENGTH = config['autoencoder']['sequence_length']
EMBEDDING_DIM_PROB = config['autoencoder']['embedding_dim']

# --- Main Script ---
if __name__ == '__main__':
    # Record overall start time
    overall_start_time = time.time()

    logger.info(f"Using PyTorch on device: {DEVICE} with {GPUS_TO_USE} GPU(s) for DataParallel if > 1.")

    # Record GPU configuration
    record_metric("gpu_count", GPUS_TO_USE, "hardware")
    record_metric("device_type", "cuda" if DEVICE.type == "cuda" else "cpu", "hardware")

    # Step 1: Load data from final_dataset.csv
    logger.info("Step 1: Loading data from final_dataset.csv...")
    data_load_start = time.time()

    # Load final_dataset.csv which already contains all features
    logger.info("Loading final_dataset.csv which already contains all features...")
    combined_df = pd.read_csv('final_dataset.csv')

    # Split into train and test sets
    train_df_orig = combined_df[combined_df['is_train'] == 1].copy()
    test_df_orig = combined_df[combined_df['is_train'] == 0].copy()
    test_puzzle_ids = test_df_orig['PuzzleId']

    if 'Rating' not in test_df_orig.columns:
        test_df_orig['Rating'] = np.nan

    data_load_time = time.time() - data_load_start
    logger.info(f"Data loaded in {data_load_time:.2f} seconds")
    logger.info(f"Training data shape: {train_df_orig.shape}, Test data shape: {test_df_orig.shape}")

    # Record data metrics
    record_metric("data_load_time", data_load_time, "performance")
    record_metric("train_rows", train_df_orig.shape[0], "data_stats")
    record_metric("train_columns", train_df_orig.shape[1], "data_stats")
    record_metric("test_rows", test_df_orig.shape[0], "data_stats")
    record_metric("test_columns", test_df_orig.shape[1], "data_stats")

    # Step 2: Load autoencoders trained by train_stacking.py
    logger.info("Step 2: Loading autoencoders trained by train_stacking.py...")
    
    prob_cols_all = [col for col in combined_df.columns if 'success_prob_' in col]
    rapid_prob_cols_all = sorted([col for col in prob_cols_all if 'rapid' in col], key=lambda x: int(x.split('_')[-1]))
    blitz_prob_cols_all = sorted([col for col in prob_cols_all if 'blitz' in col], key=lambda x: int(x.split('_')[-1]))

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
            current_ae_model = ProbAutoencoder().to(DEVICE)
            current_ae_model.load_state_dict(torch.load(save_path, map_location=DEVICE))
            current_ae_model.eval()
            logger.info(f"{ae_type} model loaded successfully.")
            
            if model_var_name == "trained_rapid_ae":
                trained_rapid_ae = current_ae_model
            elif model_var_name == "trained_blitz_ae":
                trained_blitz_ae = current_ae_model
        else:
            logger.warning(f"Pre-trained {ae_type} model not found at {save_path}. Autoencoder features will not be available.")

    # Step 3: Extract autoencoder features
    logger.info("Step 3: Extracting autoencoder features...")
    feature_engineering_start = time.time()

    # Success probability features with autoencoder
    if trained_rapid_ae or trained_blitz_ae:
        logger.info("Extracting success probability features with autoencoder...")
        success_prob_df_subset = combined_df[prob_cols_all] if prob_cols_all else pd.DataFrame()
        if not success_prob_df_subset.empty:
            success_prob_features_df = get_success_prob_features_with_trained_ae(
                success_prob_df_subset, trained_rapid_ae, trained_blitz_ae, DEVICE
            )
            combined_df = pd.concat([combined_df, success_prob_features_df], axis=1)
            del success_prob_features_df
            gc.collect()
        else:
            logger.warning("No success_prob columns found to pass to AE feature extractor.")
    else:
        logger.warning("No AEs available. Using only basic aggregates for success_prob.")
        if 'prob_all_mean' not in combined_df.columns and prob_cols_all:
            combined_df['prob_all_mean'] = combined_df[prob_cols_all].mean(axis=1)
            combined_df['prob_all_std'] = combined_df[prob_cols_all].std(axis=1)

    feature_engineering_time = time.time() - feature_engineering_start
    logger.info(f"Feature engineering completed in {feature_engineering_time:.2f} seconds")
    record_metric("feature_engineering_time", feature_engineering_time, "performance")

    # Step 4: Prepare data for LightGBM model
    logger.info("Step 4: Preparing data for LightGBM model...")
    target_col = 'Rating'
    
    # Columns to exclude from features
    cols_to_drop = ['PuzzleId', 'FEN', 'Moves', 'Rating', 'is_train', 'Themes', 'GameUrl', 'OpeningTags']
    
    # Handle log-transformed columns
    if 'Popularity_log' in combined_df.columns:
        cols_to_drop.append('Popularity')
    if 'NbPlays_log' in combined_df.columns:
        cols_to_drop.append('NbPlays')
    
    # Use all numeric columns as features except those in cols_to_drop
    feature_columns = []
    for col in combined_df.columns:
        if col not in cols_to_drop and pd.api.types.is_numeric_dtype(combined_df[col]):
            feature_columns.append(col)
        elif col not in cols_to_drop and not pd.api.types.is_numeric_dtype(combined_df[col]):
            logger.warning(f"Dropping non-numeric feature: {col} (dtype: {combined_df[col].dtype})")
    
    logger.info(f"Using {len(feature_columns)} features for LightGBM model")
    train_processed_df = combined_df[combined_df['is_train'] == 1].copy()
    test_processed_df = combined_df[combined_df['is_train'] == 0].copy()
    X_train = train_processed_df[feature_columns].astype(np.float32)
    y_train = train_processed_df[target_col].astype(np.float32)
    X_test = test_processed_df[feature_columns].astype(np.float32)
    X_train = X_train.fillna(-999.0)
    X_test = X_test.fillna(-999.0)
    logger.info(f"Training LightGBM with {X_train.shape[1]} features.")

    # Create an explicit train/validation split for final validation metrics
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    # Step 5: Train LightGBM model
    logger.info("Step 5: Training LightGBM model...")
    kf = KFold(n_splits=N_SPLITS_LGBM, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(X_train.shape[0])
    test_preds_lgbm = np.zeros(X_test.shape[0])

    # Load LightGBM parameters from configuration
    lgb_params = config['lgbm_params'].copy()

    # Record training start time
    lgbm_training_start = time.time()
    logger.info(f"Starting LightGBM KFold training with {N_SPLITS_LGBM} folds")

    # Train LightGBM model with K-fold cross-validation
    kfold_splits = list(kf.split(X_train, y_train))
    for fold, (train_idx, val_idx) in enumerate(
            tqdm(kfold_splits, total=N_SPLITS_LGBM, desc="LGBM KFold Training")):

        fold_start_time = time.time()
        logger.info(f"Training fold {fold + 1}/{N_SPLITS_LGBM}")

        lgb_params['seed'] = RANDOM_STATE + fold
        X_tr_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        logger.info(f"Fold {fold + 1} train shape: {X_tr_fold.shape}, validation shape: {X_val_fold.shape}")

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr_fold, y_tr_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False)]
        )

        model_fold_path = os.path.join(LGBM_MODEL_SAVE_DIR,
                                       f"lgbm_fold_{fold + 1}_best_iter_{model.best_iteration_}.txt")
        model.booster_.save_model(model_fold_path)
        logger.info(f"Fold {fold + 1} LGBM model saved: {model_fold_path} (Best iter: {model.best_iteration_})")

        # Make predictions
        oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration_)
        test_preds_lgbm += model.predict(X_test, num_iteration=model.best_iteration_) / N_SPLITS_LGBM

        # Calculate fold metrics
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, oof_preds[val_idx]))
        fold_time = time.time() - fold_start_time

        # Record fold metrics
        logger.info(f"Fold {fold + 1} completed in {fold_time:.2f} seconds. RMSE: {fold_rmse:.4f}")
        record_metric(f"fold_{fold + 1}_rmse", fold_rmse, "fold_performance")
        record_metric(f"fold_{fold + 1}_time", fold_time, "performance")
        record_metric(f"fold_{fold + 1}_best_iteration", model.best_iteration_, "model_stats")

    # Record total training time
    lgbm_training_time = time.time() - lgbm_training_start
    logger.info(f"LightGBM KFold training completed in {lgbm_training_time:.2f} seconds")
    record_metric("lgbm_training_time", lgbm_training_time, "performance")

    # Calculate overall out-of-fold RMSE
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    logger.info(f"Overall OOF RMSE (LGBM with Stacking AE): {oof_rmse:.4f}")
    record_metric("oof_rmse", oof_rmse, "model_performance")

    # Step 6: Evaluate model on validation set
    logger.info("Step 6: Evaluating model on validation set...")
    
    # Train a model on the explicit train/validation split for validation metrics
    logger.info("Training model on explicit validation split for final metrics...")
    val_model = lgb.LGBMRegressor(**lgb_params)
    val_model.fit(
        X_train_final, y_train_final,
        eval_set=[(X_val_final, y_val_final)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False)]
    )

    val_preds = val_model.predict(X_val_final, num_iteration=val_model.best_iteration_)
    val_rmse = np.sqrt(mean_squared_error(y_val_final, val_preds))
    logger.info(f"Validation RMSE on explicit 80/20 split: {val_rmse:.4f}")
    record_metric("validation_rmse", val_rmse, "model_performance")

    # Additional detailed validation metrics
    val_mae = mean_absolute_error(y_val_final, val_preds)
    val_r2 = r2_score(y_val_final, val_preds)
    logger.info(f"Validation MAE: {val_mae:.4f}")
    logger.info(f"Validation R² Score: {val_r2:.4f}")

    # Record validation metrics
    record_metric("validation_mae", val_mae, "model_performance")
    record_metric("validation_r2", val_r2, "model_performance")

    # Calculate RMSE by rating range
    logger.info("RMSE by rating range:")
    rating_ranges = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000, float('inf'))]
    range_results = []

    for low, high in rating_ranges:
        mask = (y_val_final >= low) & (y_val_final < high)
        if np.sum(mask) > 0:
            range_rmse = np.sqrt(mean_squared_error(y_val_final[mask], val_preds[mask]))
            range_count = np.sum(mask)
            range_name = f"{low}-{high if high != float('inf') else 'inf'}"

            logger.info(f"  {range_name}: {range_rmse:.4f} (n={range_count})")

            # Record metrics for each range
            record_metric(f"rmse_{range_name}", range_rmse, "range_performance")
            record_metric(f"count_{range_name}", range_count, "range_performance")

            range_results.append({
                'range': range_name,
                'rmse': range_rmse,
                'count': range_count
            })

    # Step 7: Generate submission file
    logger.info("Step 7: Generating submission file...")
    final_predictions = np.round(test_preds_lgbm).astype(int)
    submission_df = pd.DataFrame({'PuzzleId': test_puzzle_ids, 'Rating': final_predictions})
    submission_file_path = SUBMISSION_FILE

    if not submission_file_path.lower().endswith('.txt'):
        submission_file_path = os.path.splitext(submission_file_path)[0] + ".txt"

    with open(submission_file_path, 'w') as f:
        for pred_rating in submission_df['Rating']:
            f.write(f"{pred_rating}\n")

    logger.info(f"Submission file '{submission_file_path}' created.")
    logger.info(f"First 5 predictions:\n{submission_df.head().to_string()}")

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
            dashboard_name = f"lgbm_with_stacking_ae_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dashboard_path = create_performance_dashboard(output_dir=output_dir, dashboard_name=dashboard_name)
            logger.info(f"Performance dashboard created at: {dashboard_path}")
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {str(e)}")

    logger.info("Training completed successfully.")