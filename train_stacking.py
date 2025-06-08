"""
Training script for model stacking approach to chess puzzle rating prediction.

This script demonstrates how to use the stacking model implementation to combine
multiple base models (LightGBM, XGBoost, Neural Networks) for improved prediction accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time
import datetime
import logging
import gc

# --- Configuration Import ---
from chess_puzzle_rating.utils.config import get_config

# --- Progress Tracking Import ---
from chess_puzzle_rating.utils.progress import (
    setup_logging, get_logger, log_time, ProgressTracker, 
    track_progress, record_metric, create_performance_dashboard
)

# --- Stacking Model Import ---
from chess_puzzle_rating.models.stacking import (
    LightGBMModel, XGBoostModel, NeuralNetworkModel, 
    StackingModel, RangeSpecificStackingModel, create_stacking_model
)

# --- Feature Engineering Functions ---
from train_lgbm_pt_ae_no_engine_direct import (
    get_extended_fen_features, get_moves_features, 
    process_text_tags, get_success_prob_features_with_trained_ae,
    ProbAutoencoder, extract_embeddings_from_autoencoder
)

# --- Initialize Configuration ---
config = get_config()
logging_config = config.get('logging', {})
logger = setup_logging(
    log_dir=logging_config.get('log_dir', 'logs'),
    log_level=getattr(logging, logging_config.get('log_level', 'INFO')),
    log_to_console=logging_config.get('log_to_console', True),
    log_to_file=logging_config.get('log_to_file', True),
    log_file_name=logging_config.get('log_file_name', f"stacking_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
)

# --- Data Paths ---
TRAIN_FILE = config['data_paths']['train_file']
TEST_FILE = config['data_paths']['test_file']
SUBMISSION_FILE = config['data_paths']['submission_file']

logger.info("Starting model stacking training")
logger.info(f"Training data: {TRAIN_FILE}")
logger.info(f"Test data: {TEST_FILE}")
logger.info(f"Submission file: {SUBMISSION_FILE}")

# --- Training Parameters ---
RANDOM_STATE = config['training']['random_state']
N_SPLITS_STACKING = config.get('stacking', {}).get('n_splits', 5)
OPTIMIZE_META_LEARNER = config.get('stacking', {}).get('optimize_meta_learner', True)
META_LEARNER_TYPE = config.get('stacking', {}).get('meta_learner_type', 'lightgbm')
USE_FEATURES_IN_META = config.get('stacking', {}).get('use_features_in_meta', True)

# --- Rating Range-Specific Models Configuration ---
RATING_RANGES_ENABLED = config.get('stacking', {}).get('rating_ranges', {}).get('enabled', False)
RATING_RANGES = config.get('stacking', {}).get('rating_ranges', {}).get('ranges', [])
RANGE_OVERLAP = config.get('stacking', {}).get('rating_ranges', {}).get('range_overlap', 100)
ENSEMBLE_METHOD = config.get('stacking', {}).get('rating_ranges', {}).get('ensemble_method', 'weighted_average')

# --- Model Save Directories ---
MODEL_SAVE_DIR = config['stacking']['model_save_dir']
AE_MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "autoencoders")
STACKING_MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "stacking")

os.makedirs(AE_MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(STACKING_MODEL_SAVE_DIR, exist_ok=True)

# --- PyTorch Autoencoder Training Config ---
AE_EPOCHS = config['autoencoder']['epochs']
AE_BATCH_SIZE_PER_GPU = config['autoencoder']['batch_size_per_gpu']
AE_LEARNING_RATE = config['autoencoder']['learning_rate']
AE_VALID_SPLIT = config['autoencoder']['valid_split']
AE_EARLY_STOPPING_PATIENCE = config['autoencoder']['early_stopping_patience']
FORCE_RETRAIN_AES = config['autoencoder']['force_retrain']

# --- GPU Configuration ---
import torch
TARGET_NUM_GPUS = config['gpu']['target_num_gpus']
if torch.cuda.is_available():
    AVAILABLE_GPUS = torch.cuda.device_count()
    print(f"Found {AVAILABLE_GPUS} CUDA GPUs.")
    GPUS_TO_USE = min(AVAILABLE_GPUS, TARGET_NUM_GPUS)
    if GPUS_TO_USE > 0:
        DEVICE_IDS = list(range(GPUS_TO_USE))
        PRIMARY_DEVICE_STR = f'cuda:{DEVICE_IDS[0]}'
        print(f"Will use {GPUS_TO_USE} GPUs: {DEVICE_IDS} for PyTorch. Primary: {PRIMARY_DEVICE_STR}")
        AE_TOTAL_BATCH_SIZE = AE_BATCH_SIZE_PER_GPU * GPUS_TO_USE
        print(f"PyTorch Autoencoder Total Batch Size for Training: {AE_TOTAL_BATCH_SIZE} (across {GPUS_TO_USE} GPUs)")
    else:
        DEVICE_IDS = None
        PRIMARY_DEVICE_STR = 'cpu'
        print("CUDA available but 0 GPUs selected. Using CPU.")
        AE_TOTAL_BATCH_SIZE = AE_BATCH_SIZE_PER_GPU
else:
    AVAILABLE_GPUS = 0
    GPUS_TO_USE = 0
    DEVICE_IDS = None
    PRIMARY_DEVICE_STR = 'cpu'
    print("No CUDA GPUs. Using CPU for PyTorch.")
    AE_TOTAL_BATCH_SIZE = AE_BATCH_SIZE_PER_GPU
DEVICE = torch.device(PRIMARY_DEVICE_STR)

# --- PyTorch Autoencoder Sequence Length and Embedding Dimension ---
PROB_SEQ_LENGTH = config['autoencoder']['sequence_length']
EMBEDDING_DIM_PROB = config['autoencoder']['embedding_dim']

if __name__ == '__main__':
    # Record overall start time
    overall_start_time = time.time()

    logger.info(f"Using PyTorch on device: {DEVICE} with {GPUS_TO_USE} GPU(s) for DataParallel if > 1.")

    # Record GPU configuration
    record_metric("gpu_count", GPUS_TO_USE, "hardware")
    record_metric("device_type", "cuda" if DEVICE.type == "cuda" else "cpu", "hardware")

    # Step 1: Load data
    logger.info("Step 1: Loading data...")
    data_load_start = time.time()

    train_df_orig = pd.read_csv(TRAIN_FILE)
    test_df_orig = pd.read_csv(TEST_FILE)
    test_puzzle_ids = test_df_orig['PuzzleId']
    train_df_orig['is_train'] = 1
    test_df_orig['is_train'] = 0

    if 'Rating' not in test_df_orig.columns:
        test_df_orig['Rating'] = np.nan

    combined_df = pd.concat([train_df_orig, test_df_orig], ignore_index=True, sort=False)

    data_load_time = time.time() - data_load_start
    logger.info(f"Data loaded in {data_load_time:.2f} seconds")
    logger.info(f"Training data shape: {train_df_orig.shape}, Test data shape: {test_df_orig.shape}")

    # Record data metrics
    record_metric("data_load_time", data_load_time, "performance")
    record_metric("train_rows", train_df_orig.shape[0], "data_stats")
    record_metric("train_columns", train_df_orig.shape[1], "data_stats")
    record_metric("test_rows", test_df_orig.shape[0], "data_stats")
    record_metric("test_columns", test_df_orig.shape[1], "data_stats")

    # Step 2: Load or train autoencoders for success probability features
    prob_cols_all = [col for col in combined_df.columns if 'success_prob_' in col]
    rapid_prob_cols_all = sorted([col for col in prob_cols_all if 'rapid' in col], key=lambda x: int(x.split('_')[-1]))
    blitz_prob_cols_all = sorted([col for col in prob_cols_all if 'blitz' in col], key=lambda x: int(x.split('_')[-1]))

    trained_rapid_ae, trained_blitz_ae = None, None
    rapid_ae_save_path = os.path.join(AE_MODEL_SAVE_DIR, "rapid_ae_best.pth")
    blitz_ae_save_path = os.path.join(AE_MODEL_SAVE_DIR, "blitz_ae_best.pth")

    # Load or train autoencoders (same as in train_lgbm_pt_ae_no_engine_direct.py)
    for ae_type, prob_cols_subset, save_path, model_var_name in [
        ("RapidProbAE", rapid_prob_cols_all, rapid_ae_save_path, "trained_rapid_ae"),
        ("BlitzProbAE", blitz_prob_cols_all, blitz_ae_save_path, "trained_blitz_ae")
    ]:
        current_ae_model = None
        if os.path.exists(save_path) and not FORCE_RETRAIN_AES:
            print(f"Loading pre-trained {ae_type} model from {save_path}...")
            current_ae_model = ProbAutoencoder().to(DEVICE)
            current_ae_model.load_state_dict(torch.load(save_path, map_location=DEVICE))
            current_ae_model.eval()
            print(f"{ae_type} model loaded successfully.")
        elif prob_cols_subset and len(prob_cols_subset) == PROB_SEQ_LENGTH:
            # Ensure we use .dropna() on the correct subset of columns *before* .values
            seq_data_all_for_current_ae = combined_df[prob_cols_subset].dropna().values.astype(np.float32)
            if seq_data_all_for_current_ae.shape[0] > AE_TOTAL_BATCH_SIZE:
                X_tr, X_val = train_test_split(seq_data_all_for_current_ae, test_size=AE_VALID_SPLIT,
                                               random_state=RANDOM_STATE)
                loader_tr = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.from_numpy(X_tr)), 
                    batch_size=AE_TOTAL_BATCH_SIZE,
                    shuffle=True, num_workers=2, pin_memory=(DEVICE.type == 'cuda')
                )
                loader_val = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.from_numpy(X_val)), 
                    batch_size=AE_TOTAL_BATCH_SIZE,
                    shuffle=False, num_workers=2, pin_memory=(DEVICE.type == 'cuda')
                )
                base_ae_model = ProbAutoencoder().to(DEVICE)
                # Import train_autoencoder function from train_lgbm_pt_ae_no_engine_direct
                from train_lgbm_pt_ae_no_engine_direct import train_autoencoder
                current_ae_model = train_autoencoder(base_ae_model, loader_tr, loader_val, AE_EPOCHS, AE_LEARNING_RATE,
                                                     AE_EARLY_STOPPING_PATIENCE, save_path, ae_type)
            else:
                print(
                    f"Not enough {ae_type.split('ProbAE')[0]} data for AE training (found {seq_data_all_for_current_ae.shape[0]} sequences).")
        else:
            print(f"{ae_type.split('ProbAE')[0]} prob columns not found/wrong length for AE.")

        if model_var_name == "trained_rapid_ae":
            trained_rapid_ae = current_ae_model
        elif model_var_name == "trained_blitz_ae":
            trained_blitz_ae = current_ae_model

    # Step 3: Feature Engineering
    logger.info("Step 3: Feature Engineering...")
    feature_engineering_start = time.time()

    # FEN features
    logger.info("Extracting FEN features...")
    fen_features_df = pd.DataFrame(combined_df['FEN'].progress_apply(get_extended_fen_features).tolist(),
                                   index=combined_df.index)
    combined_df = pd.concat([combined_df, fen_features_df], axis=1)
    del fen_features_df
    gc.collect()

    # Moves features
    logger.info("Extracting moves features...")
    moves_features_df = pd.DataFrame(combined_df['Moves'].progress_apply(get_moves_features).tolist(),
                                     index=combined_df.index)
    combined_df = pd.concat([combined_df, moves_features_df], axis=1)
    del moves_features_df
    gc.collect()

    # Text vectorization
    themes_min_df = config['feature_engineering']['text_vectorization']['themes_min_df']
    openings_min_df = config['feature_engineering']['text_vectorization']['openings_min_df']

    logger.info("Processing theme tags...")
    themes_df = process_text_tags(combined_df['Themes'], prefix='theme', min_df=themes_min_df)
    combined_df = pd.concat([combined_df, themes_df], axis=1)
    del themes_df
    gc.collect()

    logger.info("Processing opening tags...")
    openings_df = process_text_tags(combined_df['OpeningTags'], prefix='opening', min_df=openings_min_df)
    combined_df = pd.concat([combined_df, openings_df], axis=1)
    del openings_df
    gc.collect()

    # Success probability features with autoencoder
    if trained_rapid_ae or trained_blitz_ae:
        logger.info("Extracting success probability features with autoencoder...")
        success_prob_df_subset = combined_df[
            prob_cols_all] if prob_cols_all else pd.DataFrame()
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

    # Log transform for popularity and plays
    for col in ['Popularity', 'NbPlays']:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)
            min_val = combined_df.loc[combined_df['is_train'] == 1, col].min()
            if min_val <= 0:
                combined_df[f'{col}_log'] = np.log1p(combined_df[col] - (min_val if min_val < 0 else 0))
            else:
                combined_df[f'{col}_log'] = np.log(combined_df[col])

    feature_engineering_time = time.time() - feature_engineering_start
    logger.info(f"Feature engineering completed in {feature_engineering_time:.2f} seconds")
    record_metric("feature_engineering_time", feature_engineering_time, "performance")

    # Step 4: Prepare data for model training
    logger.info("Step 4: Preparing data for model training...")
    target_col = 'Rating'
    original_cols_to_drop = ['PuzzleId', 'FEN', 'Moves', 'Themes', 'GameUrl', 'OpeningTags']

    if 'Popularity_log' in combined_df.columns:
        original_cols_to_drop.append('Popularity')
    if 'NbPlays_log' in combined_df.columns:
        original_cols_to_drop.append('NbPlays')

    feature_columns = [col for col in combined_df.columns if
                       col not in [target_col, 'is_train'] + original_cols_to_drop]
    numeric_feature_columns = []

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(combined_df[col]):
            numeric_feature_columns.append(col)
        else:
            logger.warning(f"Dropping non-numeric feature: {col} (dtype: {combined_df[col].dtype})")

    feature_columns = numeric_feature_columns
    train_processed_df = combined_df[combined_df['is_train'] == 1].copy()
    test_processed_df = combined_df[combined_df['is_train'] == 0].copy()
    X_train = train_processed_df[feature_columns].astype(np.float32)
    y_train = train_processed_df[target_col].astype(np.float32)
    X_test = test_processed_df[feature_columns].astype(np.float32)
    X_train = X_train.fillna(-999.0)
    X_test = X_test.fillna(-999.0)
    logger.info(f"Training with {X_train.shape[1]} features.")

    # Create an explicit train/validation split for final validation metrics
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    # Step 5: Train stacking model
    logger.info("Step 5: Training stacking model...")
    stacking_start_time = time.time()

    # Get base model parameters from config
    base_model_params = config.get('stacking', {}).get('base_models', {})

    # Create base models with parameters from config
    base_models = []

    # LightGBM base model
    lgbm_params = base_model_params.get('lightgbm', {})
    lgbm_params['random_state'] = RANDOM_STATE
    base_models.append(
        LightGBMModel(
            name="lightgbm_base",
            model_params={
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                **lgbm_params
            }
        )
    )

    # XGBoost base model
    xgb_params = base_model_params.get('xgboost', {})
    xgb_params['random_state'] = RANDOM_STATE
    base_models.append(
        XGBoostModel(
            name="xgboost_base",
            model_params={
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                **xgb_params
            }
        )
    )

    # Neural Network base model
    nn_params = base_model_params.get('neural_network', {})
    base_models.append(
        NeuralNetworkModel(
            name="neural_network_base",
            model_params=nn_params
        )
    )

    logger.info(f"Created {len(base_models)} base models with parameters from config")

    # Create and train model based on configuration
    if RATING_RANGES_ENABLED and RATING_RANGES:
        logger.info(f"Using rating range-specific models with {len(RATING_RANGES)} ranges")
        logger.info(f"Rating ranges: {RATING_RANGES}")
        logger.info(f"Range overlap: {RANGE_OVERLAP}")
        logger.info(f"Ensemble method: {ENSEMBLE_METHOD}")

        # Create range-specific stacking model
        stacking_model = RangeSpecificStackingModel(
            rating_ranges=RATING_RANGES,
            range_overlap=RANGE_OVERLAP,
            ensemble_method=ENSEMBLE_METHOD,
            n_splits=N_SPLITS_STACKING,
            random_state=RANDOM_STATE,
            optimize_meta=OPTIMIZE_META_LEARNER,
            meta_learner_type=META_LEARNER_TYPE,
            use_features_in_meta=USE_FEATURES_IN_META
        )

        # Create range-specific model directory
        RANGE_SPECIFIC_MODEL_DIR = os.path.join(STACKING_MODEL_SAVE_DIR, "range_specific")
        os.makedirs(RANGE_SPECIFIC_MODEL_DIR, exist_ok=True)

        # Train the range-specific stacking model
        stacking_model.fit(
            X=X_train_final,
            y=y_train_final,
            X_val=X_val_final,
            y_val=y_val_final,
            model_dir=RANGE_SPECIFIC_MODEL_DIR
        )

        # Save the range-specific stacking model configuration
        stacking_config_path = os.path.join(RANGE_SPECIFIC_MODEL_DIR, "range_specific_model_config.pkl")
        stacking_model.save(stacking_config_path)
        logger.info(f"Range-specific stacking model configuration saved to {stacking_config_path}")

    else:
        logger.info("Using standard stacking model (no rating ranges)")

        # Create standard stacking model
        stacking_model = StackingModel(
            base_models=base_models,
            meta_learner=None,  # Will be created during optimization if enabled
            n_splits=N_SPLITS_STACKING,
            random_state=RANDOM_STATE,
            use_features_in_meta=USE_FEATURES_IN_META
        )

        # Train the stacking model
        stacking_model.fit(
            X=X_train_final,
            y=y_train_final,
            X_val=X_val_final,
            y_val=y_val_final,
            model_dir=STACKING_MODEL_SAVE_DIR
        )

        # Save the stacking model configuration
        stacking_config_path = os.path.join(STACKING_MODEL_SAVE_DIR, "stacking_model_config.pkl")
        stacking_model.save(stacking_config_path)
        logger.info(f"Stacking model configuration saved to {stacking_config_path}")

    stacking_training_time = time.time() - stacking_start_time
    logger.info(f"Stacking model training completed in {stacking_training_time:.2f} seconds")
    record_metric("stacking_training_time", stacking_training_time, "performance")

    # Step 6: Evaluate model on validation set
    logger.info("Step 6: Evaluating model on validation set...")
    val_preds = stacking_model.predict(X_val_final)
    val_rmse = np.sqrt(mean_squared_error(y_val_final, val_preds))
    val_mae = mean_absolute_error(y_val_final, val_preds)
    val_r2 = r2_score(y_val_final, val_preds)

    logger.info(f"Validation RMSE: {val_rmse:.4f}")
    logger.info(f"Validation MAE: {val_mae:.4f}")
    logger.info(f"Validation R² Score: {val_r2:.4f}")

    # Record validation metrics
    record_metric("validation_rmse", val_rmse, "model_performance")
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

    # Step 7: Make predictions on test set
    logger.info("Step 7: Making predictions on test set...")
    test_preds = stacking_model.predict(X_test)
    final_predictions = np.round(test_preds).astype(int)

    # Step 8: Generate submission file
    logger.info("Step 8: Generating submission file...")
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
    record_metric("final_validation_rmse", val_rmse, "model_performance")
    record_metric("final_validation_mae", val_mae, "model_performance")
    record_metric("final_validation_r2", val_r2, "model_performance")

    # Generate performance dashboard
    dashboard_config = config.get('dashboards', {})
    if dashboard_config.get('enabled', True):
        try:
            logger.info("Generating performance dashboard...")
            output_dir = dashboard_config.get('output_dir', 'dashboards')
            dashboard_name = f"stacking_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dashboard_path = create_performance_dashboard(output_dir=output_dir, dashboard_name=dashboard_name)
            logger.info(f"Performance dashboard created at: {dashboard_path}")
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {str(e)}")

    logger.info("Stacking model training completed successfully.")
