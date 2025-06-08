"""
Module for feature engineering pipelines.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import concurrent.futures
import functools
import os
import time
import datetime
import logging
from joblib import Memory
import threading

from chess_puzzle_rating.utils.progress import (
    setup_logging, get_logger, log_time, ProgressTracker, 
    track_progress, record_metric, create_performance_dashboard
)

from chess_puzzle_rating.features.position_features import extract_fen_features
from chess_puzzle_rating.features.move_features import extract_opening_move_features, infer_eco_codes, analyze_move_sequence
from chess_puzzle_rating.features.opening_tags import predict_missing_opening_tags
from chess_puzzle_rating.features.opening_features import engineer_chess_opening_features
from chess_puzzle_rating.features.endgame_features import extract_endgame_features
from chess_puzzle_rating.utils.config import get_config

# Get logger
logger = get_logger()

# Initialize memory as None, will be set up in complete_feature_engineering
memory = None

def setup_caching(config_path=None):
    """
    Set up caching with the given configuration.

    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file. If None, use default configuration.

    Returns
    -------
    joblib.Memory
        Memory object for caching
    """
    global memory

    # Get configuration
    config = get_config(config_path)
    performance_config = config.get('performance', {})
    caching_config = performance_config.get('caching', {})

    # Set up caching if enabled
    caching_enabled = caching_config.get('enabled', True)
    if caching_enabled:
        cache_dir = os.path.expanduser(caching_config.get('cache_dir', '~/.chess_puzzle_rating_cache'))
        os.makedirs(cache_dir, exist_ok=True)

        # Get cache size limit
        cache_size_gb = caching_config.get('max_cache_size_gb', 10)
        bytes_limit = cache_size_gb * 1024 * 1024 * 1024

        memory = Memory(
            cache_dir, 
            verbose=0,
            bytes_limit=bytes_limit,
            mmap_mode='r'
        )
        logger.info(f"Caching enabled. Using cache directory: {cache_dir} (max size: {cache_size_gb} GB)")

        # Record cache configuration
        record_metric("cache_enabled", 1, "cache_config")
        record_metric("cache_size_gb", cache_size_gb, "cache_config")
    else:
        # Create a no-op memory cache when caching is disabled
        memory = Memory(None, verbose=0)
        logger.info("Caching disabled.")
        record_metric("cache_enabled", 0, "cache_config")

    return memory


@log_time
def complete_feature_engineering(df, tag_column='OpeningTags', n_workers=None, config_path=None):
    """
    Complete pipeline for feature engineering with opening tag prediction.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles
    tag_column : str, optional
        Name of the column containing opening tags, by default 'OpeningTags'
    n_workers : int, optional
        Number of worker processes to use for parallel processing.
        If None, uses the value from config or the number of CPU cores.
    config_path : str, optional
        Path to the configuration file. If None, use default configuration.

    Returns
    -------
    tuple
        (final_features_df, model, predictions_df)
        - final_features_df: DataFrame with all engineered features
        - model: Trained model for predicting opening tags
        - predictions_df: DataFrame with predicted opening tags and confidence scores
    """
    # Get logger
    logger = get_logger()

    # Set up caching with the given configuration
    global memory
    memory = setup_caching(config_path)

    # Get configuration
    config = get_config(config_path)
    performance_config = config.get('performance', {})

    # Record start time for overall process
    start_time = time.time()

    # Get parallel processing configuration
    parallel_config = performance_config.get('parallel', {})

    # Determine the number of workers
    if n_workers is None:
        # Use config value if available, otherwise use all CPU cores
        n_workers = parallel_config.get('n_workers', None)
        if n_workers is None:
            n_workers = os.cpu_count()

    # Set thread limit per worker
    max_threads = parallel_config.get('max_threads_per_worker', 4)

    # Set environment variables to control thread usage in libraries like NumPy and OpenMP
    os.environ['OMP_NUM_THREADS'] = str(max_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(max_threads)
    os.environ['MKL_NUM_THREADS'] = str(max_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(max_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(max_threads)

    logger.info(f"Using {n_workers} workers for parallel feature extraction (max {max_threads} threads per worker)")

    # Step 1: Extract position, move, and endgame features in parallel
    logger.info("Step 1: Extracting position, move, and endgame features in parallel")

    # Cache the expensive feature extraction functions
    cached_extract_fen_features = memory.cache(extract_fen_features)
    cached_analyze_move_sequence = memory.cache(analyze_move_sequence)
    cached_extract_endgame_features = memory.cache(extract_endgame_features)

    # Record feature extraction start time
    feature_extraction_start = time.time()

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        logger.info("Submitting feature extraction tasks to process pool")

        # Submit all feature extraction tasks
        position_features_future = executor.submit(cached_extract_fen_features, df)
        move_features_future = executor.submit(extract_opening_move_features, df)
        eco_features_future = executor.submit(infer_eco_codes, df)
        move_analysis_features_future = executor.submit(cached_analyze_move_sequence, df)
        endgame_features_future = executor.submit(cached_extract_endgame_features, df)

        # Create a list of futures and their descriptions for tracking
        futures = [
            (position_features_future, "Position features"),
            (move_features_future, "Move features"),
            (eco_features_future, "ECO code features"),
            (move_analysis_features_future, "Move analysis features"),
            (endgame_features_future, "Endgame features")
        ]

        # Track progress of futures as they complete
        completed = 0
        total = len(futures)
        logger.info(f"Waiting for {total} feature extraction tasks to complete")

        # Get results as they complete
        results = {}
        for future, description in futures:
            try:
                # Wait for the future to complete
                result = future.result()
                results[description] = result

                # Update progress
                completed += 1
                progress_pct = (completed / total) * 100
                logger.info(f"Completed {description} ({completed}/{total}, {progress_pct:.1f}%)")

                # Record metrics
                if isinstance(result, pd.DataFrame):
                    record_metric(f"{description.lower().replace(' ', '_')}_columns", result.shape[1], "feature_stats")
                    logger.info(f"{description} generated {result.shape[1]} features")
            except Exception as e:
                logger.error(f"Error extracting {description}: {str(e)}")
                raise

        # Assign results to variables
        position_features = results["Position features"]
        move_features = results["Move features"]
        eco_features = results["ECO code features"]
        move_analysis_features = results["Move analysis features"]
        endgame_features = results["Endgame features"]

        # Record total feature extraction time
        feature_extraction_time = time.time() - feature_extraction_start
        logger.info(f"Parallel feature extraction completed in {feature_extraction_time:.2f} seconds")
        record_metric("parallel_feature_extraction_time", feature_extraction_time, "performance")

    # Step 2: Predict missing opening tags
    logger.info("Step 2: Predicting missing opening tags")
    tag_prediction_start = time.time()
    predictions, model, combined_features = predict_missing_opening_tags(
        df, 
        tag_column=tag_column,
        fen_features=position_features,
        move_features=move_features,
        eco_features=eco_features
    )
    tag_prediction_time = time.time() - tag_prediction_start
    logger.info(f"Opening tag prediction completed in {tag_prediction_time:.2f} seconds")
    record_metric("tag_prediction_time", tag_prediction_time, "performance")

    if predictions is not None:
        logger.info(f"Generated predictions for {len(predictions)} puzzles")

        # Record metrics about predictions
        high_conf_predictions = predictions[predictions['prediction_confidence'] >= 0.7]
        high_conf_count = len(high_conf_predictions)
        high_conf_pct = (high_conf_count / len(predictions)) * 100 if len(predictions) > 0 else 0

        logger.info(f"High confidence predictions: {high_conf_count} ({high_conf_pct:.1f}%)")
        record_metric("high_confidence_predictions", high_conf_count, "prediction_stats")
        record_metric("high_confidence_predictions_pct", high_conf_pct, "prediction_stats")

        if 'prediction_confidence' in predictions.columns:
            avg_confidence = predictions['prediction_confidence'].mean()
            record_metric("avg_prediction_confidence", avg_confidence, "prediction_stats")
            logger.info(f"Average prediction confidence: {avg_confidence:.4f}")

    # Step 3: Create a new column with original + predicted tags
    logger.info("Step 3: Creating enhanced opening tags")
    enhanced_tags = df[tag_column].copy()

    # Add high-confidence predictions
    high_conf_mask = predictions['prediction_confidence'] >= 0.7
    enhanced_count = 0
    for idx in predictions[high_conf_mask].index:
        family = predictions.loc[idx, 'predicted_family']
        # Only add prediction if original is empty
        if pd.isna(enhanced_tags.loc[idx]) or enhanced_tags.loc[idx] == '':
            enhanced_tags.loc[idx] = f"{family} (predicted)"
            enhanced_count += 1

    logger.info(f"Enhanced {enhanced_count} empty tags with high-confidence predictions")
    record_metric("enhanced_tags_count", enhanced_count, "prediction_stats")

    # Step 4: Engineer opening features using the enhanced tags
    logger.info("Step 4: Engineering opening features with enhanced tags")
    opening_features_start = time.time()

    # We can use the original function but with the enhanced tags
    df_with_enhanced_tags = df.copy()
    df_with_enhanced_tags['EnhancedOpeningTags'] = enhanced_tags

    # Use the opening feature engineering function
    opening_features = engineer_chess_opening_features(
        df_with_enhanced_tags,
        tag_column='EnhancedOpeningTags',
        min_family_freq=20,
        min_variation_freq=10,
        min_keyword_freq=50
    )

    opening_features_time = time.time() - opening_features_start
    logger.info(f"Opening features engineering completed in {opening_features_time:.2f} seconds")
    logger.info(f"Generated {opening_features.shape[1]} opening features")
    record_metric("opening_features_time", opening_features_time, "performance")
    record_metric("opening_features_count", opening_features.shape[1], "feature_stats")

    # Step 5: Add prediction metadata
    logger.info("Step 5: Adding prediction metadata")
    opening_features['has_original_tag'] = (~df[tag_column].isna() & (df[tag_column] != '')).astype(int)
    opening_features['has_predicted_tag'] = (~df[tag_column].isna() & (df[tag_column] != '') |
                                            (predictions['prediction_confidence'] >= 0.7)).astype(int)
    opening_features['tag_prediction_confidence'] = 0.0

    # Add confidence for predicted tags
    for idx in predictions.index:
        opening_features.loc[idx, 'tag_prediction_confidence'] = predictions.loc[idx, 'prediction_confidence']

    # Set confidence to 1.0 for original tags
    original_tag_mask = (~df[tag_column].isna() & (df[tag_column] != ''))
    opening_features.loc[original_tag_mask, 'tag_prediction_confidence'] = 1.0

    # Record tag statistics
    original_tags_count = original_tag_mask.sum()
    total_tags_with_predictions = opening_features['has_predicted_tag'].sum()
    logger.info(f"Original tags: {original_tags_count}, Total tags after prediction: {total_tags_with_predictions}")
    record_metric("original_tags_count", original_tags_count, "tag_stats")
    record_metric("total_tags_with_predictions", total_tags_with_predictions, "tag_stats")

    # Step 6: Combine all feature sets
    logger.info("Step 6: Combining all feature sets")
    combine_start = time.time()

    final_features = pd.concat([
        opening_features,
        position_features,
        move_features,
        eco_features,
        move_analysis_features,
        endgame_features
    ], axis=1)

    # Fill any remaining NaN values
    final_features = final_features.fillna(0)

    combine_time = time.time() - combine_start
    logger.info(f"Feature combination completed in {combine_time:.2f} seconds")
    record_metric("feature_combination_time", combine_time, "performance")

    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Created final feature set with {final_features.shape[1]} features in {total_time:.2f} seconds")
    record_metric("total_feature_engineering_time", total_time, "performance")
    record_metric("total_features_count", final_features.shape[1], "feature_stats")

    return final_features, model, predictions
