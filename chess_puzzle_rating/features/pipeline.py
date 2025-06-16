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
import hashlib
import json

from chess_puzzle_rating.utils.progress import (
    setup_logging, get_logger, log_time, ProgressTracker, 
    track_progress, record_metric, create_performance_dashboard
)

from chess_puzzle_rating.features.position_features import extract_fen_features
from chess_puzzle_rating.features.move_features import extract_opening_move_features, infer_eco_codes, analyze_move_sequence
from chess_puzzle_rating.features.opening_tags import predict_missing_opening_tags
from chess_puzzle_rating.features.opening_features import engineer_chess_opening_features
from chess_puzzle_rating.features.endgame_features import extract_endgame_features
from chess_puzzle_rating.features.theme_features import engineer_chess_theme_features
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
    tuple
        (joblib.Memory, data_dir)
        - memory: Memory object for caching
        - data_dir: Directory for storing dataframes
    """
    global memory

    # Get configuration
    config = get_config(config_path)
    performance_config = config.get('performance', {})
    caching_config = performance_config.get('caching', {})

    # Set up caching if enabled
    caching_enabled = caching_config.get('enabled', True)

    # Create data directory for dataframe caching
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)

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
        logger.info(f"DataFrame caching enabled. Using data directory: {data_dir}")

        # Record cache configuration
        record_metric("cache_enabled", 1, "cache_config")
        record_metric("cache_size_gb", cache_size_gb, "cache_config")
    else:
        # Create a no-op memory cache when caching is disabled
        memory = Memory(None, verbose=0)
        logger.info("Caching disabled.")
        record_metric("cache_enabled", 0, "cache_config")

    return memory, data_dir


def generate_dataframe_hash(df):
    """
    Generate a hash for a dataframe based on its content and structure.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to hash

    Returns
    -------
    str
        A hash string that uniquely identifies the dataframe
    """
    # Create a hash based on dataframe shape, column names, and a sample of values
    hash_input = f"{df.shape}_{list(df.columns)}"

    # Add a sample of values if the dataframe is not empty
    if not df.empty:
        # Sample values from the dataframe (first and last rows)
        sample_values = df.iloc[[0, -1]].to_json() if len(df) > 1 else df.iloc[0].to_json()
        hash_input += sample_values

    # Create a hash
    return hashlib.md5(hash_input.encode()).hexdigest()


def save_dataframe(df, step_name, data_dir):
    """
    Save a dataframe to disk with metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to save
    step_name : str
        The name of the feature engineering step
    data_dir : str
        The directory to save the dataframe in

    Returns
    -------
    str
        The path to the saved dataframe
    """
    # Create a hash for the dataframe
    df_hash = generate_dataframe_hash(df)

    # Create a subdirectory for this step if it doesn't exist
    step_dir = os.path.join(data_dir, step_name)
    os.makedirs(step_dir, exist_ok=True)

    # Create file paths
    csv_path = os.path.join(step_dir, f"{df_hash}.csv")
    metadata_path = os.path.join(step_dir, f"{df_hash}.json")

    # Save the dataframe
    df.to_csv(csv_path, index=True)

    # Save metadata
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'shape': df.shape,
        'columns': list(df.columns),
        'hash': df_hash
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved {step_name} dataframe to {csv_path} (shape: {df.shape})")

    return csv_path


def load_dataframe(step_name, data_dir, df_hash=None):
    """
    Load a dataframe from disk.

    Parameters
    ----------
    step_name : str
        The name of the feature engineering step
    data_dir : str
        The directory to load the dataframe from
    df_hash : str, optional
        The hash of the dataframe to load. If None, loads the most recent dataframe.

    Returns
    -------
    pandas.DataFrame or None
        The loaded dataframe, or None if no dataframe was found
    """
    # Create the step directory path
    step_dir = os.path.join(data_dir, step_name)

    # Check if the directory exists
    if not os.path.exists(step_dir):
        logger.info(f"No cached data found for {step_name}")
        return None

    # If a specific hash is provided, try to load that dataframe
    if df_hash:
        csv_path = os.path.join(step_dir, f"{df_hash}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            logger.info(f"Loaded {step_name} dataframe from {csv_path} (shape: {df.shape})")
            return df
        else:
            logger.info(f"No cached data found for {step_name} with hash {df_hash}")
            return None

    # Otherwise, find the most recent dataframe
    csv_files = [f for f in os.listdir(step_dir) if f.endswith('.csv')]

    if not csv_files:
        logger.info(f"No cached data found for {step_name}")
        return None

    # Get the most recent file based on modification time
    most_recent = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(step_dir, f)))
    csv_path = os.path.join(step_dir, most_recent)

    # Load the dataframe
    df = pd.read_csv(csv_path, index_col=0)
    logger.info(f"Loaded {step_name} dataframe from {csv_path} (shape: {df.shape})")

    return df


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
    memory, data_dir = setup_caching(config_path)

    # Get configuration
    config = get_config(config_path)
    performance_config = config.get('performance', {})

    # Record start time for overall process
    start_time = time.time()

    # Generate a hash for the input dataframe to use for caching
    input_df_hash = generate_dataframe_hash(df)

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

    # Try to load cached feature dataframes
    position_features = load_dataframe("position_features", data_dir)
    move_features = load_dataframe("move_features", data_dir)
    eco_features = load_dataframe("eco_features", data_dir)
    move_analysis_features = load_dataframe("move_analysis_features", data_dir)
    endgame_features = load_dataframe("endgame_features", data_dir)

    # Check which features need to be computed
    compute_position = position_features is None
    compute_move = move_features is None
    compute_eco = eco_features is None
    compute_move_analysis = move_analysis_features is None
    compute_endgame = endgame_features is None

    # If all features are already cached, skip computation
    if not any([compute_position, compute_move, compute_eco, compute_move_analysis, compute_endgame]):
        logger.info("All Step 1 features loaded from cache")
    else:
        # Use ProcessPoolExecutor for CPU-bound tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            logger.info("Submitting feature extraction tasks to process pool")

            futures = []

            # Submit only the tasks that need to be computed
            if compute_position:
                position_features_future = executor.submit(cached_extract_fen_features, df)
                futures.append((position_features_future, "Position features"))

            if compute_move:
                move_features_future = executor.submit(extract_opening_move_features, df, moves_column='MovesPGN')
                futures.append((move_features_future, "Move features"))

            if compute_eco:
                eco_features_future = executor.submit(infer_eco_codes, df, moves_column='MovesPGN')
                futures.append((eco_features_future, "ECO code features"))

            if compute_move_analysis:
                move_analysis_features_future = executor.submit(cached_analyze_move_sequence, df, moves_column='MovesPGN')
                futures.append((move_analysis_features_future, "Move analysis features"))

            if compute_endgame:
                endgame_features_future = executor.submit(cached_extract_endgame_features, df)
                futures.append((endgame_features_future, "Endgame features"))

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

                        # Save the dataframe to cache
                        step_name = description.lower().replace(' ', '_')
                        save_dataframe(result, step_name, data_dir)
                except Exception as e:
                    logger.error(f"Error extracting {description}: {str(e)}")
                    raise

            # Assign results to variables
            if compute_position:
                position_features = results["Position features"]
            if compute_move:
                move_features = results["Move features"]
            if compute_eco:
                eco_features = results["ECO code features"]
            if compute_move_analysis:
                move_analysis_features = results["Move analysis features"]
            if compute_endgame:
                endgame_features = results["Endgame features"]

            # Record total feature extraction time
            feature_extraction_time = time.time() - feature_extraction_start
            logger.info(f"Parallel feature extraction completed in {feature_extraction_time:.2f} seconds")
            record_metric("parallel_feature_extraction_time", feature_extraction_time, "performance")

    # Step 2: Predict missing opening tags
    logger.info("Step 2: Predicting missing opening tags")
    tag_prediction_start = time.time()

    # Try to load cached predictions
    predictions = load_dataframe("opening_tag_predictions", data_dir)
    combined_features = load_dataframe("opening_tag_combined_features", data_dir)

    # Check if we need to compute predictions
    compute_predictions = predictions is None or combined_features is None

    if not compute_predictions:
        logger.info("Loaded opening tag predictions from cache")
        model = None  # We don't cache the model, but it's not needed if we have the predictions
    else:
        # Compute predictions
        predictions, model, combined_features = predict_missing_opening_tags(
            df, 
            tag_column=tag_column,
            fen_features=position_features,
            move_features=move_features,
            eco_features=eco_features
        )

        # Save predictions and combined features to cache
        if predictions is not None:
            save_dataframe(predictions, "opening_tag_predictions", data_dir)
        if combined_features is not None:
            save_dataframe(combined_features, "opening_tag_combined_features", data_dir)

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

    # Try to load cached enhanced tags
    enhanced_tags_df = load_dataframe("enhanced_opening_tags", data_dir)

    if enhanced_tags_df is not None:
        logger.info("Loaded enhanced opening tags from cache")
        enhanced_tags = enhanced_tags_df['EnhancedOpeningTags']
        enhanced_count = enhanced_tags_df['enhanced_count'].iloc[0] if 'enhanced_count' in enhanced_tags_df.columns else 0
    else:
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

        # Save enhanced tags to cache
        enhanced_tags_df = pd.DataFrame({
            'EnhancedOpeningTags': enhanced_tags,
            'enhanced_count': [enhanced_count] * len(enhanced_tags)
        }, index=enhanced_tags.index)
        save_dataframe(enhanced_tags_df, "enhanced_opening_tags", data_dir)

    logger.info(f"Enhanced {enhanced_count} empty tags with high-confidence predictions")
    record_metric("enhanced_tags_count", enhanced_count, "prediction_stats")

    # Step 4: Engineer opening features using the enhanced tags
    logger.info("Step 4: Engineering opening features with enhanced tags")
    opening_features_start = time.time()

    # Try to load cached opening features
    opening_features = load_dataframe("opening_features", data_dir)

    if opening_features is not None:
        logger.info("Loaded opening features from cache")
    else:
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

        # Save opening features to cache
        save_dataframe(opening_features, "opening_features", data_dir)

        opening_features_time = time.time() - opening_features_start
        logger.info(f"Opening features engineering completed in {opening_features_time:.2f} seconds")
        logger.info(f"Generated {opening_features.shape[1]} opening features")
        record_metric("opening_features_time", opening_features_time, "performance")
        record_metric("opening_features_count", opening_features.shape[1], "feature_stats")

    # Record metrics even if loaded from cache
    if opening_features is not None and not opening_features.empty:
        record_metric("opening_features_count", opening_features.shape[1], "feature_stats")

    # Step 5: Add prediction metadata
    logger.info("Step 5: Adding prediction metadata")

    # Try to load cached opening features with metadata
    opening_features_with_metadata = load_dataframe("opening_features_with_metadata", data_dir)

    if opening_features_with_metadata is not None:
        logger.info("Loaded opening features with metadata from cache")
        opening_features = opening_features_with_metadata

        # Extract statistics for logging
        original_tag_mask = (~df[tag_column].isna() & (df[tag_column] != ''))
        original_tags_count = original_tag_mask.sum()
        total_tags_with_predictions = opening_features['has_predicted_tag'].sum()
    else:
        # Add metadata to opening features
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

        # Save opening features with metadata to cache
        save_dataframe(opening_features, "opening_features_with_metadata", data_dir)

    logger.info(f"Original tags: {original_tags_count}, Total tags after prediction: {total_tags_with_predictions}")
    record_metric("original_tags_count", original_tags_count, "tag_stats")
    record_metric("total_tags_with_predictions", total_tags_with_predictions, "tag_stats")

    # Step 6: Engineer theme features
    logger.info("Step 6: Engineering theme features")
    theme_features_start = time.time()

    # Try to load cached theme features
    theme_features = load_dataframe("theme_features", data_dir)

    if theme_features is not None:
        logger.info("Loaded theme features from cache")
    else:
        # Use the theme feature engineering function
        theme_features = engineer_chess_theme_features(
            df,
            theme_column='Themes',
            min_theme_freq=5,
            max_themes=100,
            n_svd_components=10,
            n_hash_features=15
        )

        # Save theme features to cache
        save_dataframe(theme_features, "theme_features", data_dir)

        theme_features_time = time.time() - theme_features_start
        logger.info(f"Theme features engineering completed in {theme_features_time:.2f} seconds")
        logger.info(f"Generated {theme_features.shape[1]} theme features")
        record_metric("theme_features_time", theme_features_time, "performance")
        record_metric("theme_features_count", theme_features.shape[1], "feature_stats")

    # Record metrics even if loaded from cache
    if theme_features is not None and not theme_features.empty:
        record_metric("theme_features_count", theme_features.shape[1], "feature_stats")

    # Step 7: Combine all feature sets
    logger.info("Step 7: Combining all feature sets")
    combine_start = time.time()

    # Try to load cached final features
    final_features = load_dataframe("final_features", data_dir)

    if final_features is not None:
        logger.info("Loaded final combined features from cache")
    else:
        # Combine all feature sets
        final_features = pd.concat([
            opening_features,
            theme_features,
            position_features,
            move_features,
            eco_features,
            move_analysis_features,
            endgame_features
        ], axis=1)

        # Fill any remaining NaN values
        final_features = final_features.fillna(0)

        # Save final features to cache
        save_dataframe(final_features, "final_features", data_dir)

        combine_time = time.time() - combine_start
        logger.info(f"Feature combination completed in {combine_time:.2f} seconds")
        logger.info(f"Final feature set has {final_features.shape[1]} features")
        record_metric("feature_combination_time", combine_time, "performance")
        record_metric("total_features", final_features.shape[1], "feature_stats")

    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Created final feature set with {final_features.shape[1]} features in {total_time:.2f} seconds")
    record_metric("total_feature_engineering_time", total_time, "performance")
    record_metric("total_features_count", final_features.shape[1], "feature_stats")

    return final_features, model, predictions
