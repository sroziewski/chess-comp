"""
Test script for the data pipeline.

This script tests the data pipeline by running it and logging information
about the resulting datasets.
"""

import os
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from chess_puzzle_rating.data.pipeline import run_data_pipeline
from chess_puzzle_rating.utils.progress import get_logger
import pandas as pd
from chess_puzzle_rating.features.pipeline import (
    complete_feature_engineering as original_complete_feature_engineering,
    load_dataframe, save_dataframe
)

# Set the boost_compute directory to /raid/sroziewski/.boost_compute
boost_compute_dir = '/raid/sroziewski/.boost_compute'
os.environ['BOOST_COMPUTE_DEFAULT_TEMP_PATH'] = boost_compute_dir

# Create the boost_compute directory if it doesn't exist
os.makedirs(boost_compute_dir, exist_ok=True)

# Path to pre-computed theme features
THEME_FEATURES_PATH = '/raid/sroziewski/dev/chess-comp/data/themes_features'

def custom_complete_feature_engineering(df, tag_column='OpeningTags', n_workers=None, config_path=None):
    """
    Custom version of complete_feature_engineering that loads pre-computed theme features
    instead of computing them.

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
    log = get_logger()
    log.info("Using custom feature engineering with pre-computed theme features")

    # First, run the original feature engineering to get all features
    log.info("Running original feature engineering")
    final_features, model, predictions = original_complete_feature_engineering(df, tag_column, n_workers, config_path)

    # Now, load the pre-computed theme features
    log.info(f"Loading pre-computed theme features from {THEME_FEATURES_PATH}")

    # Try to find the most recent theme features file in the directory
    try:
        # Check if the directory exists
        if not os.path.exists(THEME_FEATURES_PATH):
            log.error(f"Theme features directory not found: {THEME_FEATURES_PATH}")
            log.error("Using original features instead")
            return final_features, model, predictions

        theme_files = [f for f in os.listdir(THEME_FEATURES_PATH) if f.endswith('.csv')]
        if not theme_files:
            log.error(f"No theme feature files found in {THEME_FEATURES_PATH}")
            log.error("Using original features instead")
            return final_features, model, predictions

        # Get the most recent file based on modification time
        most_recent = max(theme_files, key=lambda f: os.path.getmtime(os.path.join(THEME_FEATURES_PATH, f)))
        theme_file_path = os.path.join(THEME_FEATURES_PATH, most_recent)

        log.info(f"Loading theme features from {theme_file_path}")
        theme_features = pd.read_csv(theme_file_path, index_col=0)

        # Check if the theme features have the same index as the final features
        missing_indices = [idx for idx in final_features.index if idx not in theme_features.index]
        if missing_indices:
            log.warning(f"{len(missing_indices)} indices in final features are missing from theme features")
            log.warning("This may cause issues with feature combination")

        # Replace theme features in the final features DataFrame
        # First, identify theme feature columns in the final features
        theme_cols = [col for col in final_features.columns if col.startswith('theme_') or col in ['is_mate', 'is_fork', 'is_pin', 'is_skewer', 'is_discovery', 'is_sacrifice', 'is_promotion', 'is_endgame', 'is_middlegame', 'is_opening']]

        # Remove existing theme features
        if theme_cols:
            log.info(f"Removing {len(theme_cols)} existing theme features")
            final_features = final_features.drop(columns=theme_cols)

        # Add pre-computed theme features
        log.info(f"Adding {theme_features.shape[1]} pre-computed theme features")

        # Ensure the theme features have the same index as the final features
        # First, convert index to string if it's not already
        if not all(isinstance(idx, str) for idx in theme_features.index):
            log.info("Converting theme features index to string")
            theme_features.index = theme_features.index.astype(str)

        if not all(isinstance(idx, str) for idx in final_features.index):
            log.info("Converting final features index to string")
            final_features.index = final_features.index.astype(str)

        # Now reindex
        theme_features = theme_features.reindex(final_features.index)

        # Combine the features
        final_features = pd.concat([final_features, theme_features], axis=1)

        # Fill any NaN values that might have been introduced
        final_features = final_features.fillna(0)

        log.info(f"Final feature set has {final_features.shape[1]} features")

    except Exception as e:
        log.error(f"Error loading pre-computed theme features: {str(e)}")
        log.error("Using original features instead")

    return final_features, model, predictions

def main():
    """Run the data pipeline and log information about the results."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test the data pipeline with custom configuration.')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--cache-dir', type=str, help='Path to the cache directory')
    args = parser.parse_args()

    # Get logger
    log = get_logger()

    log.info("Testing the data pipeline...")

    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    # If cache-dir is provided, create a temporary config file with the custom cache directory
    config_path = args.config
    if args.cache_dir:
        import yaml
        import tempfile
        from chess_puzzle_rating.utils.config import get_config

        # Load existing config or get default
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = get_config()

        # Ensure performance.caching section exists
        if 'performance' not in config:
            config['performance'] = {}
        if 'caching' not in config['performance']:
            config['performance']['caching'] = {}

        # Set the cache directory
        config['performance']['caching']['cache_dir'] = args.cache_dir

        # Create a temporary config file
        fd, temp_config_path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(fd, 'w') as f:
            yaml.dump(config, f)

        config_path = temp_config_path
        log.info(f"Using temporary config with cache directory: {args.cache_dir}")

    # Monkey patch the complete_feature_engineering function to use our custom one
    log.info("Monkey patching complete_feature_engineering to use pre-computed theme features")
    import chess_puzzle_rating.features.pipeline
    original_function = chess_puzzle_rating.features.pipeline.complete_feature_engineering
    chess_puzzle_rating.features.pipeline.complete_feature_engineering = custom_complete_feature_engineering

    # Run the pipeline
    try:
        X_train, X_test, y_train, test_ids = run_data_pipeline(config_path)

        # Log information about the results
        log.info("\nPipeline completed successfully!")
        log.info(f"X_train shape: {X_train.shape}")
        log.info(f"X_test shape: {X_test.shape}")
        log.info(f"y_train shape: {y_train.shape}")
        log.info(f"test_ids shape: {test_ids.shape}")

        # Log some sample data
        log.info("\nSample of X_train features:")
        log.info(X_train.head())

        log.info("\nSample of y_train values:")
        log.info(y_train.head())

        log.info("\nSample of test_ids:")
        log.info(test_ids.head())

        # Log feature statistics
        log.info("\nFeature statistics (X_train):")
        log.info(X_train.describe().T[['count', 'mean', 'min', 'max']].head(10))

        # Check for any remaining NaN values
        train_nan_count = X_train.isna().sum().sum()
        test_nan_count = X_test.isna().sum().sum()
        log.info(f"\nNaN values in X_train: {train_nan_count}")
        log.info(f"NaN values in X_test: {test_nan_count}")

        # List checkpoint files
        log.info("\nCheckpoint files created:")
        checkpoint_files = [f for f in os.listdir("checkpoints") if f.endswith(".parquet")]
        for i, file in enumerate(checkpoint_files[:10]):  # Show first 10 files
            log.info(f"  {i+1}. {file}")
        if len(checkpoint_files) > 10:
            log.info(f"  ... and {len(checkpoint_files) - 10} more files")

        # Check for analysis files
        analysis_dir = Path("analysis")
        if analysis_dir.exists():
            log.info("\nAnalysis files created (for feature exploration):")
            analysis_files = list(analysis_dir.glob("*.csv")) + list(analysis_dir.glob("*.parquet"))

            # Group files by base name (without extension)
            file_groups = {}
            for file in analysis_files:
                base_name = file.stem
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append(file)

            # Log grouped files
            for i, (base_name, files) in enumerate(sorted(file_groups.items())):
                log.info(f"  {i+1}. {base_name}:")
                for file in sorted(files):
                    log.info(f"     - {file.name} ({file.stat().st_size / (1024*1024):.2f} MB)")

                # Check if metadata exists
                metadata_file = analysis_dir / f"{base_name}_metadata.json"
                if metadata_file.exists():
                    log.info(f"     - {metadata_file.name} (metadata)")

            log.info("\nYou can use these files to analyze the features without re-running the pipeline.")
            log.info("For example, load them in a Jupyter notebook with:")
            log.info("  features_df = pd.read_parquet('analysis/features_after_engineering_TIMESTAMP.parquet')")
        else:
            log.info("\nNo analysis files were created. This is unexpected.")

        log.info("\nTest completed successfully!")

    except Exception as e:
        log.error(f"Error running the pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
