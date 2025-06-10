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

# Set the boost_compute directory to /raid/sroziewski/.boost_compute
boost_compute_dir = '/raid/sroziewski/.boost_compute'
os.environ['BOOST_COMPUTE_DEFAULT_TEMP_PATH'] = boost_compute_dir

# Create the boost_compute directory if it doesn't exist
os.makedirs(boost_compute_dir, exist_ok=True)

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
