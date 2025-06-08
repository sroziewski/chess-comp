"""
Test script for the data pipeline.

This script tests the data pipeline by running it and printing information
about the resulting datasets.
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from chess_puzzle_rating.data.pipeline import run_data_pipeline

def main():
    """Run the data pipeline and print information about the results."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test the data pipeline with custom configuration.')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--cache-dir', type=str, help='Path to the cache directory')
    args = parser.parse_args()

    print("Testing the data pipeline...")

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
        print(f"Using temporary config with cache directory: {args.cache_dir}")

    # Run the pipeline
    try:
        X_train, X_test, y_train, test_ids = run_data_pipeline(config_path)

        # Print information about the results
        print("\nPipeline completed successfully!")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"test_ids shape: {test_ids.shape}")

        # Print some sample data
        print("\nSample of X_train features:")
        print(X_train.head())

        print("\nSample of y_train values:")
        print(y_train.head())

        print("\nSample of test_ids:")
        print(test_ids.head())

        # Print feature statistics
        print("\nFeature statistics (X_train):")
        print(X_train.describe().T[['count', 'mean', 'min', 'max']].head(10))

        # Check for any remaining NaN values
        train_nan_count = X_train.isna().sum().sum()
        test_nan_count = X_test.isna().sum().sum()
        print(f"\nNaN values in X_train: {train_nan_count}")
        print(f"NaN values in X_test: {test_nan_count}")

        # List checkpoint files
        print("\nCheckpoint files created:")
        checkpoint_files = [f for f in os.listdir("checkpoints") if f.endswith(".parquet")]
        for i, file in enumerate(checkpoint_files[:10]):  # Show first 10 files
            print(f"  {i+1}. {file}")
        if len(checkpoint_files) > 10:
            print(f"  ... and {len(checkpoint_files) - 10} more files")

        # Check for analysis files
        analysis_dir = Path("analysis")
        if analysis_dir.exists():
            print("\nAnalysis files created (for feature exploration):")
            analysis_files = list(analysis_dir.glob("*.csv")) + list(analysis_dir.glob("*.parquet"))

            # Group files by base name (without extension)
            file_groups = {}
            for file in analysis_files:
                base_name = file.stem
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append(file)

            # Print grouped files
            for i, (base_name, files) in enumerate(sorted(file_groups.items())):
                print(f"  {i+1}. {base_name}:")
                for file in sorted(files):
                    print(f"     - {file.name} ({file.stat().st_size / (1024*1024):.2f} MB)")

                # Check if metadata exists
                metadata_file = analysis_dir / f"{base_name}_metadata.json"
                if metadata_file.exists():
                    print(f"     - {metadata_file.name} (metadata)")

            print("\nYou can use these files to analyze the features without re-running the pipeline.")
            print("For example, load them in a Jupyter notebook with:")
            print("  features_df = pd.read_parquet('analysis/features_after_engineering_TIMESTAMP.parquet')")
        else:
            print("\nNo analysis files were created. This is unexpected.")

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"Error running the pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
