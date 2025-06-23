#!/usr/bin/env python
"""
Script to combine features from different feature sets into a single dataframe.

This script:
1. Reads eco_code_features_dense.csv from the eco_code_features directory
2. Reads the appropriate files from endgame_features, position_features, move_features, and move_analysis_features directories
3. Combines all datasets into a single dataframe, preserving the 'idx' column
4. Saves the combined dataframe to a file

Usage:
    python combine_features.py [--output OUTPUT_FILE] [--data-dir DATA_DIR]

Arguments:
    --output: Path to the output file (default: combined_features.csv)
    --data-dir: Path to the directory containing feature directories (default: /raid/sroziewski/dev/chess-comp/data)

Example:
    python combine_features.py --output all_features.csv --data-dir /path/to/data
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_feature_file(file_path):
    """
    Load a feature file and return a dataframe.

    Parameters:
    -----------
    file_path : str or Path
        Path to the feature file

    Returns:
    --------
    pd.DataFrame
        Dataframe containing the features
    """
    logger.info(f"Loading features from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded features with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading features from {file_path}: {str(e)}")
        return None

def main():
    """Main function to combine features."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Combine features from different feature sets into a single dataframe.')
    parser.add_argument('--output', type=str, default='combined_features.csv', help='Path to the output file')
    parser.add_argument('--data-dir', type=str, default='/raid/sroziewski/dev/chess-comp/data', help='Path to the directory containing feature directories')
    args = parser.parse_args()

    output_path = args.output
    data_dir = Path(args.data_dir)

    # Check if data directory exists
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist.")
        logger.info("Please provide a valid data directory path using the --data-dir argument.")
        return

    # Define paths to feature directories
    eco_code_dir = data_dir / "eco_code_features"
    endgame_dir = data_dir / "endgame_features"
    position_dir = data_dir / "position_features"
    move_dir = data_dir / "move_features"
    move_analysis_dir = data_dir / "move_analysis_features"

    # Check if feature directories exist
    directories = {
        "eco_code_features": eco_code_dir,
        "endgame_features": endgame_dir,
        "position_features": position_dir,
        "move_features": move_dir,
        "move_analysis_features": move_analysis_dir
    }

    missing_dirs = []
    for dir_name, dir_path in directories.items():
        if not dir_path.exists():
            logger.warning(f"Directory {dir_name} not found at {dir_path}")
            missing_dirs.append(dir_name)

    if missing_dirs:
        logger.warning(f"The following feature directories are missing: {', '.join(missing_dirs)}")
        logger.info("The script will attempt to load features from the directories that exist.")

    # Define paths to feature files
    eco_code_file = eco_code_dir / "eco_code_features_dense.csv"

    # For other feature sets, we'll use the files mentioned in the issue description
    # The issue description lists multiple files in each directory, but we'll use the .csv files
    endgame_file = endgame_dir / "70da7fd0c8a35b85d039d154acc76884.csv"
    position_file = position_dir / "80984165471aea0a3585fbb2e5c94fd8.csv"
    move_file = move_dir / "9a891f4e93390e9b037ceff94711b318.csv"
    move_analysis_file = move_analysis_dir / "d413861d5eac188ea1e14b2a92079121.csv"

    # Check if feature files exist
    files = {
        "eco_code_features_dense.csv": eco_code_file,
        "70da7fd0c8a35b85d039d154acc76884.csv (endgame)": endgame_file,
        "80984165471aea0a3585fbb2e5c94fd8.csv (position)": position_file,
        "9a891f4e93390e9b037ceff94711b318.csv (move)": move_file,
        "d413861d5eac188ea1e14b2a92079121.csv (move_analysis)": move_analysis_file
    }

    missing_files = []
    for file_name, file_path in files.items():
        if not file_path.exists():
            logger.warning(f"File {file_name} not found at {file_path}")
            missing_files.append(file_name)

    if missing_files:
        logger.warning(f"The following feature files are missing: {', '.join(missing_files)}")
        logger.info("The script will attempt to load features from the files that exist.")

    # Load feature files
    feature_dfs = {}

    # Load ECO code features
    eco_code_df = load_feature_file(eco_code_file)
    if eco_code_df is not None:
        feature_dfs['eco_code'] = eco_code_df

    # Load endgame features
    endgame_df = load_feature_file(endgame_file)
    if endgame_df is not None:
        feature_dfs['endgame'] = endgame_df

    # Load position features
    position_df = load_feature_file(position_file)
    if position_df is not None:
        feature_dfs['position'] = position_df

    # Load move features
    move_df = load_feature_file(move_file)
    if move_df is not None:
        feature_dfs['move'] = move_df

    # Load move analysis features
    move_analysis_df = load_feature_file(move_analysis_file)
    if move_analysis_df is not None:
        feature_dfs['move_analysis'] = move_analysis_df

    # Check if we have any features to combine
    if not feature_dfs:
        logger.error("No feature files were loaded successfully. Exiting.")
        return

    logger.info(f"Loaded {len(feature_dfs)} feature sets")

    # Combine features
    logger.info("Combining features")

    # Start with the first feature set
    first_key = list(feature_dfs.keys())[0]
    combined_df = feature_dfs[first_key].copy()
    logger.info(f"Starting with {first_key} features: {combined_df.shape}")

    # Merge with other feature sets on 'idx' column
    for key, df in list(feature_dfs.items())[1:]:
        logger.info(f"Merging with {key} features: {df.shape}")

        # Check if 'idx' column exists in both dataframes
        if 'idx' not in combined_df.columns:
            logger.error(f"'idx' column not found in combined dataframe")
            return
        if 'idx' not in df.columns:
            logger.error(f"'idx' column not found in {key} features")
            return

        # Check for duplicate column names (other than 'idx')
        duplicate_cols = set(combined_df.columns).intersection(set(df.columns)) - {'idx'}
        if duplicate_cols:
            logger.warning(f"Found {len(duplicate_cols)} duplicate column names: {duplicate_cols}")
            # Rename duplicate columns to avoid conflicts
            df = df.rename(columns={col: f"{col}_{key}" for col in duplicate_cols})

        # Merge on 'idx' column
        combined_df = pd.merge(combined_df, df, on='idx', how='outer')
        logger.info(f"Combined shape after merging {key} features: {combined_df.shape}")

    # Save the combined dataset
    logger.info(f"Saving combined features to {output_path}")
    try:
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved combined features to {output_path}")
    except Exception as e:
        logger.error(f"Error saving combined features: {str(e)}")

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Script started at {start_time}")

    main()

    end_time = datetime.now()
    logger.info(f"Script completed at {end_time}")
    logger.info(f"Total execution time: {end_time - start_time}")
