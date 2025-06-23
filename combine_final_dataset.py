#!/usr/bin/env python
"""
Script to combine combined_features.csv and combined_puzzle_data.csv into a single dataset.

This script:
1. Loads combined_features.csv and combined_puzzle_data.csv
2. Checks if the 'idx' column in both datasets match
3. Merges the datasets on the 'idx' column
4. Saves the combined dataset to a file

Usage:
    python combine_final_dataset.py [--output OUTPUT_FILE] [--features FEATURES_FILE] [--puzzle PUZZLE_FILE]

Arguments:
    --output: Path to the output file (default: final_dataset.csv)
    --features: Path to the features file (default: combined_features.csv)
    --puzzle: Path to the puzzle data file (default: combined_puzzle_data.csv)

Example:
    python combine_final_dataset.py --output my_final_dataset.csv
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

def main():
    """Main function to combine datasets."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Combine features and puzzle data into a single dataset.')
    parser.add_argument('--output', type=str, default='final_dataset.csv', help='Path to the output file')
    parser.add_argument('--features', type=str, default='combined_features.csv', help='Path to the features file')
    parser.add_argument('--puzzle', type=str, default='combined_puzzle_data.csv', help='Path to the puzzle data file')
    args = parser.parse_args()

    output_path = args.output
    features_path = args.features
    puzzle_path = args.puzzle

    # Check if input files exist
    files = {
        "features": features_path,
        "puzzle_data": puzzle_path
    }

    missing_files = []
    for file_name, file_path in files.items():
        if not os.path.exists(file_path):
            logger.warning(f"File {file_name} not found at {file_path}")
            missing_files.append(file_name)

    if missing_files:
        logger.error(f"The following data files are missing: {', '.join(missing_files)}")
        logger.error("Cannot proceed without both features and puzzle data files.")
        return

    # Load features data
    logger.info(f"Loading features data from {features_path}")
    try:
        features_df = pd.read_csv(features_path)
        logger.info(f"Loaded features data with shape: {features_df.shape}")
    except Exception as e:
        logger.error(f"Error loading features data: {str(e)}")
        return

    # Load puzzle data
    logger.info(f"Loading puzzle data from {puzzle_path}")
    try:
        puzzle_df = pd.read_csv(puzzle_path)
        logger.info(f"Loaded puzzle data with shape: {puzzle_df.shape}")
    except Exception as e:
        logger.error(f"Error loading puzzle data: {str(e)}")
        return

    # Check if 'idx' column exists in both dataframes
    if 'idx' not in features_df.columns:
        logger.error("'idx' column not found in features data")
        return
    if 'idx' not in puzzle_df.columns:
        logger.error("'idx' column not found in puzzle data")
        return

    # Check if idx values match between corresponding rows
    logger.info("Checking if idx values match between datasets")
    
    # Sort both dataframes by idx to ensure proper comparison
    features_df = features_df.sort_values('idx').reset_index(drop=True)
    puzzle_df = puzzle_df.sort_values('idx').reset_index(drop=True)
    
    # Get the common idx values
    common_idx = set(features_df['idx']).intersection(set(puzzle_df['idx']))
    logger.info(f"Found {len(common_idx)} common idx values between datasets")
    
    if len(common_idx) == 0:
        logger.error("No common idx values found between datasets. Cannot merge.")
        return
    
    # Check if idx values are the same for both datasets
    if len(features_df) == len(puzzle_df) and set(features_df['idx']) == set(puzzle_df['idx']):
        logger.info("All idx values match between datasets")
    else:
        logger.warning("Not all idx values match between datasets")
        logger.warning(f"Features dataset has {len(features_df)} rows, Puzzle dataset has {len(puzzle_df)} rows")
        logger.warning(f"Features dataset has {len(set(features_df['idx']))} unique idx values")
        logger.warning(f"Puzzle dataset has {len(set(puzzle_df['idx']))} unique idx values")
        logger.warning(f"There are {len(common_idx)} common idx values")
        
        # Filter both dataframes to include only common idx values
        logger.info("Filtering datasets to include only common idx values")
        features_df = features_df[features_df['idx'].isin(common_idx)]
        puzzle_df = puzzle_df[puzzle_df['idx'].isin(common_idx)]
        
        logger.info(f"After filtering: Features dataset has {len(features_df)} rows, Puzzle dataset has {len(puzzle_df)} rows")

    # Merge datasets on 'idx' column
    logger.info("Merging datasets on 'idx' column")
    
    # Check for duplicate column names (other than 'idx')
    duplicate_cols = set(features_df.columns).intersection(set(puzzle_df.columns)) - {'idx'}
    if duplicate_cols:
        logger.warning(f"Found {len(duplicate_cols)} duplicate column names: {duplicate_cols}")
        # Rename duplicate columns in puzzle_df to avoid conflicts
        puzzle_df = puzzle_df.rename(columns={col: f"{col}_puzzle" for col in duplicate_cols})
        logger.info(f"Renamed duplicate columns in puzzle data: {', '.join([f'{col} -> {col}_puzzle' for col in duplicate_cols])}")
    
    # Merge on 'idx' column
    combined_df = pd.merge(features_df, puzzle_df, on='idx', how='inner')
    logger.info(f"Combined data shape after merging: {combined_df.shape}")
    
    # Save the combined dataset
    logger.info(f"Saving combined data to {output_path}")
    try:
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved combined data to {output_path}")
    except Exception as e:
        logger.error(f"Error saving combined data: {str(e)}")

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Script started at {start_time}")

    main()

    end_time = datetime.now()
    logger.info(f"Script completed at {end_time}")
    logger.info(f"Total execution time: {end_time - start_time}")