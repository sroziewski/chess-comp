#!/usr/bin/env python
"""
Script to process chess puzzle data from training and testing datasets.

This script:
1. Reads training data from /raid/sroziewski/chess/training_data_02_01.csv
2. Reads testing data from /raid/sroziewski/chess/testing_data_cropped.csv
3. Extracts specific columns from both datasets
4. Concatenates the datasets
5. Adds an 'idx' column representing the row count
6. Saves the combined dataset to a file

Usage:
    python process_chess_data.py [--output OUTPUT_FILE]

Arguments:
    --output: Path to the output file (default: combined_puzzle_data.csv)

Example:
    python process_chess_data.py --output puzzle_data_combined.csv
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
    """Main function to process chess puzzle data."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process chess puzzle data from training and testing datasets.')
    parser.add_argument('--output', type=str, default='combined_puzzle_data.csv', help='Path to the output file')
    args = parser.parse_args()

    output_path = args.output

    # Define paths to data files
    train_file = "/raid/sroziewski/chess/training_data_02_01.csv"
    test_file = "/raid/sroziewski/chess/testing_data_cropped.csv"

    # Check if data files exist
    files = {
        "training_data": train_file,
        "testing_data": test_file
    }

    missing_files = []
    for file_name, file_path in files.items():
        if not os.path.exists(file_path):
            logger.warning(f"File {file_name} not found at {file_path}")
            missing_files.append(file_name)

    if missing_files:
        logger.error(f"The following data files are missing: {', '.join(missing_files)}")
        logger.error("Cannot proceed without both training and testing data files.")
        return

    # Load training data
    logger.info(f"Loading training data from {train_file}")
    try:
        train_df = pd.read_csv(train_file)
        logger.info(f"Loaded training data with shape: {train_df.shape}")
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        return

    # Load testing data
    logger.info(f"Loading testing data from {test_file}")
    try:
        test_df = pd.read_csv(test_file)
        logger.info(f"Loaded testing data with shape: {test_df.shape}")
    except Exception as e:
        logger.error(f"Error loading testing data: {str(e)}")
        return

    # Extract required columns from training data
    logger.info("Extracting required columns from training data")
    # Get all success_prob_* columns from training data
    success_prob_cols = [col for col in train_df.columns if col.startswith('success_prob_')]
    # Add other required columns
    required_cols = ['PuzzleId', 'FEN', 'Moves', 'Rating'] + success_prob_cols

    # Check if all required columns exist in training data
    missing_cols = [col for col in required_cols if col not in train_df.columns]
    if missing_cols:
        logger.warning(f"The following required columns are missing from training data: {', '.join(missing_cols)}")
        # Remove missing columns from required_cols
        required_cols = [col for col in required_cols if col not in missing_cols]

    # Extract the required columns from training data
    train_features = train_df[required_cols].copy()
    # Add is_train column to training data (1 = train)
    train_features['is_train'] = 1
    logger.info(f"Extracted {len(required_cols)} columns from training data and added is_train column")

    # Extract required columns from testing data
    logger.info("Extracting required columns from testing data")
    # Check if all required columns exist in testing data
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        logger.warning(f"The following required columns are missing from testing data: {', '.join(missing_cols)}")
        # Keep only columns that exist in both dataframes
        common_cols = [col for col in required_cols if col not in missing_cols]
        train_features = train_features[common_cols].copy()
        required_cols = common_cols

    # Extract the required columns from testing data
    test_features = test_df[required_cols].copy()
    # Add is_train column to testing data (0 = test)
    test_features['is_train'] = 0
    logger.info(f"Extracted {len(required_cols)} columns from testing data and added is_train column")

    # Concatenate training and testing data
    logger.info("Concatenating training and testing data")
    combined_df = pd.concat([train_features, test_features], ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")

    # Add idx column
    logger.info("Adding idx column")
    combined_df['idx'] = range(len(combined_df))
    logger.info(f"Final data shape: {combined_df.shape}")

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
