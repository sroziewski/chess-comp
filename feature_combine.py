#!/usr/bin/env python
"""
Script to combine features from eng_features.csv with selected columns from train_probe.csv and testing_data_cropped.csv.

This script:
1. Loads engineered features from eng_features.csv
2. Loads original features from train_probe.csv
3. Loads test data from testing_data_cropped.csv
4. Extracts required columns from original features (RatingDeviation, Popularity, NbPlays, and success_prob_* columns)
5. Combines train and test data
6. Maps the numeric indices in the engineered features to the corresponding PuzzleId values in the combined original data
7. Combines the datasets
8. Saves the combined dataset to a file named features_combined.csv

Usage:
    python feature_combine.py [--output OUTPUT_FILE] [--eng-features ENG_FEATURES_FILE] 
                             [--original-data ORIGINAL_DATA_FILE] [--test-data TEST_DATA_FILE]

Arguments:
    --output: Path to the output file (default: features_combined.csv)
    --eng-features: Path to the engineered features file (default: eng_features.csv)
    --original-data: Path to the original data file (default: train_probe.csv)
    --test-data: Path to the test data file (default: /raid/sroziewski/chess/testing_data_cropped.csv)

Example:
    python feature_combine.py --output combined_features.csv --eng-features my_features.csv --original-data my_data.csv --test-data my_test_data.csv
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
    """Main function to combine features."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Combine features from engineered features, original data, and test data.')
    parser.add_argument('--output', type=str, default='features_combined.csv', help='Path to the output file')
    parser.add_argument('--eng-features', type=str, default='eng_features.csv', help='Path to the engineered features file')
    parser.add_argument('--original-data', type=str, default='train_probe.csv', help='Path to the original data file')
    parser.add_argument('--test-data', type=str, default='/raid/sroziewski/chess/testing_data_cropped.csv', help='Path to the test data file')
    args = parser.parse_args()

    output_path = args.output
    eng_features_path = args.eng_features
    original_data_path = args.original_data
    test_data_path = args.test_data

    # Load engineered features
    logger.info(f"Loading engineered features from {eng_features_path}")
    try:
        final_features = pd.read_csv(eng_features_path, index_col='idx')
        logger.info(f"Loaded engineered features with shape: {final_features.shape}")
    except Exception as e:
        logger.error(f"Error loading engineered features: {str(e)}")
        return

    # Load original data
    logger.info(f"Loading original data from {original_data_path}")
    try:
        original_data = pd.read_csv(original_data_path)
        logger.info(f"Loaded original data with shape: {original_data.shape}")
    except Exception as e:
        logger.error(f"Error loading original data: {str(e)}")
        return

    # Load test data
    logger.info(f"Loading test data from {test_data_path}")
    try:
        test_data = pd.read_csv(test_data_path)
        logger.info(f"Loaded test data with shape: {test_data.shape}")
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        return

    # Extract required columns from original data
    logger.info("Extracting required columns from original data")
    # Get all success_prob_* columns from original data
    success_prob_cols = [col for col in original_data.columns if col.startswith('success_prob_')]
    # Add other required columns
    required_cols = ['PuzzleId', 'RatingDeviation', 'Popularity', 'NbPlays'] + success_prob_cols

    # Extract the required columns from original data
    original_features = original_data[required_cols].copy()
    # Add is_train column to original data (1 = train)
    original_features['is_train'] = 1
    logger.info(f"Extracted {len(required_cols)} columns from original data and added is_train column")

    # Extract required columns from test data
    logger.info("Extracting required columns from test data")
    # Determine which required columns exist in test data
    test_required_cols = ['PuzzleId'] + [col for col in required_cols[1:] if col in test_data.columns]

    # Extract the required columns from test data
    test_features = test_data[test_required_cols].copy()
    # Add is_train column to test data (0 = test)
    test_features['is_train'] = 0
    logger.info(f"Extracted {len(test_required_cols)} columns from test data and added is_train column")

    # Combine original and test features
    logger.info("Combining original and test features")
    combined_original_features = pd.concat([original_features, test_features], ignore_index=True)
    logger.info(f"Combined original and test features with shape: {combined_original_features.shape}")
    logger.info(f"Train samples: {sum(combined_original_features['is_train'] == 1)}, Test samples: {sum(combined_original_features['is_train'] == 0)}")

    # Combine the datasets
    logger.info("Combining engineered features with original and test features")

    # Reset index of engineered features to get numeric index as a column
    final_features_reset = final_features.reset_index()

    # Create a new column in final_features with PuzzleId from combined_original_features
    # We'll map the numeric index to the corresponding PuzzleId
    # Since both DataFrames have the same number of rows and are in the same order,
    # we can use the row position to map between them
    if len(final_features_reset) == len(combined_original_features):
        logger.info("Both datasets have the same number of rows, creating mapping based on row position")
        final_features_reset['PuzzleId'] = combined_original_features['PuzzleId'].values
    else:
        logger.warning(f"Datasets have different number of rows: {len(final_features_reset)} vs {len(combined_original_features)}")
        # If datasets have different number of rows, we'll use a subset of the combined data
        # that matches the number of rows in the engineered features
        if len(final_features_reset) <= len(combined_original_features):
            final_features_reset['PuzzleId'] = combined_original_features['PuzzleId'].values[:len(final_features_reset)]
        else:
            # If engineered features has more rows, we'll pad the PuzzleId with generated values
            puzzle_ids = list(combined_original_features['PuzzleId'].values)
            for i in range(len(combined_original_features), len(final_features_reset)):
                puzzle_ids.append(f"generated_{i}")
            final_features_reset['PuzzleId'] = puzzle_ids

    # Set PuzzleId as index for both DataFrames
    final_features_reset.set_index('PuzzleId', inplace=True)
    combined_original_features.set_index('PuzzleId', inplace=True)

    # Check for duplicate column names
    duplicate_cols = set(final_features_reset.columns).intersection(set(combined_original_features.columns))
    if duplicate_cols:
        logger.warning(f"Found {len(duplicate_cols)} duplicate column names: {duplicate_cols}")
        # Rename duplicate columns in combined_original_features to avoid conflicts
        combined_original_features = combined_original_features.rename(columns={col: f"{col}_original" for col in duplicate_cols})

    # Combine the features using an inner join to keep only rows that exist in both DataFrames
    combined_features = pd.concat([final_features_reset, combined_original_features], axis=1, join='inner')
    logger.info(f"Combined features shape: {combined_features.shape}")

    # Save the combined dataset
    logger.info(f"Saving combined features to {output_path}")
    try:
        combined_features.to_csv(output_path)
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
