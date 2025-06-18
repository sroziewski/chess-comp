#!/usr/bin/env python
"""
Script to merge final features with combined features.

This script:
1. Loads final features from final_features_latest.csv
2. Loads combined features from features_combined.csv
3. Merges the two dataframes
4. Saves the result to a new file

Usage:
    python merge_final_features.py [--output OUTPUT_FILE] [--final-features FINAL_FEATURES_FILE] 
                                  [--combined-features COMBINED_FEATURES_FILE]

Arguments:
    --output: Path to the output file (default: final_merged_features.csv)
    --final-features: Path to the final features file (default: final_features_latest.csv)
    --combined-features: Path to the combined features file (default: features_combined.csv)

Example:
    python merge_final_features.py --output merged_features.csv --final-features my_final_features.csv --combined-features my_combined_features.csv
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
    """Main function to merge features."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Merge final features with combined features.')
    parser.add_argument('--output', type=str, default='final_merged_features.csv', help='Path to the output file')
    parser.add_argument('--final-features', type=str, default='final_features_latest.csv', help='Path to the final features file')
    parser.add_argument('--combined-features', type=str, default='features_combined.csv', help='Path to the combined features file')
    args = parser.parse_args()

    output_path = args.output
    final_features_path = args.final_features
    combined_features_path = args.combined_features

    # Load final features
    logger.info(f"Loading final features from {final_features_path}")
    try:
        final_features = pd.read_csv(final_features_path)
        logger.info(f"Loaded final features with shape: {final_features.shape}")
    except Exception as e:
        logger.error(f"Error loading final features: {str(e)}")
        return

    # Load combined features
    logger.info(f"Loading combined features from {combined_features_path}")
    try:
        combined_features = pd.read_csv(combined_features_path)
        logger.info(f"Loaded combined features with shape: {combined_features.shape}")
    except Exception as e:
        logger.error(f"Error loading combined features: {str(e)}")
        return

    # Check if PuzzleId is in combined features dataframe
    if 'PuzzleId' not in combined_features.columns:
        logger.error("PuzzleId column not found in combined features")
        return

    # Check if PuzzleId is in final features dataframe
    if 'PuzzleId' not in final_features.columns:
        logger.info("PuzzleId column not found in final features. Using numeric index (idx) instead.")
        # Add a PuzzleId column based on the index (starting from 0)
        final_features['PuzzleId'] = [f"idx_{i}" for i in range(len(final_features))]

    # Set PuzzleId as index for both dataframes
    final_features.set_index('PuzzleId', inplace=True)
    combined_features.set_index('PuzzleId', inplace=True)

    # Check for duplicate column names
    duplicate_cols = set(final_features.columns).intersection(set(combined_features.columns))
    if duplicate_cols:
        logger.warning(f"Found {len(duplicate_cols)} duplicate column names: {duplicate_cols}")
        # Rename duplicate columns in combined_features to avoid conflicts
        combined_features = combined_features.rename(columns={col: f"{col}_combined" for col in duplicate_cols})
        logger.info(f"Renamed duplicate columns in combined_features")

    # Check for duplicate index values in each dataframe
    if final_features.index.duplicated().any():
        logger.warning("Found duplicate PuzzleId values in final_features index. Making index unique.")
        # Make the index unique by adding a suffix to duplicate values
        final_features = final_features.reset_index()
        final_features['PuzzleId'] = final_features['PuzzleId'].astype(str)
        # Create a suffix based on the cumulative count of each PuzzleId
        final_features['suffix'] = final_features.groupby('PuzzleId').cumcount().apply(
            lambda x: f"_{x}" if x > 0 else "")
        # Add the suffix to the PuzzleId
        final_features['PuzzleId'] = final_features['PuzzleId'] + final_features['suffix']
        # Drop the suffix column
        final_features = final_features.drop('suffix', axis=1)
        # Set PuzzleId as the index again
        final_features = final_features.set_index('PuzzleId')

    if combined_features.index.duplicated().any():
        logger.warning("Found duplicate PuzzleId values in combined_features index. Making index unique.")
        # Make the index unique by adding a suffix to duplicate values
        combined_features = combined_features.reset_index()
        combined_features['PuzzleId'] = combined_features['PuzzleId'].astype(str)
        # Create a suffix based on the cumulative count of each PuzzleId
        combined_features['suffix'] = combined_features.groupby('PuzzleId').cumcount().apply(
            lambda x: f"_{x}" if x > 0 else "")
        # Add the suffix to the PuzzleId
        combined_features['PuzzleId'] = combined_features['PuzzleId'] + combined_features['suffix']
        # Drop the suffix column
        combined_features = combined_features.drop('suffix', axis=1)
        # Set PuzzleId as the index again
        combined_features = combined_features.set_index('PuzzleId')

    # Merge the dataframes
    logger.info("Merging final features with combined features")
    try:
        # Use inner join to ensure the number of rows stays the same
        merged_features = pd.concat([final_features, combined_features], axis=1, join='inner')
        logger.info(f"Merged features shape: {merged_features.shape}")

        # Check for missing values after merge
        missing_values = merged_features.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values after merge")
            # Fill missing values with 0
            merged_features = merged_features.fillna(0)
            logger.info("Filled missing values with 0")
    except Exception as e:
        logger.error(f"Error merging features: {str(e)}")
        return

    # Save the merged dataset
    logger.info(f"Saving merged features to {output_path}")
    try:
        merged_features.to_csv(output_path)
        logger.info(f"Successfully saved merged features to {output_path}")
    except Exception as e:
        logger.error(f"Error saving merged features: {str(e)}")

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Script started at {start_time}")

    main()

    end_time = datetime.now()
    logger.info(f"Script completed at {end_time}")
    logger.info(f"Total execution time: {end_time - start_time}")
