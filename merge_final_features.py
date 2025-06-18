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

    # Check for duplicate column names
    duplicate_cols = set(final_features.columns).intersection(set(combined_features.columns))
    if duplicate_cols and 'PuzzleId' in duplicate_cols:
        # Remove PuzzleId from duplicate_cols as we'll use it for merging
        duplicate_cols.remove('PuzzleId')

    if duplicate_cols:
        logger.warning(f"Found {len(duplicate_cols)} duplicate column names: {duplicate_cols}")
        # Rename duplicate columns in combined_features to avoid conflicts
        combined_features = combined_features.rename(columns={col: f"{col}_combined" for col in duplicate_cols})
        logger.info(f"Renamed duplicate columns in combined_features")

    # Check for duplicate PuzzleId values
    if final_features['PuzzleId'].duplicated().any():
        logger.warning("Found duplicate PuzzleId values in final_features. This may cause issues with merging.")

    if combined_features['PuzzleId'].duplicated().any():
        logger.warning("Found duplicate PuzzleId values in combined_features. This may cause issues with merging.")

    # Merge the dataframes on PuzzleId column
    logger.info("Merging final features with combined features on PuzzleId")
    try:
        # Use merge instead of concat to ensure proper matching on PuzzleId
        merged_features = pd.merge(final_features, combined_features, on='PuzzleId', how='inner')
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
