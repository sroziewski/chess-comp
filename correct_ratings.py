#!/usr/bin/env python
"""
Script to correct Rating values in final_dataset.csv based on training_data_02_01.csv.

This script:
1. Opens final_dataset.csv and /raid/sroziewski/chess/training_data_02_01.csv
2. For each row (except headers), checks if PuzzleId values match
3. If they match, swaps the Rating value from the training data into the final dataset
4. Saves the modified data as final_dataset_corrected.csv

Usage:
    python correct_ratings.py

"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to correct Rating values in final_dataset.csv."""
    # Define paths to data files
    final_dataset_path = "final_dataset.csv"
    training_data_path = "/raid/sroziewski/chess/training_data_02_01.csv"
    output_path = "final_dataset_corrected.csv"

    # Check if data files exist
    files = {
        "final_dataset": final_dataset_path,
        "training_data": training_data_path
    }

    missing_files = []
    for file_name, file_path in files.items():
        if not os.path.exists(file_path):
            logger.warning(f"File {file_name} not found at {file_path}")
            missing_files.append(file_name)

    if missing_files:
        logger.error(f"The following data files are missing: {', '.join(missing_files)}")
        logger.error("Cannot proceed without both final dataset and training data files.")
        return

    # Load final dataset
    logger.info(f"Loading final dataset from {final_dataset_path}")
    try:
        final_df = pd.read_csv(final_dataset_path)
        logger.info(f"Loaded final dataset with shape: {final_df.shape}")
    except Exception as e:
        logger.error(f"Error loading final dataset: {str(e)}")
        return

    # Load training data
    logger.info(f"Loading training data from {training_data_path}")
    try:
        training_df = pd.read_csv(training_data_path)
        logger.info(f"Loaded training data with shape: {training_df.shape}")
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        return

    # Check if 'PuzzleId' column exists in both dataframes
    if 'PuzzleId' not in final_df.columns:
        logger.error("'PuzzleId' column not found in final dataset")
        return
    if 'PuzzleId' not in training_df.columns:
        logger.error("'PuzzleId' column not found in training data")
        return

    # Check if 'Rating' column exists in both dataframes
    if 'Rating' not in final_df.columns:
        logger.error("'Rating' column not found in final dataset")
        return
    if 'Rating' not in training_df.columns:
        logger.error("'Rating' column not found in training data")
        return

    # Create a copy of the final dataset to modify
    corrected_df = final_df.copy()

    # Count how many PuzzleIds from final dataset are found in training data
    common_puzzles = set(final_df['PuzzleId']).intersection(set(training_df['PuzzleId']))
    logger.info(f"Found {len(common_puzzles)} common PuzzleIds between datasets")

    # Swap Rating values where PuzzleId matches
    logger.info("Swapping Rating values where PuzzleId matches")

    # Create a dictionary to store the swapped ratings
    swapped_ratings = {}

    # Count how many rows will be updated
    rows_to_update = 0

    # For each row in final dataset, check if PuzzleId exists in training data
    for idx, row in corrected_df.iterrows():
        puzzle_id = row['PuzzleId']
        # Find matching rows in training data
        training_matches = training_df[training_df['PuzzleId'] == puzzle_id]

        # If there's a match, swap the Rating values
        if not training_matches.empty:
            rows_to_update += 1
            # Get the Rating from training data
            training_rating = training_matches.iloc[0]['Rating']
            # Store the original Rating from final dataset
            final_rating = row['Rating']
            # Swap the Rating values
            corrected_df.at[idx, 'Rating'] = training_rating
            # Store the swapped rating for logging
            swapped_ratings[puzzle_id] = (final_rating, training_rating)

    logger.info(f"Swapped Rating for {rows_to_update} rows")

    # Log some statistics about the changes
    if rows_to_update > 0:
        original_ratings = [original for original, _ in swapped_ratings.values()]
        new_ratings = [new for _, new in swapped_ratings.values()]
        logger.info(f"Original Rating range: {min(original_ratings)} to {max(original_ratings)}")
        logger.info(f"New Rating range: {min(new_ratings)} to {max(new_ratings)}")

    # Save the corrected dataset
    logger.info(f"Saving corrected data to {output_path}")
    try:
        corrected_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved corrected data to {output_path}")
    except Exception as e:
        logger.error(f"Error saving corrected data: {str(e)}")

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Script started at {start_time}")

    main()

    end_time = datetime.now()
    logger.info(f"Script completed at {end_time}")
    logger.info(f"Total execution time: {end_time - start_time}")
