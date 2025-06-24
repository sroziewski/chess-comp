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
import concurrent.futures
import multiprocessing
import math
from datetime import datetime
from typing import List, Dict, Tuple, Any
from chess_puzzle_rating.utils.progress import track_progress, get_logger

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_chunk(chunk_df: pd.DataFrame, training_ratings: Dict[Any, float]) -> Tuple[pd.DataFrame, Dict[Any, Tuple[float, float]], int]:
    """
    Process a chunk of the dataframe to swap Rating values.

    Parameters
    ----------
    chunk_df : pd.DataFrame
        A chunk of the dataframe to process
    training_ratings : Dict[Any, float]
        Dictionary mapping PuzzleId to Rating from training data

    Returns
    -------
    Tuple[pd.DataFrame, Dict[Any, Tuple[float, float]], int]
        Processed dataframe chunk, dictionary of swapped ratings, and count of updated rows
    """
    chunk_swapped_ratings = {}
    chunk_rows_updated = 0

    # Create a copy of the chunk to modify
    chunk_result = chunk_df.copy()

    # For each row in the chunk, check if PuzzleId exists in training data
    for idx, row in chunk_df.iterrows():
        puzzle_id = row['PuzzleId']

        # Check if puzzle_id exists in training_ratings dictionary
        if puzzle_id in training_ratings:
            chunk_rows_updated += 1
            # Get the Rating from training data
            training_rating = training_ratings[puzzle_id]
            # Store the original Rating from final dataset
            final_rating = row['Rating']
            # Swap the Rating values
            chunk_result.at[idx, 'Rating'] = training_rating
            # Store the swapped rating for logging
            chunk_swapped_ratings[puzzle_id] = (final_rating, training_rating)

    return chunk_result, chunk_swapped_ratings, chunk_rows_updated

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

    # Create a dictionary mapping PuzzleId to Rating from training data for faster lookups
    logger.info("Creating PuzzleId to Rating mapping from training data")
    training_ratings = dict(zip(training_df['PuzzleId'], training_df['Rating']))

    # Determine the number of worker processes
    max_workers = multiprocessing.cpu_count()
    logger.info(f"Using {max_workers} worker processes for parallel processing")

    # Calculate chunk size for parallel processing
    total_rows = len(corrected_df)
    # Aim for at least 10 chunks per worker for better load balancing
    chunk_size = max(1, math.ceil(total_rows / (max_workers * 10)))
    logger.info(f"Using chunk size of {chunk_size} rows")


    # Split the dataframe into chunks
    logger.info("Splitting dataframe into chunks for parallel processing")
    chunks = [corrected_df.iloc[i:i+chunk_size] for i in range(0, len(corrected_df), chunk_size)]
    logger.info(f"Created {len(chunks)} chunks")

    # Process chunks in parallel with progress tracking
    logger.info("Swapping Rating values where PuzzleId matches (in parallel)")

    # Initialize variables to store results
    processed_chunks = []
    swapped_ratings = {}
    rows_to_update = 0

    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for processing
        futures = [executor.submit(process_chunk, chunk, training_ratings) for chunk in chunks]

        # Process results as they complete with progress tracking
        for future in track_progress(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            description="Processing chunks",
            logger=logger
        ):
            try:
                # Get the result from the future
                chunk_result, chunk_swapped_ratings, chunk_rows_updated = future.result()

                # Add the processed chunk to the list
                processed_chunks.append(chunk_result)

                # Update the swapped ratings dictionary
                swapped_ratings.update(chunk_swapped_ratings)

                # Update the count of rows updated
                rows_to_update += chunk_rows_updated

            except Exception as e:
                logger.error(f"Error processing chunk: {e}")

    # Combine the processed chunks back into a single dataframe
    logger.info("Combining processed chunks")
    corrected_df = pd.concat(processed_chunks)

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
