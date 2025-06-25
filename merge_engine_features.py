#!/usr/bin/env python3
"""
Merge Engine Features Script

This script reads final_dataset.csv and engine_features_dataset.csv, adds columns
from engine_features_dataset.csv to final_dataset.csv based on matching 'FEN' values,
and saves the result to final_dataset_engine.csv.

Usage:
    python merge_engine_features.py [--final FINAL_DATASET] [--engine ENGINE_DATASET] [--output OUTPUT_FILE] [--chunks CHUNK_SIZE]

Default paths:
    --final: final_dataset.csv
    --engine: engine_features_dataset.csv
    --output: final_dataset_engine.csv
    --chunks: 0 (process entire dataset at once)
"""

import pandas as pd
import os
import sys
import argparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_entire_dataset(final_dataset_file, engine_features_file, output_file):
    """Process the entire dataset at once."""
    try:
        # Read the CSV files
        logger.info(f"Reading {final_dataset_file}...")
        final_df = pd.read_csv(final_dataset_file)
        logger.info(f"Shape of final dataset: {final_df.shape}")
        
        logger.info(f"Reading {engine_features_file}...")
        engine_df = pd.read_csv(engine_features_file)
        logger.info(f"Shape of engine features dataset: {engine_df.shape}")

        # Check if 'FEN' column exists in both dataframes
        if 'FEN' not in final_df.columns:
            logger.error("'FEN' column not found in final dataset")
            return 1
        if 'FEN' not in engine_df.columns:
            logger.error("'FEN' column not found in engine features dataset")
            return 1

        # Get the list of columns to add from engine_df (excluding 'FEN' which is used for matching)
        engine_columns = [col for col in engine_df.columns if col != 'FEN']
        logger.info(f"Number of engine feature columns to add: {len(engine_columns)}")
        
        # Check for duplicate column names
        duplicate_cols = set(final_df.columns).intersection(set(engine_columns))
        if duplicate_cols:
            logger.warning(f"Found {len(duplicate_cols)} duplicate column names: {duplicate_cols}")
            # Rename duplicate columns in engine_df to avoid conflicts
            rename_dict = {col: f"engine_{col}" for col in duplicate_cols}
            engine_df = engine_df.rename(columns=rename_dict)
            # Update engine_columns list with new names
            engine_columns = [rename_dict.get(col, col) for col in engine_columns]
            logger.info(f"Renamed duplicate columns in engine features dataset")

        # Merge the dataframes on 'FEN' column
        logger.info("Merging dataframes on 'FEN' column...")
        # Use left join to keep all rows from final_df
        merged_df = pd.merge(final_df, engine_df, on='FEN', how='left')
        
        # Check how many rows had matching FEN values
        if engine_columns:
            matched_rows = merged_df[engine_columns].notna().any(axis=1).sum()
            match_percentage = (matched_rows / len(final_df)) * 100
            logger.info(f"Matched {matched_rows} out of {len(final_df)} rows ({match_percentage:.2f}%) based on 'FEN' column")
            
            if matched_rows == 0:
                logger.warning("No rows were matched between the datasets. The output will be the same as the input final dataset.")
            elif matched_rows < len(final_df) * 0.1:  # Less than 10% matched
                logger.warning(f"Only {match_percentage:.2f}% of rows were matched. Please check if the 'FEN' values are in the same format in both datasets.")
        else:
            logger.warning("No engine feature columns to add (after handling duplicates).")

        # Save the merged dataframe
        logger.info(f"Saving merged dataframe to {output_file}...")
        merged_df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved merged dataframe to {output_file}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in process_entire_dataset: {str(e)}")
        return 1

def process_in_chunks(final_dataset_file, engine_features_file, output_file, chunk_size):
    """Process the dataset in chunks to handle large files."""
    try:
        # First, read the engine features dataset (usually smaller) into memory
        logger.info(f"Reading {engine_features_file}...")
        engine_df = pd.read_csv(engine_features_file)
        logger.info(f"Shape of engine features dataset: {engine_df.shape}")
        
        # Check if 'FEN' column exists in engine_df
        if 'FEN' not in engine_df.columns:
            logger.error("'FEN' column not found in engine features dataset")
            return 1
        
        # Get the list of columns to add from engine_df (excluding 'FEN' which is used for matching)
        engine_columns = [col for col in engine_df.columns if col != 'FEN']
        logger.info(f"Number of engine feature columns to add: {len(engine_columns)}")
        
        # Initialize variables for tracking progress
        total_rows = 0
        matched_rows = 0
        first_chunk = True
        
        # Process the final dataset in chunks
        logger.info(f"Processing {final_dataset_file} in chunks of size {chunk_size}...")
        
        # Get the total number of rows in the final dataset for progress reporting
        with pd.read_csv(final_dataset_file, chunksize=chunk_size) as reader:
            for chunk_num, chunk in enumerate(reader, 1):
                total_rows += len(chunk)
        
        # Process the final dataset in chunks
        with pd.read_csv(final_dataset_file, chunksize=chunk_size) as reader:
            for chunk_num, chunk in enumerate(reader, 1):
                logger.info(f"Processing chunk {chunk_num} ({len(chunk)} rows)...")
                
                # Check if 'FEN' column exists in the chunk
                if 'FEN' not in chunk.columns:
                    logger.error("'FEN' column not found in final dataset")
                    return 1
                
                # Check for duplicate column names in the first chunk
                if first_chunk:
                    duplicate_cols = set(chunk.columns).intersection(set(engine_columns))
                    if duplicate_cols:
                        logger.warning(f"Found {len(duplicate_cols)} duplicate column names: {duplicate_cols}")
                        # Rename duplicate columns in engine_df to avoid conflicts
                        rename_dict = {col: f"engine_{col}" for col in duplicate_cols}
                        engine_df = engine_df.rename(columns=rename_dict)
                        # Update engine_columns list with new names
                        engine_columns = [rename_dict.get(col, col) for col in engine_columns]
                        logger.info(f"Renamed duplicate columns in engine features dataset")
                    first_chunk = False
                
                # Merge the chunk with engine_df on 'FEN' column
                merged_chunk = pd.merge(chunk, engine_df, on='FEN', how='left')
                
                # Count matched rows in this chunk
                if engine_columns:
                    chunk_matched = merged_chunk[engine_columns].notna().any(axis=1).sum()
                    matched_rows += chunk_matched
                    logger.info(f"Matched {chunk_matched} out of {len(chunk)} rows in chunk {chunk_num}")
                
                # Write the merged chunk to the output file
                mode = 'w' if chunk_num == 1 else 'a'
                header = chunk_num == 1
                merged_chunk.to_csv(output_file, mode=mode, header=header, index=False)
                
                logger.info(f"Processed {len(chunk)} rows in chunk {chunk_num}")
        
        # Report overall matching statistics
        if total_rows > 0 and engine_columns:
            match_percentage = (matched_rows / total_rows) * 100
            logger.info(f"Overall: Matched {matched_rows} out of {total_rows} rows ({match_percentage:.2f}%) based on 'FEN' column")
            
            if matched_rows == 0:
                logger.warning("No rows were matched between the datasets. The output will be the same as the input final dataset.")
            elif matched_rows < total_rows * 0.1:  # Less than 10% matched
                logger.warning(f"Only {match_percentage:.2f}% of rows were matched. Please check if the 'FEN' values are in the same format in both datasets.")
        else:
            logger.warning("No engine feature columns to add (after handling duplicates).")
        
        logger.info(f"Successfully saved merged dataframe to {output_file}")
        return 0
    
    except Exception as e:
        logger.error(f"Error in process_in_chunks: {str(e)}")
        return 1

def main():
    try:
        # Set up command-line argument parsing
        parser = argparse.ArgumentParser(description='Merge engine features into final dataset.')
        parser.add_argument('--final', type=str, default='final_dataset.csv',
                            help='Path to the final dataset CSV file (default: final_dataset.csv)')
        parser.add_argument('--engine', type=str, default='engine_features_dataset.csv',
                            help='Path to the engine features CSV file (default: engine_features_dataset.csv)')
        parser.add_argument('--output', type=str, default='final_dataset_engine.csv',
                            help='Path to the output CSV file (default: final_dataset_engine.csv)')
        parser.add_argument('--chunks', type=int, default=0,
                            help='Process data in chunks of this size (0 means no chunking, default: 0)')

        args = parser.parse_args()

        # Define file paths from arguments
        final_dataset_file = args.final
        engine_features_file = args.engine
        output_file = args.output
        chunk_size = args.chunks

        # Check if input files exist
        for file_path, file_name in [(final_dataset_file, "Final dataset"), (engine_features_file, "Engine features")]:
            if not os.path.exists(file_path):
                logger.error(f"Error: {file_name} file not found: {file_path}")
                return 1

        # Process data based on chunking option
        if chunk_size > 0:
            logger.info(f"Processing data in chunks of size {chunk_size}...")
            return process_in_chunks(final_dataset_file, engine_features_file, output_file, chunk_size)
        else:
            logger.info("Processing entire dataset at once...")
            return process_entire_dataset(final_dataset_file, engine_features_file, output_file)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Script started at {start_time}")
    
    exit_code = main()
    
    end_time = datetime.now()
    logger.info(f"Script completed at {end_time}")
    logger.info(f"Total execution time: {end_time - start_time}")
    
    sys.exit(exit_code)