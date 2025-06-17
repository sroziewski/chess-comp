#!/usr/bin/env python
"""
Script to extract theme features from chess puzzles and save them separately.

Usage:
    extract_theme_features.py [-h] (--input INPUT | --input-train INPUT_TRAIN) [--input-test INPUT_TEST] 
                              --output OUTPUT [--theme-column THEME_COLUMN] [--min-theme-freq MIN_THEME_FREQ] 
                              [--max-themes MAX_THEMES] [--n-svd-components N_SVD_COMPONENTS]
                              [--n-hash-features N_HASH_FEATURES] [--log-file LOG_FILE]

The script can process either a single input file (--input) or separate train and test files 
(--input-train and --input-test). When using separate files, they will be merged before feature 
extraction to ensure consistent feature engineering across both datasets.
"""

import os
import pandas as pd
import argparse
import logging
from datetime import datetime

from chess_puzzle_rating.features.theme_features import engineer_chess_theme_features
from chess_puzzle_rating.utils.progress import setup_logging, get_logger

def load_data(input_file):
    """
    Load input data from a CSV file.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the input data
    """
    logger = get_logger()
    logger.info(f"Loading data from {input_file}")

    # Check if file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} rows from {input_file}")

    return df

def load_and_merge_data(input_train_file, input_test_file=None):
    """
    Load and merge train and test data from CSV files.

    Parameters
    ----------
    input_train_file : str
        Path to the training data CSV file
    input_test_file : str, optional
        Path to the test data CSV file, by default None

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the merged data
    """
    logger = get_logger()

    # Load training data
    train_df = load_data(input_train_file)
    train_df['is_train'] = 1

    # If test file is provided, load and merge with train data
    if input_test_file:
        test_df = load_data(input_test_file)
        test_df['is_train'] = 0

        # Merge train and test data
        logger.info(f"Merging {len(train_df)} training rows and {len(test_df)} test rows")
        merged_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
        logger.info(f"Merged dataset has {len(merged_df)} rows")
        return merged_df
    else:
        # If no test file, just return train data
        logger.info(f"No test file provided, using only training data with {len(train_df)} rows")
        return train_df

def extract_theme_features(df, theme_column='Themes', min_theme_freq=5, max_themes=100, 
                          n_svd_components=10, n_hash_features=15, output_file=None):
    """
    Extract theme features from the input DataFrame and save them to a file.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzle data
    theme_column : str, optional
        Column name containing themes, by default 'Themes'
    min_theme_freq : int, optional
        Minimum frequency for a theme to be one-hot encoded, by default 5
    max_themes : int, optional
        Max number of most frequent themes to one-hot encode, by default 100
    n_svd_components : int, optional
        Number of SVD components for theme embeddings, by default 10
    n_hash_features : int, optional
        Number of features to use for hashing vectorizer, by default 15
    output_file : str, optional
        Path to save the output features, by default None

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the extracted theme features
    """
    logger = get_logger()
    logger.info("Extracting theme features...")

    # Extract theme features
    theme_features = engineer_chess_theme_features(
        df,
        theme_column=theme_column,
        min_theme_freq=min_theme_freq,
        max_themes=max_themes,
        n_svd_components=n_svd_components,
        n_hash_features=n_hash_features
    )

    logger.info(f"Extracted {theme_features.shape[1]} theme features from {len(df)} puzzles")

    # Save features if output file is specified
    if output_file:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save features
        theme_features.to_csv(output_file)
        logger.info(f"Saved theme features to {output_file}")

    return theme_features

def main():
    """Main function to run the script."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract theme features from chess puzzles')

    # Input file options - either use --input or both --input-train and --input-test
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', help='Path to input CSV file (deprecated, use --input-train and --input-test instead)')
    input_group.add_argument('--input-train', help='Path to training data CSV file')

    parser.add_argument('--input-test', help='Path to test data CSV file (optional)')
    parser.add_argument('--output', '-o', required=True, help='Path to output CSV file')
    parser.add_argument('--theme-column', default='Themes', help='Column name containing themes')
    parser.add_argument('--min-theme-freq', type=int, default=5, help='Minimum frequency for a theme to be one-hot encoded')
    parser.add_argument('--max-themes', type=int, default=100, help='Max number of most frequent themes to one-hot encode')
    parser.add_argument('--n-svd-components', type=int, default=10, help='Number of SVD components for theme embeddings')
    parser.add_argument('--n-hash-features', type=int, default=15, help='Number of features to use for hashing vectorizer')
    parser.add_argument('--log-file', help='Path to log file')

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    log_file = args.log_file or f"theme_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file)
    logger = get_logger()

    logger.info("Starting theme feature extraction")

    try:
        # Load data - either from a single input file or from train and test files
        if args.input:
            logger.warning("Using --input is deprecated. Please use --input-train and --input-test instead.")
            df = load_data(args.input)
        else:
            # Load and merge train and test data
            df = load_and_merge_data(args.input_train, args.input_test)

        # Extract theme features
        theme_features = extract_theme_features(
            df,
            theme_column=args.theme_column,
            min_theme_freq=args.min_theme_freq,
            max_themes=args.max_themes,
            n_svd_components=args.n_svd_components,
            n_hash_features=args.n_hash_features,
            output_file=args.output
        )

        logger.info("Theme feature extraction completed successfully")

    except Exception as e:
        logger.error(f"Error during theme feature extraction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
