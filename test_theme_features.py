#!/usr/bin/env python
"""
Test script for extract_theme_features.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Import the extract_theme_features module
from extract_theme_features import extract_theme_features
from chess_puzzle_rating.utils.progress import setup_logging, get_logger

def create_sample_data(n_samples=100, output_file='sample_data.csv'):
    """
    Create a sample dataset for testing theme feature extraction.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to create, by default 100
    output_file : str, optional
        Path to save the sample data, by default 'sample_data.csv'

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the sample data
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create sample themes
    themes = [
        'mate', 'fork', 'pin', 'skewer', 'discoveredAttack', 'sacrifice',
        'promotion', 'endgame', 'middlegame', 'opening', 'queenside',
        'kingside', 'attack', 'defense', 'tactical', 'strategic'
    ]

    # Create sample data
    data = []
    for i in range(n_samples):
        # Randomly select 0-3 themes for each sample
        n_themes = np.random.randint(0, 4)
        sample_themes = np.random.choice(themes, size=n_themes, replace=False)

        # Create a row with the selected themes
        row = {
            'PuzzleId': f'puzzle_{i}',
            'Rating': np.random.randint(800, 2200),
            'Themes': ' '.join(sample_themes)
        }
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV if output_file is specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Saved sample data to {output_file}")

    return df

def main():
    """Main function to test theme feature extraction."""
    # Set up logging
    log_file = f"test_theme_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file)
    logger = get_logger()

    logger.info("Starting theme feature extraction test")

    try:
        # Create sample data
        sample_file = 'sample_data.csv'
        df = create_sample_data(n_samples=100, output_file=sample_file)
        logger.info(f"Created sample data with {len(df)} rows")

        # Extract theme features
        output_file = 'theme_features.csv'
        theme_features = extract_theme_features(
            df,
            theme_column='Themes',
            min_theme_freq=1,  # Lower threshold for test data
            max_themes=20,
            n_svd_components=5,
            n_hash_features=10,
            output_file=output_file
        )

        # Print feature statistics
        logger.info(f"Extracted {theme_features.shape[1]} theme features")
        logger.info(f"Feature columns: {', '.join(theme_features.columns[:10])}...")

        # Load the saved features and verify
        if os.path.exists(output_file):
            loaded_features = pd.read_csv(output_file, index_col=0)
            logger.info(f"Loaded {loaded_features.shape[1]} features from {output_file}")

            # Verify that the loaded features match the extracted features
            assert loaded_features.shape == theme_features.shape, "Shape mismatch between extracted and loaded features"
            logger.info("Verification successful: Extracted and loaded features have the same shape")

        logger.info("Theme feature extraction test completed successfully")

    except Exception as e:
        logger.error(f"Error during theme feature extraction test: {str(e)}")
        raise
    finally:
        # Clean up sample files
        for file in [sample_file, output_file]:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Removed test file: {file}")

if __name__ == "__main__":
    main()
