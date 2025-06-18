#!/usr/bin/env python
"""
Test script to verify that the 'is_train' column is preserved in the output of engineer_chess_theme_features.
"""

import pandas as pd
import numpy as np
import logging
from chess_puzzle_rating.features.theme_features import engineer_chess_theme_features

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the preservation of the 'is_train' column in engineer_chess_theme_features."""
    logger.info("Creating test data...")
    
    # Create a simple test DataFrame with a Themes column and is_train column
    data = {
        'PuzzleId': [f'puzzle_{i}' for i in range(10)],
        'Themes': ['mate middlegame opening', 'fork pin', 'skewer', 'discoveredattack', 
                  'sacrifice', 'promotion', 'endgame', 'middlegame', 'opening', 'mate'],
        'is_train': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 5 train, 5 test
    }
    df = pd.DataFrame(data)
    df.set_index('PuzzleId', inplace=True)
    
    logger.info(f"Test DataFrame shape: {df.shape}")
    logger.info(f"Test DataFrame columns: {df.columns.tolist()}")
    logger.info(f"is_train values: {df['is_train'].value_counts().to_dict()}")
    
    # Call engineer_chess_theme_features
    logger.info("Calling engineer_chess_theme_features...")
    theme_features = engineer_chess_theme_features(df)
    
    logger.info(f"Output DataFrame shape: {theme_features.shape}")
    logger.info(f"Output DataFrame columns: {theme_features.columns.tolist()}")
    
    # Check if is_train column is preserved
    if 'is_train' in theme_features.columns:
        logger.info("SUCCESS: 'is_train' column is preserved in the output")
        logger.info(f"is_train values in output: {theme_features['is_train'].value_counts().to_dict()}")
        
        # Verify that the values are correct
        original_train_count = df['is_train'].sum()
        output_train_count = theme_features['is_train'].sum()
        
        if original_train_count == output_train_count:
            logger.info(f"SUCCESS: Train count matches: {original_train_count}")
        else:
            logger.error(f"ERROR: Train count mismatch: original={original_train_count}, output={output_train_count}")
    else:
        logger.error("ERROR: 'is_train' column is NOT preserved in the output")
    
    logger.info("Test completed")

if __name__ == "__main__":
    main()