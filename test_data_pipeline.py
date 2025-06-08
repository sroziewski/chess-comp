"""
Test script for the data pipeline.

This script tests the data pipeline by running it and printing information
about the resulting datasets.
"""

import os
import pandas as pd
import numpy as np
from chess_puzzle_rating.data.pipeline import run_data_pipeline

def main():
    """Run the data pipeline and print information about the results."""
    print("Testing the data pipeline...")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    # Run the pipeline
    try:
        X_train, X_test, y_train, test_ids = run_data_pipeline()
        
        # Print information about the results
        print("\nPipeline completed successfully!")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"test_ids shape: {test_ids.shape}")
        
        # Print some sample data
        print("\nSample of X_train features:")
        print(X_train.head())
        
        print("\nSample of y_train values:")
        print(y_train.head())
        
        print("\nSample of test_ids:")
        print(test_ids.head())
        
        # Print feature statistics
        print("\nFeature statistics (X_train):")
        print(X_train.describe().T[['count', 'mean', 'min', 'max']].head(10))
        
        # Check for any remaining NaN values
        train_nan_count = X_train.isna().sum().sum()
        test_nan_count = X_test.isna().sum().sum()
        print(f"\nNaN values in X_train: {train_nan_count}")
        print(f"NaN values in X_test: {test_nan_count}")
        
        # List checkpoint files
        print("\nCheckpoint files created:")
        checkpoint_files = [f for f in os.listdir("checkpoints") if f.endswith(".parquet")]
        for i, file in enumerate(checkpoint_files[:10]):  # Show first 10 files
            print(f"  {i+1}. {file}")
        if len(checkpoint_files) > 10:
            print(f"  ... and {len(checkpoint_files) - 10} more files")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error running the pipeline: {e}")
        raise

if __name__ == "__main__":
    main()