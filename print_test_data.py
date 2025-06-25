#!/usr/bin/env python3
"""
Script to open final_dataset_engine_with_matches.csv and print rows where is_train = 0.
"""

import pandas as pd

def main():
    # Load the dataset
    try:
        print("Loading final_dataset_engine_with_matches.csv...")
        df = pd.read_csv('final_dataset_engine_with_matches.csv')
        print(f"Dataset loaded. Total rows: {len(df)}")
        
        # Filter rows where is_train = 0
        test_df = df[df['is_train'] == 0]
        print(f"Number of rows with is_train = 0: {len(test_df)}")
        
        # Print the filtered rows
        if len(test_df) > 0:
            print("\nRows with is_train = 0:")
            print(test_df)
        else:
            print("\nNo rows found with is_train = 0.")
            
    except FileNotFoundError:
        print("Error: final_dataset_engine_with_matches.csv not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()