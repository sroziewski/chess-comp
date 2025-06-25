#!/usr/bin/env python3
"""
Concatenate Engine Features Script

This script reads two CSV files containing chess engine features, concatenates them,
and saves the result to a new CSV file. It handles potential differences in column
order between the input files.

Usage:
    python concat_engine_features.py [--input1 PATH] [--input2 PATH] [--output PATH]

Default paths:
    --input1: /raid/sroziewski/chess/engine_features.csv
    --input2: /raid/sroziewski/chess/engine_features_test.csv
    --output: engine_features_dataset.csv
"""

import pandas as pd
import os
import sys
import argparse

def main():
    try:
        # Set up command-line argument parsing
        parser = argparse.ArgumentParser(description='Concatenate two CSV files containing engine features.')
        parser.add_argument('--input1', type=str, default='/raid/sroziewski/chess/engine_features.csv',
                            help='Path to the first input CSV file (default: /raid/sroziewski/chess/engine_features.csv)')
        parser.add_argument('--input2', type=str, default='/raid/sroziewski/chess/engine_features_test.csv',
                            help='Path to the second input CSV file (default: /raid/sroziewski/chess/engine_features_test.csv)')
        parser.add_argument('--output', type=str, default='engine_features_dataset.csv',
                            help='Path to the output CSV file (default: engine_features_dataset.csv)')

        args = parser.parse_args()

        # Define file paths from arguments
        input_file1 = args.input1
        input_file2 = args.input2
        output_file = args.output

        # Check if input files exist
        for file_path in [input_file1, input_file2]:
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                return 1

        # Read the CSV files
        print(f"Reading {input_file1}...")
        df1 = pd.read_csv(input_file1)
        print(f"Shape of first dataframe: {df1.shape}")
        print(f"Columns in first dataframe: {df1.columns.tolist()}")

        print(f"Reading {input_file2}...")
        df2 = pd.read_csv(input_file2)
        print(f"Shape of second dataframe: {df2.shape}")
        print(f"Columns in second dataframe: {df2.columns.tolist()}")

        # Check if columns are the same (regardless of order)
        if set(df1.columns) != set(df2.columns):
            print("Warning: The two dataframes have different columns.")
            print(f"Columns only in first dataframe: {set(df1.columns) - set(df2.columns)}")
            print(f"Columns only in second dataframe: {set(df2.columns) - set(df1.columns)}")
            print("Will proceed with concatenation, but this might cause issues.")

        # Concatenate the dataframes
        print("Concatenating dataframes...")
        # Ensure both dataframes have the same column order before concatenation
        # This is important because the sample data shows different column orders
        all_columns = list(set(df1.columns) | set(df2.columns))

        # Add any missing columns with NaN values
        for col in all_columns:
            if col not in df1.columns:
                df1[col] = pd.NA
            if col not in df2.columns:
                df2[col] = pd.NA

        # Now concatenate with aligned columns
        combined_df = pd.concat([df1, df2], ignore_index=True)
        print(f"Shape of combined dataframe: {combined_df.shape}")
        print(f"Columns in combined dataframe: {combined_df.columns.tolist()}")

        # Save the combined dataframe
        print(f"Saving combined dataframe to {output_file}...")
        combined_df.to_csv(output_file, index=False)
        print(f"Successfully saved combined dataframe to {output_file}")
        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
