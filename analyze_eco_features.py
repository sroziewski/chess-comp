import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def analyze_eco_features(csv_path):
    """
    Analyze ECO code features to identify columns that are always 0 and propose methods to make the data denser.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing ECO code features
    
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    print(f"Loading ECO code features from {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Basic information
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Analyze which columns are always 0
    zero_cols = []
    for col in df.columns:
        if col == 'idx':  # Skip index column
            continue
        if (df[col] == 0).all():
            zero_cols.append(col)
    
    # Calculate sparsity for each column
    sparsity = {}
    for col in df.columns:
        if col == 'idx':  # Skip index column
            continue
        zeros = (df[col] == 0).sum()
        sparsity[col] = zeros / len(df)
    
    # Calculate overall sparsity
    total_cells = (df.shape[0] * (df.shape[1] - 1))  # Exclude idx column
    total_zeros = (df.iloc[:, 1:] == 0).sum().sum()
    overall_sparsity = total_zeros / total_cells
    
    # Find columns with at least some non-zero values
    non_zero_cols = [col for col in df.columns if col != 'idx' and col not in zero_cols]
    
    # Calculate correlation between non-zero columns
    corr_matrix = None
    if len(non_zero_cols) > 1:
        corr_matrix = df[non_zero_cols].corr().to_dict()
    
    # Prepare results
    results = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "zero_columns": zero_cols,
        "non_zero_columns": non_zero_cols,
        "sparsity_by_column": sparsity,
        "overall_sparsity": overall_sparsity,
        "correlation_matrix": corr_matrix
    }
    
    return results

def propose_density_methods(analysis_results):
    """
    Propose methods to make the ECO code features denser based on analysis results.
    
    Parameters:
    -----------
    analysis_results : dict
        Dictionary containing analysis results
    
    Returns:
    --------
    dict
        Dictionary containing proposed methods
    """
    # Extract relevant information from analysis results
    zero_cols = analysis_results["zero_columns"]
    non_zero_cols = analysis_results["non_zero_columns"]
    sparsity = analysis_results["sparsity_by_column"]
    overall_sparsity = analysis_results["overall_sparsity"]
    corr_matrix = analysis_results["correlation_matrix"]
    
    proposals = []
    
    # Proposal 1: Remove columns that are always zero
    if zero_cols:
        proposals.append({
            "method": "Remove zero columns",
            "description": f"Remove {len(zero_cols)} columns that are always zero",
            "columns_to_remove": zero_cols
        })
    
    # Proposal 2: Group similar ECO codes
    if len(non_zero_cols) > 1:
        # Group by ECO letter (A, B, C, D, E)
        eco_groups = {}
        for col in non_zero_cols:
            if col.startswith('eco_'):
                eco_letter = col.split('_')[1][0]  # Extract the letter (A, B, C, D, E)
                if eco_letter not in eco_groups:
                    eco_groups[eco_letter] = []
                eco_groups[eco_letter].append(col)
        
        if eco_groups:
            proposals.append({
                "method": "Group similar ECO codes",
                "description": "Combine similar ECO codes into broader categories",
                "eco_groups": eco_groups
            })
    
    # Proposal 3: Dimensionality reduction
    if len(non_zero_cols) > 5:
        proposals.append({
            "method": "Dimensionality reduction",
            "description": "Apply PCA or t-SNE to reduce dimensions while preserving information",
            "suggested_dimensions": min(5, len(non_zero_cols) // 2)
        })
    
    # Proposal 4: Create composite features
    if len(non_zero_cols) > 1:
        proposals.append({
            "method": "Create composite features",
            "description": "Create new features that combine information from multiple ECO code features",
            "examples": [
                "Sum of all ECO A features",
                "Sum of all ECO B features",
                "Binary indicator for any ECO code present"
            ]
        })
    
    return proposals

def main():
    # Define the path to the CSV file
    # Note: In a real scenario, you would use the actual path from the issue description
    # For Windows, we'll use a relative path
    data_dir = Path("data/eco_code_features")
    csv_path = data_dir / "eco_code_features_latest.csv"
    
    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if the file exists
    if not csv_path.exists():
        print(f"Warning: File {csv_path} does not exist.")
        print("This script assumes the file is available at the specified path.")
        print("Please update the path or provide the file before running this script.")
        return
    
    # Analyze the ECO code features
    analysis_results = analyze_eco_features(csv_path)
    
    # Propose methods to make the data denser
    proposals = propose_density_methods(analysis_results)
    
    # Print the results
    print("\nAnalysis Results:")
    print(f"Total columns: {len(analysis_results['columns'])}")
    print(f"Columns that are always zero: {len(analysis_results['zero_columns'])}")
    print(f"Overall sparsity: {analysis_results['overall_sparsity']:.2%}")
    
    print("\nProposed Methods to Make Data Denser:")
    for i, proposal in enumerate(proposals, 1):
        print(f"{i}. {proposal['method']}: {proposal['description']}")
    
    # Save the results to a JSON file
    output_dir = Path("results")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_dir / "eco_features_analysis.json", "w") as f:
        json.dump({
            "analysis": analysis_results,
            "proposals": proposals
        }, f, indent=2)
    
    print("\nResults saved to results/eco_features_analysis.json")
    
    # Implement the first proposal: Remove columns that are always zero
    if analysis_results["zero_columns"]:
        df = pd.read_csv(csv_path)
        df_dense = df.drop(columns=analysis_results["zero_columns"])
        
        # Save the denser dataset
        dense_path = data_dir / "eco_code_features_dense.csv"
        df_dense.to_csv(dense_path, index=False)
        
        print(f"\nDenser dataset created by removing zero columns.")
        print(f"Original shape: {df.shape}")
        print(f"New shape: {df_dense.shape}")
        print(f"Saved to {dense_path}")

if __name__ == "__main__":
    main()