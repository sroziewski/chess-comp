import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_eco_features(csv_path):
    """
    Load ECO code features from CSV file.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the CSV file containing ECO code features
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing ECO code features
    """
    print(f"Loading ECO code features from {csv_path}")
    return pd.read_csv(csv_path)

def group_similar_eco_codes(df):
    """
    Group similar ECO codes to create denser features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing ECO code features
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with grouped ECO code features
    """
    print("Grouping similar ECO codes...")
    
    # Create a copy of the DataFrame
    df_grouped = df.copy()
    
    # Group by ECO letter (A, B, C, D, E)
    eco_groups = {}
    for col in df.columns:
        if col == 'idx':
            continue
        
        if col.startswith('eco_'):
            parts = col.split('_')
            if len(parts) > 1:
                eco_letter = parts[1][0]  # Extract the letter (A, B, C, D, E)
                if eco_letter not in eco_groups:
                    eco_groups[eco_letter] = []
                eco_groups[eco_letter].append(col)
    
    # Create aggregated features for each ECO group
    for letter, cols in eco_groups.items():
        if len(cols) > 1:
            # Create a new column that is the sum of all columns in this group
            df_grouped[f'eco_{letter}_combined'] = df[cols].sum(axis=1)
            
            # Create a binary indicator for any ECO code in this group
            df_grouped[f'eco_{letter}_any'] = (df[cols].sum(axis=1) > 0).astype(int)
            
            # Drop the original columns if desired
            # df_grouped = df_grouped.drop(columns=cols)
    
    return df_grouped

def apply_dimensionality_reduction(df, n_components=5):
    """
    Apply dimensionality reduction to ECO code features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing ECO code features
    n_components : int, optional
        Number of components to keep, by default 5
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with reduced dimensions
    """
    print(f"Applying dimensionality reduction with {n_components} components...")
    
    # Create a copy of the DataFrame
    df_reduced = df.copy()
    
    # Identify ECO code columns (excluding 'idx' and any newly created columns)
    eco_cols = [col for col in df.columns if col != 'idx' and col.startswith('eco_')]
    
    if len(eco_cols) <= n_components:
        print(f"Warning: Number of ECO columns ({len(eco_cols)}) is less than or equal to n_components ({n_components}).")
        print("Skipping dimensionality reduction.")
        return df_reduced
    
    # Extract the ECO code features
    X = df[eco_cols].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a DataFrame with the PCA components
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f'eco_pca_{i+1}' for i in range(n_components)],
        index=df.index
    )
    
    # Add the PCA components to the original DataFrame
    for col in pca_df.columns:
        df_reduced[col] = pca_df[col]
    
    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    print(f"Explained variance by component: {explained_variance}")
    print(f"Cumulative explained variance: {cumulative_variance}")
    
    return df_reduced

def create_composite_features(df):
    """
    Create composite features from ECO code features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing ECO code features
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with composite features
    """
    print("Creating composite features...")
    
    # Create a copy of the DataFrame
    df_composite = df.copy()
    
    # Identify ECO code columns (excluding 'idx')
    eco_cols = [col for col in df.columns if col != 'idx' and col.startswith('eco_')]
    
    # Create a feature for the total number of ECO codes present
    df_composite['eco_total_count'] = df[eco_cols].sum(axis=1)
    
    # Create a binary indicator for any ECO code present
    df_composite['eco_any_present'] = (df[eco_cols].sum(axis=1) > 0).astype(int)
    
    # Group ECO codes by their first letter (A, B, C, D, E)
    eco_by_letter = {}
    for col in eco_cols:
        parts = col.split('_')
        if len(parts) > 1:
            letter = parts[1][0]  # Extract the letter (A, B, C, D, E)
            if letter not in eco_by_letter:
                eco_by_letter[letter] = []
            eco_by_letter[letter].append(col)
    
    # Create features for each ECO letter group
    for letter, cols in eco_by_letter.items():
        # Sum of all ECO codes in this group
        df_composite[f'eco_{letter}_sum'] = df[cols].sum(axis=1)
        
        # Ratio of this ECO group to total ECO codes
        # Avoid division by zero
        total_eco = df[eco_cols].sum(axis=1)
        ratio = np.zeros(len(df))
        mask = total_eco > 0
        ratio[mask] = df[cols].sum(axis=1)[mask] / total_eco[mask]
        df_composite[f'eco_{letter}_ratio'] = ratio
    
    return df_composite

def visualize_eco_features(df, output_dir):
    """
    Visualize ECO code features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing ECO code features
    output_dir : str or Path
        Directory to save visualizations
    """
    print("Visualizing ECO code features...")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify ECO code columns (excluding 'idx')
    eco_cols = [col for col in df.columns if col != 'idx' and col.startswith('eco_')]
    
    # 1. Heatmap of correlation between ECO code features
    if len(eco_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr_matrix = df[eco_cols].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Between ECO Code Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eco_correlation_heatmap.png'))
        plt.close()
    
    # 2. Bar chart of ECO code frequency
    plt.figure(figsize=(12, 6))
    eco_sums = df[eco_cols].sum().sort_values(ascending=False)
    eco_sums.plot(kind='bar')
    plt.title('Frequency of ECO Codes')
    plt.ylabel('Count')
    plt.xlabel('ECO Code')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eco_frequency_barchart.png'))
    plt.close()
    
    # 3. Pie chart of ECO code distribution by letter
    eco_by_letter = {}
    for col in eco_cols:
        parts = col.split('_')
        if len(parts) > 1:
            letter = parts[1][0]  # Extract the letter (A, B, C, D, E)
            if letter not in eco_by_letter:
                eco_by_letter[letter] = 0
            eco_by_letter[letter] += df[col].sum()
    
    if eco_by_letter:
        plt.figure(figsize=(10, 8))
        plt.pie(eco_by_letter.values(), labels=eco_by_letter.keys(), autopct='%1.1f%%')
        plt.title('Distribution of ECO Codes by Letter')
        plt.savefig(os.path.join(output_dir, 'eco_distribution_piechart.png'))
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    # Define paths
    data_dir = Path("data/eco_code_features")
    csv_path = data_dir / "eco_code_features_latest.csv"
    output_dir = Path("results/eco_features")
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the file exists
    if not csv_path.exists():
        print(f"Warning: File {csv_path} does not exist.")
        print("This script assumes the file is available at the specified path.")
        print("Please update the path or provide the file before running this script.")
        return
    
    # Load the ECO code features
    df = load_eco_features(csv_path)
    
    # Apply the methods to make the data denser
    
    # 1. Group similar ECO codes
    df_grouped = group_similar_eco_codes(df)
    grouped_path = data_dir / "eco_code_features_grouped.csv"
    df_grouped.to_csv(grouped_path, index=False)
    print(f"Grouped ECO features saved to {grouped_path}")
    
    # 2. Apply dimensionality reduction
    df_reduced = apply_dimensionality_reduction(df, n_components=5)
    reduced_path = data_dir / "eco_code_features_reduced.csv"
    df_reduced.to_csv(reduced_path, index=False)
    print(f"Reduced ECO features saved to {reduced_path}")
    
    # 3. Create composite features
    df_composite = create_composite_features(df)
    composite_path = data_dir / "eco_code_features_composite.csv"
    df_composite.to_csv(composite_path, index=False)
    print(f"Composite ECO features saved to {composite_path}")
    
    # 4. Combine all methods
    # Start with the grouped features
    df_combined = df_grouped.copy()
    
    # Add the PCA components
    pca_cols = [col for col in df_reduced.columns if col.startswith('eco_pca_')]
    for col in pca_cols:
        df_combined[col] = df_reduced[col]
    
    # Add the composite features
    composite_cols = [
        'eco_total_count', 'eco_any_present',
        *[col for col in df_composite.columns if col.endswith('_sum') or col.endswith('_ratio')]
    ]
    for col in composite_cols:
        if col in df_composite.columns:
            df_combined[col] = df_composite[col]
    
    # Save the combined dataset
    combined_path = data_dir / "eco_code_features_combined.csv"
    df_combined.to_csv(combined_path, index=False)
    print(f"Combined ECO features saved to {combined_path}")
    
    # Visualize the ECO features
    visualize_eco_features(df, output_dir)
    
    print("\nSummary of created datasets:")
    print(f"1. Grouped ECO features: {df_grouped.shape} - {grouped_path}")
    print(f"2. Reduced ECO features: {df_reduced.shape} - {reduced_path}")
    print(f"3. Composite ECO features: {df_composite.shape} - {composite_path}")
    print(f"4. Combined ECO features: {df_combined.shape} - {combined_path}")

if __name__ == "__main__":
    main()