# ECO Code Features Analysis and Densification

This directory contains scripts for analyzing and densifying ECO (Encyclopedia of Chess Openings) code features. The ECO code features are sparse, with many columns containing mostly zeros. These scripts help identify which columns are always zero and provide methods to make the data denser.

## Scripts

### 1. `analyze_eco_features.py`

This script analyzes the ECO code features to identify columns that are always zero and calculates sparsity metrics.

**Features:**
- Identifies columns that are always zero
- Calculates sparsity for each column and overall
- Proposes methods to make the data denser
- Implements the first proposal by removing columns that are always zero

**Usage:**
```bash
python analyze_eco_features.py
```

**Output:**
- `results/eco_features_analysis.json`: JSON file containing analysis results and proposals
- `data/eco_code_features/eco_code_features_dense.csv`: CSV file with zero columns removed

### 2. `make_eco_features_denser.py`

This script implements several methods to make the ECO code features denser.

**Features:**
- Groups similar ECO codes by their letter (A, B, C, D, E)
- Applies dimensionality reduction using PCA
- Creates composite features that combine information from multiple ECO code features
- Combines all methods to create a comprehensive set of denser features
- Visualizes the ECO code features to help understand their distribution and relationships

**Usage:**
```bash
python make_eco_features_denser.py
```

**Output:**
- `data/eco_code_features/eco_code_features_grouped.csv`: CSV file with grouped ECO code features
- `data/eco_code_features/eco_code_features_reduced.csv`: CSV file with reduced dimensions
- `data/eco_code_features/eco_code_features_composite.csv`: CSV file with composite features
- `data/eco_code_features/eco_code_features_combined.csv`: CSV file with all methods combined
- `results/eco_features/`: Directory containing visualizations

## Methods for Densifying ECO Code Features

### 1. Remove Zero Columns

The simplest approach is to remove columns that are always zero. This reduces the dimensionality of the data without losing any information.

### 2. Group Similar ECO Codes

ECO codes are grouped by their letter (A, B, C, D, E) to create denser features. For each group, we create:
- A combined feature that is the sum of all columns in the group
- A binary indicator for any ECO code in the group

### 3. Dimensionality Reduction

Principal Component Analysis (PCA) is applied to reduce the dimensionality of the ECO code features while preserving as much information as possible. The number of components can be adjusted based on the desired level of information retention.

### 4. Composite Features

New features are created that combine information from multiple ECO code features:
- Total number of ECO codes present
- Binary indicator for any ECO code present
- Sum of all ECO codes in each letter group
- Ratio of each ECO group to the total

### 5. Combined Approach

All of the above methods are combined to create a comprehensive set of denser features. This approach provides the most information-rich representation of the ECO code features.

## Visualizations

The `make_eco_features_denser.py` script generates several visualizations to help understand the ECO code features:

1. **Correlation Heatmap**: Shows the correlation between ECO code features
2. **Frequency Bar Chart**: Shows the frequency of each ECO code
3. **Distribution Pie Chart**: Shows the distribution of ECO codes by letter

## Data Structure

The original ECO code features CSV file has the following structure:
- `idx`: Index column
- `eco_A`, `eco_A_English`, etc.: ECO code features for various openings

The densified versions of the data add new features while preserving the original structure.

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Notes

- The scripts assume that the ECO code features CSV file is located at `data/eco_code_features/eco_code_features_latest.csv`. If your file is in a different location, you'll need to update the path in the scripts.
- The scripts create directories for output files if they don't exist.
- The visualizations are saved to the `results/eco_features/` directory.