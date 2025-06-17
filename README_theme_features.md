# Theme Feature Extraction

This module provides a standalone script for extracting theme features from chess puzzles and saving them separately.

## Overview

The `extract_theme_features.py` script extracts various features from chess puzzle themes, including:

- Basic features (theme count, presence of specific strategic themes)
- One-hot encoding for top themes
- TF-IDF and SVD features for semantic representation
- Hashing features for high-cardinality themes
- Topic modeling with LDA

## Usage

### Command Line Interface

The script can be run from the command line with various options:

```bash
python extract_theme_features.py --input input_data.csv --output theme_features.csv
```

#### Required Arguments

- `--input`, `-i`: Path to the input CSV file containing chess puzzle data
- `--output`, `-o`: Path to save the extracted theme features

#### Optional Arguments

- `--theme-column`: Column name containing themes (default: 'Themes')
- `--min-theme-freq`: Minimum frequency for a theme to be one-hot encoded (default: 5)
- `--max-themes`: Maximum number of most frequent themes to one-hot encode (default: 100)
- `--n-svd-components`: Number of SVD components for theme embeddings (default: 10)
- `--n-hash-features`: Number of features to use for hashing vectorizer (default: 15)
- `--log-file`: Path to log file (default: auto-generated based on timestamp)

### Programmatic Usage

You can also use the functions from the script in your own code:

```python
from extract_theme_features import extract_theme_features
import pandas as pd

# Load your data
df = pd.read_csv('input_data.csv')

# Extract theme features
theme_features = extract_theme_features(
    df,
    theme_column='Themes',
    min_theme_freq=5,
    max_themes=100,
    n_svd_components=10,
    n_hash_features=15,
    output_file='theme_features.csv'
)

# Use the extracted features
print(f"Extracted {theme_features.shape[1]} theme features")
```

## Testing

A test script is provided to demonstrate how to use the theme feature extraction functionality:

```bash
python test_theme_features.py
```

This script:
1. Creates a sample dataset with random chess puzzle themes
2. Extracts theme features from the sample data
3. Verifies that the features are saved correctly
4. Cleans up the sample files after the test

## Output

The script outputs a CSV file containing the extracted theme features. Each row corresponds to a chess puzzle, and the columns represent different features derived from the themes.

Example features include:
- `theme_count`: Number of themes in the puzzle
- `has_themes`: Whether the puzzle has any themes
- `theme_<name>`: One-hot encoding for specific themes
- `is_mate`, `is_fork`, etc.: Strategic theme categories
- `theme_svd_<n>`: SVD components for semantic representation
- `theme_hash_<n>`: Hashing features for high-cardinality themes
- `theme_topic_<n>`: Topic modeling features from LDA