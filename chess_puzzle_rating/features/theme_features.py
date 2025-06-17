"""
Module for engineering features from chess puzzle themes.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from collections import Counter
import re
from tqdm import tqdm
import logging
import concurrent.futures
import os

log = logging.getLogger(__name__)

# Helper function for parsing themes
def parse_theme(item):
    idx, theme_string = item
    local_counter = Counter()

    if pd.isna(theme_string) or not theme_string.strip():
        themes_list = []
    else:
        # Split by spaces to handle multiple themes
        themes_list = theme_string.split()

        # Count each theme
        for theme in themes_list:
            local_counter[theme] += 1

    return {
        'idx': idx, 
        'themes': themes_list, 
        'counter': local_counter,
        'svd_text': ' '.join(themes_list)
    }

# Helper function for processing entries
def process_entry(entry, top_themes):
    idx = entry['idx']
    themes_list = entry['themes']

    # Basic features
    current_features = {
        'theme_count': len(themes_list),
        'has_themes': 1 if themes_list else 0
    }

    # One-hot encoding for top themes
    theme_set = set(themes_list)
    for theme in top_themes:
        current_features[f'theme_{theme}'] = 1 if theme in theme_set else 0

    # Add strategic theme categories
    current_features['is_mate'] = 1 if any('mate' in theme.lower() for theme in themes_list) else 0
    current_features['is_fork'] = 1 if 'fork' in theme_set else 0
    current_features['is_pin'] = 1 if 'pin' in theme_set else 0
    current_features['is_skewer'] = 1 if 'skewer' in theme_set else 0
    current_features['is_discovery'] = 1 if 'discoveredAttack' in theme_set else 0
    current_features['is_sacrifice'] = 1 if 'sacrifice' in theme_set else 0
    current_features['is_promotion'] = 1 if 'promotion' in theme_set else 0
    current_features['is_endgame'] = 1 if 'endgame' in theme_set else 0
    current_features['is_middlegame'] = 1 if 'middlegame' in theme_set else 0
    current_features['is_opening'] = 1 if 'opening' in theme_set else 0

    return {'idx': idx, **current_features}

# Helper function to handle multiple arguments for process_entry
def process_entry_with_themes(entry, top_themes):
    return process_entry(entry, top_themes)

def engineer_chess_theme_features(df, theme_column='Themes',
                                 min_theme_freq=10,
                                 max_themes=100,
                                 n_svd_components=15,
                                 n_hash_features=20,
                                 additional_columns=None):
    """
    Comprehensive feature engineering for chess puzzle themes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess data
    theme_column : str, optional
        Column name containing themes, by default 'Themes'
    min_theme_freq : int, optional
        Minimum frequency for a theme to be one-hot encoded, by default 10
    max_themes : int, optional
        Max number of most frequent themes to one-hot encode, by default 100
    n_svd_components : int, optional
        Number of SVD components for theme embeddings, by default 15
    n_hash_features : int, optional
        Number of features to use for hashing vectorizer, by default 20
    additional_columns : list, optional
        List of additional columns to use for interaction features, by default None

    Returns
    -------
    pandas.DataFrame
        DataFrame with engineered theme features
    """
    log.info("Engineering Chess Theme Features...")

    # Use a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Count missing values in the Themes column
    missing_values = df_copy[theme_column].isna().sum()
    missing_percentage = (missing_values / len(df_copy)) * 100
    log.info(f"Missing values in {theme_column}: {missing_values} ({missing_percentage:.2f}%)")

    # Fill missing values with empty string
    df_copy[theme_column] = df_copy[theme_column].fillna('')

    # --- Pre-computation: Parse all themes and gather statistics in parallel ---
    log.info("Parsing themes in parallel...")
    items = list(df_copy[theme_column].items())

    # Use a more reliable approach for parallel processing with limited workers and chunking
    parsed_results = []
    # Use exactly 4 workers as specified
    max_workers = 4
    log.info(f"Using {max_workers} workers for parallel theme parsing")

    # Process in larger chunks to improve performance
    chunk_size = 100000
    total_items = len(items)

    for chunk_start in range(0, total_items, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_items)
        chunk_items = items[chunk_start:chunk_end]
        log.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_items + chunk_size - 1)//chunk_size} ({len(chunk_items)} items)")

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            # Submit tasks for this chunk
            futures = [executor.submit(parse_theme, item) for item in chunk_items]

            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Parsing Themes (Chunk {chunk_start//chunk_size + 1})"):
                parsed_results.append(future.result())

    # Combine results
    all_themes_info = []
    all_themes_counter = Counter()
    all_themes_for_svd = []

    for result in parsed_results:
        all_themes_info.append({'idx': result['idx'], 'themes': result['themes']})
        all_themes_counter.update(result['counter'])
        all_themes_for_svd.append(result['svd_text'])

    # --- Select Top N for One-Hot Encoding ---
    top_themes = [theme for theme, count in all_themes_counter.most_common(max_themes)
                 if count >= min_theme_freq]

    log.info(f"Selected {len(top_themes)} themes for one-hot encoding")

    # --- Feature Generation with Parallel Processing ---
    log.info("Generating features in parallel...")

    # Use a more reliable approach for parallel processing with limited workers and chunking
    features_list = []
    # Use exactly 4 workers as specified
    max_workers = 4
    log.info(f"Using {max_workers} workers for parallel feature generation")

    # Process in larger chunks to improve performance
    chunk_size = 100000
    total_entries = len(all_themes_info)

    for chunk_start in range(0, total_entries, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_entries)
        chunk_entries = all_themes_info[chunk_start:chunk_end]
        log.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_entries + chunk_size - 1)//chunk_size} ({len(chunk_entries)} entries)")

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            # Submit tasks for this chunk
            futures = [executor.submit(process_entry_with_themes, entry, top_themes) for entry in chunk_entries]

            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Generating Features (Chunk {chunk_start//chunk_size + 1})"):
                features_list.append(future.result())

    # Create DataFrame from features list
    themes_df = pd.DataFrame(features_list).set_index('idx')

    # --- TF-IDF and SVD Features ---
    # Only proceed if we have enough puzzles with themes
    has_themes_mask = themes_df['has_themes'] == 1
    if has_themes_mask.sum() > 100 and n_svd_components > 0:
        try:
            # Process themes for better semantic representation
            processed_themes = [theme.replace('_', ' ') for theme in all_themes_for_svd]

            # TF-IDF on themes
            theme_vectorizer = TfidfVectorizer(min_df=3, max_features=1000)
            theme_matrix = theme_vectorizer.fit_transform(processed_themes)

            n_comp = min(n_svd_components, theme_matrix.shape[1] - 1, theme_matrix.shape[0] - 1)
            if n_comp > 1:
                svd = TruncatedSVD(n_components=n_comp, random_state=42)
                svd_features = svd.fit_transform(theme_matrix)

                svd_df = pd.DataFrame(
                    svd_features,
                    columns=[f'theme_svd_{i}' for i in range(n_comp)],
                    index=themes_df.index
                )
                themes_df = pd.concat([themes_df, svd_df], axis=1)
                log.info(f"Added {n_comp} SVD features for themes")
        except Exception as e:
            log.info(f"Error in SVD processing: {str(e)}")

    # --- Hashing Features for High-Cardinality ---
    if n_hash_features > 0:
        try:
            # Process themes for hashing
            processed_themes = [theme if theme else '' for theme in all_themes_for_svd]

            hasher = HashingVectorizer(n_features=n_hash_features, alternate_sign=False)
            hash_features = hasher.fit_transform(processed_themes).toarray()

            hash_df = pd.DataFrame(
                hash_features,
                columns=[f'theme_hash_{i}' for i in range(n_hash_features)],
                index=themes_df.index
            )
            themes_df = pd.concat([themes_df, hash_df], axis=1)
            log.info(f"Added {n_hash_features} hashing features for themes")
        except Exception as e:
            log.info(f"Error in hashing: {str(e)}")

    # --- Topic Modeling with LDA ---
    # Only proceed if we have enough puzzles with themes
    if has_themes_mask.sum() > 100 and len(top_themes) > 10:
        try:
            # Create document-term matrix for themes
            count_vec = CountVectorizer(vocabulary=top_themes)
            dtm = count_vec.fit_transform(all_themes_for_svd)

            # Apply LDA
            lda = LatentDirichletAllocation(n_components=8, random_state=42)
            topic_results = lda.fit_transform(dtm)

            # Add topic features
            topic_df = pd.DataFrame(
                topic_results,
                columns=[f'theme_topic_{i}' for i in range(topic_results.shape[1])],
                index=themes_df.index
            )
            themes_df = pd.concat([themes_df, topic_df], axis=1)

            # Add dominant topic
            themes_df['dominant_theme_topic'] = topic_results.argmax(axis=1)

            log.info(f"Added {topic_results.shape[1]} LDA topic features")
        except Exception as e:
            log.info(f"Error in LDA topic modeling: {str(e)}")

    # --- Handle the imbalance between puzzles with and without themes ---
    # Create features to interact with other puzzle attributes
    if additional_columns is not None and len(additional_columns) > 0:
        for col in additional_columns:
            if col in df_copy.columns:
                # Interaction features: separate signal from puzzles with/without themes
                themes_df[f'no_theme_{col}'] = (1 - themes_df['has_themes']) * df_copy[col]
                themes_df[f'with_theme_{col}'] = themes_df['has_themes'] * df_copy[col]

    # --- Final cleaning and normalization ---
    # Fill NaNs that might have been introduced
    themes_df = themes_df.fillna(0)

    # Normalize embedding features if desired
    svd_cols = [col for col in themes_df.columns if ('svd' in col or 'hash' in col)]
    if svd_cols:
        themes_df[svd_cols] = (themes_df[svd_cols] - themes_df[svd_cols].mean()) / themes_df[svd_cols].std()
        themes_df[svd_cols] = themes_df[svd_cols].fillna(0)  # Handle any NaNs from division by zero

    log.info(f"Engineered {themes_df.shape[1]} chess theme features")
    return themes_df
