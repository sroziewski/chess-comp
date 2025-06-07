"""
Module for predicting and handling chess opening tags.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from chess_puzzle_rating.features.position_features import extract_fen_features
from chess_puzzle_rating.features.move_features import extract_opening_move_features, infer_eco_codes


def extract_primary_family(tag_str):
    """
    Extract the primary opening family from a tag string.
    
    Parameters
    ----------
    tag_str : str
        String containing opening tags
        
    Returns
    -------
    str
        Primary opening family
    """
    if pd.isna(tag_str) or not tag_str:
        return 'Unknown'
    # Get first tag if multiple tags exist
    first_tag = tag_str.split()[0]
    # Get first component of the tag (typically the family name)
    family = first_tag.split('_')[0] if '_' in first_tag else first_tag
    return family


def predict_missing_opening_tags(df, tag_column='OpeningTags'):
    """
    Predict missing opening tags using position and move features.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles
    tag_column : str, optional
        Name of the column containing opening tags, by default 'OpeningTags'
        
    Returns
    -------
    tuple
        (results_df, model, combined_features_df)
        - results_df: DataFrame with predicted families and confidence scores
        - model: Trained RandomForestClassifier
        - combined_features_df: DataFrame with all features used for prediction
    """
    print("Predicting missing opening tags...")

    # Extract features
    fen_features = extract_fen_features(df)
    move_features = extract_opening_move_features(df)
    eco_features = infer_eco_codes(df)

    # Combine all features
    combined_features = pd.concat([fen_features, move_features, eco_features], axis=1)
    combined_features = combined_features.fillna(0)  # Fill any NaN values

    # Identify puzzles with and without tags
    has_tags = ~df[tag_column].isna() & (df[tag_column] != '')

    # Create target for prediction
    df_with_tags = df[has_tags].copy()
    df_with_tags['primary_family'] = df_with_tags[tag_column].apply(extract_primary_family)

    # Only keep families that appear at least 5 times for reliable prediction
    family_counts = df_with_tags['primary_family'].value_counts()
    valid_families = family_counts[family_counts >= 5].index.tolist()

    df_with_tags = df_with_tags[df_with_tags['primary_family'].isin(valid_families)]

    # Prepare data for training
    X_train = combined_features.loc[df_with_tags.index]
    y_train = df_with_tags['primary_family']

    # Train a model
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                  class_weight='balanced', random_state=42)

    # Evaluate with cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train on all data with tags
    model.fit(X_train, y_train)

    # Predict for puzzles without tags
    df_without_tags = df[~has_tags].copy()
    X_predict = combined_features.loc[df_without_tags.index]

    predicted_families = model.predict(X_predict)
    prediction_probs = model.predict_proba(X_predict)
    confidence_scores = np.max(prediction_probs, axis=1)

    # Create results DataFrame
    results = pd.DataFrame({
        'predicted_family': predicted_families,
        'prediction_confidence': confidence_scores
    }, index=df_without_tags.index)

    # Only keep high-confidence predictions
    high_conf_threshold = 0.7
    high_conf_predictions = results[results['prediction_confidence'] >= high_conf_threshold]

    print(
        f"Made {len(high_conf_predictions)} high-confidence predictions out of {len(df_without_tags)} puzzles without tags")

    return results, model, combined_features