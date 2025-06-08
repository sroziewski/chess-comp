"""
Module for feature engineering pipelines.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from chess_puzzle_rating.features.position_features import extract_fen_features
from chess_puzzle_rating.features.move_features import extract_opening_move_features, infer_eco_codes, analyze_move_sequence
from chess_puzzle_rating.features.opening_tags import predict_missing_opening_tags
from chess_puzzle_rating.features.opening_features import engineer_chess_opening_features
from chess_puzzle_rating.features.endgame_features import extract_endgame_features


def complete_feature_engineering(df, tag_column='OpeningTags'):
    """
    Complete pipeline for feature engineering with opening tag prediction.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles
    tag_column : str, optional
        Name of the column containing opening tags, by default 'OpeningTags'

    Returns
    -------
    tuple
        (final_features_df, model, predictions_df)
        - final_features_df: DataFrame with all engineered features
        - model: Trained model for predicting opening tags
        - predictions_df: DataFrame with predicted opening tags and confidence scores
    """
    # Step 1: Extract position, move, and endgame features
    position_features = extract_fen_features(df)
    move_features = extract_opening_move_features(df)
    eco_features = infer_eco_codes(df)
    move_analysis_features = analyze_move_sequence(df)
    endgame_features = extract_endgame_features(df)

    # Step 2: Predict missing opening tags
    predictions, model, combined_features = predict_missing_opening_tags(df, tag_column)

    # Step 3: Create a new column with original + predicted tags
    enhanced_tags = df[tag_column].copy()

    # Add high-confidence predictions
    high_conf_mask = predictions['prediction_confidence'] >= 0.7
    for idx in predictions[high_conf_mask].index:
        family = predictions.loc[idx, 'predicted_family']
        # Only add prediction if original is empty
        if pd.isna(enhanced_tags.loc[idx]) or enhanced_tags.loc[idx] == '':
            enhanced_tags.loc[idx] = f"{family} (predicted)"

    # Step 4: Engineer opening features using the enhanced tags
    # We can use the original function but with the enhanced tags
    df_with_enhanced_tags = df.copy()
    df_with_enhanced_tags['EnhancedOpeningTags'] = enhanced_tags

    # Use the opening feature engineering function
    opening_features = engineer_chess_opening_features(
        df_with_enhanced_tags,
        tag_column='EnhancedOpeningTags',
        min_family_freq=20,
        min_variation_freq=10,
        min_keyword_freq=50
    )

    # Step 5: Add prediction metadata
    opening_features['has_original_tag'] = (~df[tag_column].isna() & (df[tag_column] != '')).astype(int)
    opening_features['has_predicted_tag'] = (~df[tag_column].isna() & (df[tag_column] != '') |
                                            (predictions['prediction_confidence'] >= 0.7)).astype(int)
    opening_features['tag_prediction_confidence'] = 0.0

    # Add confidence for predicted tags
    for idx in predictions.index:
        opening_features.loc[idx, 'tag_prediction_confidence'] = predictions.loc[idx, 'prediction_confidence']

    # Set confidence to 1.0 for original tags
    original_tag_mask = (~df[tag_column].isna() & (df[tag_column] != ''))
    opening_features.loc[original_tag_mask, 'tag_prediction_confidence'] = 1.0

    # Step 6: Combine all feature sets
    final_features = pd.concat([
        opening_features,
        position_features,
        move_features,
        eco_features,
        move_analysis_features,
        endgame_features
    ], axis=1)

    # Fill any remaining NaN values
    final_features = final_features.fillna(0)

    print(f"Created final feature set with {final_features.shape[1]} features")
    return final_features, model, predictions
