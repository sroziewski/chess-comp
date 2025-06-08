"""
Module for predicting and handling chess opening tags.

This module implements an advanced approach to opening tag prediction with:
1. Ensemble learning: Combines multiple models (RandomForest, GradientBoosting, SVM)
   for more robust predictions
2. Hierarchical classification: First predicts the opening family, then the variation
   within that family
3. ECO code integration: Uses ECO (Encyclopedia of Chess Openings) codes to enhance
   prediction accuracy and confidence
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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


def extract_variation(tag_str):
    """
    Extract the variation from a tag string.

    Parameters
    ----------
    tag_str : str
        String containing opening tags

    Returns
    -------
    str
        Opening variation or empty string if no variation is found
    """
    if pd.isna(tag_str) or not tag_str:
        return ''

    # Get first tag if multiple tags exist
    first_tag = tag_str.split()[0]

    # If there's an underscore, everything after the first underscore is considered the variation
    if '_' in first_tag:
        parts = first_tag.split('_')
        if len(parts) > 1:
            return '_'.join(parts[1:])

    return ''


def create_eco_mapping(df, tag_column='OpeningTags'):
    """
    Create a mapping between ECO codes and opening families/variations.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles
    tag_column : str, optional
        Name of the column containing opening tags, by default 'OpeningTags'

    Returns
    -------
    dict
        Dictionary mapping ECO codes to opening families and variations
    """
    # ECO code categories
    eco_categories = ['A', 'B', 'C', 'D', 'E']

    # Get puzzles with both ECO codes and opening tags
    has_tags = ~df[tag_column].isna() & (df[tag_column] != '')
    df_with_tags = df[has_tags].copy()

    # Extract ECO features
    eco_features = infer_eco_codes(df_with_tags)

    # Extract family and variation from tags
    df_with_tags['family'] = df_with_tags[tag_column].apply(extract_primary_family)
    df_with_tags['variation'] = df_with_tags[tag_column].apply(extract_variation)

    # Create mapping dictionaries
    eco_to_family = {}
    eco_to_variation = {}

    # For each ECO category
    for eco in eco_categories:
        # Get ECO columns for this category
        eco_cols = [col for col in eco_features.columns if col.startswith(f'eco_{eco}_')]

        if not eco_cols:
            continue

        # For each specific ECO code
        for eco_col in eco_cols:
            # Get puzzles with this ECO code
            has_eco = eco_features[eco_col] == 1
            if has_eco.sum() < 5:
                continue

            # Get the most common family for this ECO code
            family_counts = df_with_tags.loc[has_eco, 'family'].value_counts()
            if len(family_counts) == 0:
                continue

            most_common_family = family_counts.index[0]
            eco_to_family[eco_col] = most_common_family

            # Get the most common variation for this ECO code
            variation_counts = df_with_tags.loc[has_eco & (df_with_tags['family'] == most_common_family), 'variation'].value_counts()
            if len(variation_counts) > 0 and variation_counts.iloc[0] >= 3:
                most_common_variation = variation_counts.index[0]
                eco_to_variation[eco_col] = most_common_variation

    return {'family': eco_to_family, 'variation': eco_to_variation}


def predict_hierarchical_opening_tags(df, tag_column='OpeningTags'):
    """
    Predict opening tags using a hierarchical approach (family → variation)
    with strengthened ECO code integration.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles
    tag_column : str, optional
        Name of the column containing opening tags, by default 'OpeningTags'

    Returns
    -------
    tuple
        (results_df, models_dict, combined_features_df)
        - results_df: DataFrame with predicted tags and confidence scores
        - models_dict: Dictionary of trained models (family model and variation models)
        - combined_features_df: DataFrame with all features used for prediction
    """
    print("Predicting opening tags using hierarchical classification with ECO code integration...")

    # Extract features
    fen_features = extract_fen_features(df)
    move_features = extract_opening_move_features(df)
    eco_features = infer_eco_codes(df)

    # Create ECO code mapping
    eco_mapping = create_eco_mapping(df, tag_column)

    # Combine all features
    combined_features = pd.concat([fen_features, move_features, eco_features], axis=1)
    combined_features = combined_features.fillna(0)  # Fill any NaN values

    # Identify puzzles with and without tags
    has_tags = ~df[tag_column].isna() & (df[tag_column] != '')

    # Create targets for family and variation prediction
    df_with_tags = df[has_tags].copy()
    df_with_tags['primary_family'] = df_with_tags[tag_column].apply(extract_primary_family)
    df_with_tags['variation'] = df_with_tags[tag_column].apply(extract_variation)

    # Only keep families that appear at least 5 times for reliable prediction
    family_counts = df_with_tags['primary_family'].value_counts()
    valid_families = family_counts[family_counts >= 5].index.tolist()

    df_with_tags = df_with_tags[df_with_tags['primary_family'].isin(valid_families)]

    # Prepare data for family prediction
    X_train = combined_features.loc[df_with_tags.index]
    y_train_family = df_with_tags['primary_family']

    # Create ensemble model for family prediction
    rf_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                     class_weight='balanced', random_state=42)

    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                         max_depth=3, random_state=42)

    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, class_weight='balanced', random_state=42))
    ])

    family_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model),
            ('svm', svm_pipeline)
        ],
        voting='soft',
        weights=[2, 1, 1]
    )

    # Evaluate family prediction with cross-validation
    cv_scores_family = cross_val_score(family_model, X_train, y_train_family, cv=5)
    print(f"Family prediction accuracy: {cv_scores_family.mean():.4f} ± {cv_scores_family.std():.4f}")

    # Train family model on all data with tags
    family_model.fit(X_train, y_train_family)

    # Create variation models for each family with sufficient data
    variation_models = {}
    for family in valid_families:
        # Get data for this family
        family_data = df_with_tags[df_with_tags['primary_family'] == family]

        # Count variations within this family
        variation_counts = family_data['variation'].value_counts()
        valid_variations = variation_counts[variation_counts >= 3].index.tolist()

        # Skip if not enough variation data
        if len(valid_variations) < 2:
            continue

        # Prepare data for variation prediction
        X_train_var = combined_features.loc[family_data.index]
        y_train_var = family_data['variation']

        # Only keep rows with valid variations
        valid_var_mask = family_data['variation'].isin(valid_variations)
        X_train_var = X_train_var[valid_var_mask]
        y_train_var = y_train_var[valid_var_mask]

        # Skip if not enough samples after filtering
        if len(X_train_var) < 10:
            continue

        # Create a simpler model for variation prediction (less data available)
        var_model = RandomForestClassifier(n_estimators=50, min_samples_leaf=2, 
                                          class_weight='balanced', random_state=42)

        # Train variation model
        try:
            var_model.fit(X_train_var, y_train_var)
            variation_models[family] = var_model
            print(f"Trained variation model for {family} with {len(valid_variations)} variations")
        except Exception as e:
            print(f"Error training variation model for {family}: {e}")

    # Store all models in a dictionary
    models_dict = {
        'family_model': family_model,
        'variation_models': variation_models
    }

    # Predict for puzzles without tags
    df_without_tags = df[~has_tags].copy()
    X_predict = combined_features.loc[df_without_tags.index]

    # First predict families
    predicted_families = family_model.predict(X_predict)
    family_probs = family_model.predict_proba(X_predict)
    family_confidence = np.max(family_probs, axis=1)

    # Then predict variations for each family
    predicted_variations = [""] * len(df_without_tags)
    variation_confidence = np.zeros(len(df_without_tags))

    # Use ECO codes to enhance predictions
    eco_family_map = eco_mapping['family']
    eco_variation_map = eco_mapping['variation']

    # For each puzzle without tags
    for i, idx in enumerate(df_without_tags.index):
        # Get the model's family prediction
        model_family = predicted_families[i]

        # Check if any ECO codes match for this puzzle
        eco_family = None
        eco_variation = None
        eco_confidence = 0.0

        # Look for matching ECO codes
        for eco_col, family in eco_family_map.items():
            if eco_col in eco_features.columns and eco_features.loc[idx, eco_col] == 1:
                eco_family = family
                eco_confidence = 0.8  # High confidence for ECO-based prediction

                # Check if there's a variation for this ECO code
                if eco_col in eco_variation_map:
                    eco_variation = eco_variation_map[eco_col]
                break

        # Combine model and ECO predictions for family
        if eco_family:
            # If model and ECO agree, increase confidence
            if model_family == eco_family:
                family_confidence[i] = min(0.95, family_confidence[i] + 0.15)
            # If they disagree but ECO confidence is high, use ECO prediction
            elif eco_confidence > family_confidence[i]:
                predicted_families[i] = eco_family
                family_confidence[i] = eco_confidence

        # Predict variation
        if predicted_families[i] in variation_models:
            var_model = variation_models[predicted_families[i]]
            var_pred = var_model.predict([X_predict.loc[idx]])[0]
            var_probs = var_model.predict_proba([X_predict.loc[idx]])
            var_conf = np.max(var_probs, axis=1)[0]

            # If ECO predicts a variation for this family, consider it
            if eco_variation and predicted_families[i] == eco_family:
                # If model and ECO agree on variation, increase confidence
                if var_pred == eco_variation:
                    var_conf = min(0.95, var_conf + 0.15)
                # If they disagree but ECO confidence is high, use ECO prediction
                elif eco_confidence > var_conf:
                    var_pred = eco_variation
                    var_conf = eco_confidence

            predicted_variations[i] = var_pred
            variation_confidence[i] = var_conf

    # Combine family and variation predictions
    predicted_tags = []
    for family, variation in zip(predicted_families, predicted_variations):
        if variation:
            predicted_tags.append(f"{family}_{variation}")
        else:
            predicted_tags.append(family)

    # Calculate overall confidence as a weighted average of family and variation confidence
    overall_confidence = 0.7 * family_confidence
    variation_mask = np.array([bool(v) for v in predicted_variations])
    if any(variation_mask):
        overall_confidence[variation_mask] += 0.3 * variation_confidence[variation_mask]

    # Create results DataFrame
    results_df = pd.DataFrame({
        'predicted_family': predicted_families,
        'predicted_variation': predicted_variations,
        'predicted_tag': predicted_tags,
        'family_confidence': family_confidence,
        'variation_confidence': variation_confidence,
        'prediction_confidence': overall_confidence
    }, index=df_without_tags.index)

    return results_df, models_dict, combined_features


def predict_missing_opening_tags(df, tag_column='OpeningTags'):
    """
    Predict missing opening tags using an ensemble approach with hierarchical classification.

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
        - results_df: DataFrame with predicted tags and confidence scores
        - model: Dictionary of trained models
        - combined_features_df: DataFrame with all features used for prediction
    """
    print("Predicting missing opening tags using ensemble approach with hierarchical classification...")

    # Extract features
    fen_features = extract_fen_features(df)
    move_features = extract_opening_move_features(df)
    eco_features = infer_eco_codes(df)

    # Combine all features
    combined_features = pd.concat([fen_features, move_features, eco_features], axis=1)
    combined_features = combined_features.fillna(0)  # Fill any NaN values

    # Use hierarchical prediction
    hierarchical_results, models_dict, _ = predict_hierarchical_opening_tags(df, tag_column)

    # Identify puzzles without tags
    has_tags = ~df[tag_column].isna() & (df[tag_column] != '')
    df_without_tags = df[~has_tags].copy()

    # Create a simplified results DataFrame for backward compatibility
    results = pd.DataFrame({
        'predicted_family': hierarchical_results['predicted_tag'],  # Use full tag (family_variation)
        'prediction_confidence': hierarchical_results['prediction_confidence']
    }, index=hierarchical_results.index)

    # Only keep high-confidence predictions
    high_conf_threshold = 0.7
    high_conf_predictions = results[results['prediction_confidence'] >= high_conf_threshold]

    print(
        f"Made {len(high_conf_predictions)} high-confidence predictions out of {len(df_without_tags)} puzzles without tags")

    # For detailed analysis, add the hierarchical results
    results['family_only'] = hierarchical_results['predicted_family']
    results['variation_only'] = hierarchical_results['predicted_variation']
    results['family_confidence'] = hierarchical_results['family_confidence']
    results['variation_confidence'] = hierarchical_results['variation_confidence']

    return results, models_dict, combined_features
