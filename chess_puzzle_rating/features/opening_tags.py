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

import concurrent.futures
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, \
    HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from chess_puzzle_rating.features.move_features import extract_opening_move_features, infer_eco_codes
from chess_puzzle_rating.features.position_features import extract_fen_features
from chess_puzzle_rating.utils.config import get_config
from chess_puzzle_rating.utils.progress import get_logger


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
            variation_counts = df_with_tags.loc[
                has_eco & (df_with_tags['family'] == most_common_family), 'variation'].value_counts()
            if len(variation_counts) > 0 and variation_counts.iloc[0] >= 3:
                most_common_variation = variation_counts.index[0]
                eco_to_variation[eco_col] = most_common_variation

    return {'family': eco_to_family, 'variation': eco_to_variation}


def process_prediction_chunk(chunk_data):
    """
    Process a chunk of puzzles without tags to predict opening tags.
    This function is designed to be run in parallel.

    Parameters
    ----------
    chunk_data : tuple
        Tuple containing (chunk_df, family_model, variation_models, X_predict, eco_features, eco_family_map, eco_variation_map)

    Returns
    -------
    pandas.DataFrame
        DataFrame with predicted tags and confidence scores for the chunk
    """
    chunk_df, family_model, variation_models, X_predict, eco_features, eco_family_map, eco_variation_map = chunk_data

    # First predict families
    predicted_families = family_model.predict(X_predict)
    family_probs = family_model.predict_proba(X_predict)
    family_confidence = np.max(family_probs, axis=1)

    # Then predict variations for each family
    predicted_variations = [""] * len(chunk_df)
    variation_confidence = np.zeros(len(chunk_df))

    # For each puzzle in the chunk
    for i, idx in enumerate(chunk_df.index):
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
    }, index=chunk_df.index)

    return results_df


def predict_hierarchical_opening_tags(df, tag_column='OpeningTags', fen_features=None, move_features=None,
                                      eco_features=None):
    """
    Predict opening tags using a hierarchical approach (family → variation)
    with strengthened ECO code integration.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles
    tag_column : str, optional
        Name of the column containing opening tags, by default 'OpeningTags'
    fen_features : pandas.DataFrame, optional
        Pre-computed position features from FEN strings, by default None
    move_features : pandas.DataFrame, optional
        Pre-computed move features, by default None
    eco_features : pandas.DataFrame, optional
        Pre-computed ECO code features, by default None

    Returns
    -------
    tuple
        (results_df, models_dict, combined_features_df)
        - results_df: DataFrame with predicted tags and confidence scores
        - models_dict: Dictionary of trained models (family model and variation models)
        - combined_features_df: DataFrame with all features used for prediction
    """
    logger = get_logger()
    logger.info("Predicting opening tags using hierarchical classification with ECO code integration...")

    # Extract features if not provided
    if fen_features is None:
        fen_features = extract_fen_features(df)
    if move_features is None:
        move_features = extract_opening_move_features(df)
    if eco_features is None:
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
    hist_gb_model = HistGradientBoostingClassifier(
        max_iter=100,  # Similar to n_estimators
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    lgb_model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,  # Increased from 3 to allow more splits
        min_child_samples=10,  # Reduced from 20 to allow more splits with smaller data
        min_child_weight=1e-5,  # Added to handle sparse data
        min_split_gain=1e-8,  # Added to prevent splits with minimal gain
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1,  # L2 regularization
        subsample=0.8,  # Added to reduce overfitting
        colsample_bytree=0.8,  # Added to reduce overfitting
        n_jobs=-1,
        verbose=-1  # Suppress warnings
    )

    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, class_weight='balanced', random_state=42))
    ])

    family_model = VotingClassifier(
        estimators=[
            ('rf', hist_gb_model),
            ('gb', lgb_model),
            ('svm', svm_pipeline)
        ],
        voting='soft',
        weights=[1, 1, 1],
        n_jobs=-1
    )

    # Evaluate family prediction with cross-validation
    cv_scores_family = cross_val_score(family_model, X_train, y_train_family, cv=5)
    logger.info(f"Family prediction accuracy: {cv_scores_family.mean():.4f} ± {cv_scores_family.std():.4f}")

    # Train family model on all data with tags
    family_model.fit(X_train, y_train_family)

    # Function to train a variation model for a specific family
    def train_variation_model(family_data):
        family, df_with_tags, combined_features = family_data

        # Get data for this family
        family_subset = df_with_tags[df_with_tags['primary_family'] == family]

        # Count variations within this family
        variation_counts = family_subset['variation'].value_counts()
        valid_variations = variation_counts[variation_counts >= 3].index.tolist()

        # Skip if not enough variation data
        if len(valid_variations) < 2:
            return family, None

        # Prepare data for variation prediction
        X_train_var = combined_features.loc[family_subset.index]
        y_train_var = family_subset['variation']

        # Only keep rows with valid variations
        valid_var_mask = family_subset['variation'].isin(valid_variations)
        X_train_var = X_train_var[valid_var_mask]
        y_train_var = y_train_var[valid_var_mask]

        # Skip if not enough samples after filtering
        if len(X_train_var) < 10:
            return family, None

        # Create a simpler model for variation prediction (less data available)
        var_model = RandomForestClassifier(n_estimators=50, min_samples_leaf=2,
                                           class_weight='balanced', random_state=42)

        # Train variation model
        try:
            var_model.fit(X_train_var, y_train_var)
            print(f"Trained variation model for {family} with {len(valid_variations)} variations")
            return family, var_model
        except Exception as e:
            print(f"Error training variation model for {family}: {e}")
            return family, None

    # Get configuration for parallelization
    config = get_config()
    performance_config = config.get('performance', {})
    parallel_config = performance_config.get('parallel', {})

    # Determine the number of worker processes to use
    n_workers = parallel_config.get('n_workers')
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    logger.info(f"Using {n_workers} worker processes for parallel variation model training")

    # Create variation models for each family with sufficient data in parallel
    variation_models = {}

    # Prepare data for parallel processing
    family_data_list = [(family, df_with_tags, combined_features) for family in valid_families]

    # Process families in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks and collect futures
        futures = [executor.submit(train_variation_model, family_data) for family_data in family_data_list]

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc="Training variation models"):
            try:
                family, model = future.result()
                if model is not None:
                    variation_models[family] = model
            except Exception as e:
                logger.error(f"Error in variation model training: {e}")

    # Store all models in a dictionary
    models_dict = {
        'family_model': family_model,
        'variation_models': variation_models
    }

    # Predict for puzzles without tags
    df_without_tags = df[~has_tags].copy()
    X_predict = combined_features.loc[df_without_tags.index]

    # Get configuration for parallelization
    config = get_config()
    performance_config = config.get('performance', {})
    parallel_config = performance_config.get('parallel', {})

    # Determine the number of worker processes to use
    n_workers = parallel_config.get('n_workers')
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    logger.info(f"Using {n_workers} worker processes for parallel prediction")

    # Use ECO codes to enhance predictions
    eco_family_map = eco_mapping['family']
    eco_variation_map = eco_mapping['variation']

    # If there are no puzzles without tags, return an empty DataFrame
    if len(df_without_tags) == 0:
        return pd.DataFrame(), models_dict, combined_features

    # Split the dataframe into chunks for parallel processing
    chunk_size = max(1, len(df_without_tags) // n_workers)
    chunks = []

    for i in range(0, len(df_without_tags), chunk_size):
        end = min(i + chunk_size, len(df_without_tags))
        chunk_df = df_without_tags.iloc[i:end]
        chunk_X = X_predict.loc[chunk_df.index]
        chunks.append(
            (chunk_df, family_model, variation_models, chunk_X, eco_features, eco_family_map, eco_variation_map))

    # Process chunks in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks and collect futures
        futures = [executor.submit(process_prediction_chunk, chunk) for chunk in chunks]

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc="Processing prediction chunks"):
            try:
                chunk_result = future.result()
                results.append(chunk_result)
            except Exception as e:
                logger.error(f"Error processing prediction chunk: {e}")

    # Combine results from all chunks
    if results:
        results_df = pd.concat(results)
        logger.info(f"Predicted tags for {len(results_df)} puzzles without tags")
        return results_df, models_dict, combined_features
    else:
        # Return empty DataFrame if no results
        return pd.DataFrame(), models_dict, combined_features


def predict_missing_opening_tags(df, tag_column='OpeningTags', fen_features=None, move_features=None,
                                 eco_features=None):
    """
    Predict missing opening tags using an ensemble approach with hierarchical classification.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles
    tag_column : str, optional
        Name of the column containing opening tags, by default 'OpeningTags'
    fen_features : pandas.DataFrame, optional
        Pre-computed position features from FEN strings, by default None
    move_features : pandas.DataFrame, optional
        Pre-computed move features, by default None
    eco_features : pandas.DataFrame, optional
        Pre-computed ECO code features, by default None

    Returns
    -------
    tuple
        (results_df, model, combined_features_df)
        - results_df: DataFrame with predicted tags and confidence scores
        - model: Dictionary of trained models
        - combined_features_df: DataFrame with all features used for prediction
    """
    logger = get_logger()
    logger.info("Predicting missing opening tags using ensemble approach with hierarchical classification...")

    # Extract features if not provided
    if fen_features is None:
        fen_features = extract_fen_features(df)
    if move_features is None:
        move_features = extract_opening_move_features(df)
    if eco_features is None:
        eco_features = infer_eco_codes(df)

    # Combine all features
    combined_features = pd.concat([fen_features, move_features, eco_features], axis=1)
    combined_features = combined_features.fillna(0)  # Fill any NaN values

    # Use hierarchical prediction
    hierarchical_results, models_dict, _ = predict_hierarchical_opening_tags(
        df,
        tag_column=tag_column,
        fen_features=fen_features,
        move_features=move_features,
        eco_features=eco_features
    )
    logger.info(f"Identify puzzles without tags")
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

    logger.info(
        f"Made {len(high_conf_predictions)} high-confidence predictions out of {len(df_without_tags)} puzzles without tags")

    # For detailed analysis, add the hierarchical results
    results['family_only'] = hierarchical_results['predicted_family']
    results['variation_only'] = hierarchical_results['predicted_variation']
    results['family_confidence'] = hierarchical_results['family_confidence']
    results['variation_confidence'] = hierarchical_results['variation_confidence']

    return results, models_dict, combined_features
