import pytest
import pandas as pd
import numpy as np
from chess_puzzle_rating.features.pipeline import complete_feature_engineering
from chess_puzzle_rating.features.position_features import extract_fen_features
from chess_puzzle_rating.features.move_features import extract_opening_move_features, analyze_move_sequence
from chess_puzzle_rating.features.opening_features import engineer_chess_opening_features

class TestFeatureEngineeringPipeline:
    """Tests for the feature engineering pipeline."""
    
    def test_complete_feature_engineering(self, sample_puzzle_df):
        """Test the complete feature engineering pipeline."""
        # Run the complete feature engineering pipeline
        features_df, model, predictions = complete_feature_engineering(sample_puzzle_df)
        
        # Check that the output has the expected shape
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == len(sample_puzzle_df)
        
        # Check that there are no missing values in the features
        assert not features_df.isna().any().any(), "Missing values in features_df"
        
        # Check that the features have reasonable values
        assert features_df.min().min() >= -1000, f"Feature value below minimum: {features_df.min().min()}"
        assert features_df.max().max() <= 1000, f"Feature value above maximum: {features_df.max().max()}"
    
    def test_feature_extraction_components(self, sample_puzzle_df):
        """Test individual feature extraction components."""
        # Extract position features
        position_features = extract_fen_features(sample_puzzle_df)
        
        # Check that the position features have the expected shape
        assert isinstance(position_features, pd.DataFrame)
        assert len(position_features) == len(sample_puzzle_df)
        
        # Extract opening move features
        move_features = extract_opening_move_features(sample_puzzle_df)
        
        # Check that the move features have the expected shape
        assert isinstance(move_features, pd.DataFrame)
        assert len(move_features) == len(sample_puzzle_df)
        
        # Extract move sequence features
        sequence_features = analyze_move_sequence(sample_puzzle_df)
        
        # Check that the sequence features have the expected shape
        assert isinstance(sequence_features, pd.DataFrame)
        assert len(sequence_features) == len(sample_puzzle_df)
        
        # Extract opening features
        opening_features = engineer_chess_opening_features(
            sample_puzzle_df,
            min_family_freq=1,
            min_variation_freq=1,
            min_keyword_freq=1
        )
        
        # Check that the opening features have the expected shape
        assert isinstance(opening_features, pd.DataFrame)
        assert len(opening_features) == len(sample_puzzle_df)
        
        # Check that there are no missing values in any of the feature sets
        assert not position_features.isna().any().any(), "Missing values in position_features"
        assert not move_features.isna().any().any(), "Missing values in move_features"
        assert not sequence_features.isna().any().any(), "Missing values in sequence_features"
        assert not opening_features.isna().any().any(), "Missing values in opening_features"
    
    def test_feature_combination(self, sample_puzzle_df):
        """Test combining features from different sources."""
        # Extract features from different sources
        position_features = extract_fen_features(sample_puzzle_df)
        move_features = extract_opening_move_features(sample_puzzle_df)
        
        # Combine the features
        combined_features = pd.concat([position_features, move_features], axis=1)
        
        # Check that the combined features have the expected shape
        assert isinstance(combined_features, pd.DataFrame)
        assert len(combined_features) == len(sample_puzzle_df)
        assert combined_features.shape[1] == position_features.shape[1] + move_features.shape[1]
        
        # Check that there are no missing values in the combined features
        assert not combined_features.isna().any().any(), "Missing values in combined_features"