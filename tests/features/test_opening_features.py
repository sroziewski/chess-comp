import pytest
import pandas as pd
import numpy as np
from chess_puzzle_rating.features.opening_features import engineer_chess_opening_features

class TestOpeningFeatures:
    """Tests for opening features engineering."""
    
    def test_engineer_chess_opening_features_empty_df(self):
        """Test engineering opening features from an empty DataFrame."""
        empty_df = pd.DataFrame(columns=['PuzzleId', 'OpeningTags'])
        features_df = engineer_chess_opening_features(empty_df)
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 0
    
    def test_engineer_chess_opening_features(self, sample_puzzle_df):
        """Test engineering opening features from a sample DataFrame."""
        features_df = engineer_chess_opening_features(
            sample_puzzle_df,
            min_family_freq=1,
            min_variation_freq=1,
            min_keyword_freq=1
        )
        
        # Check that the function returns a DataFrame
        assert isinstance(features_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected number of rows
        assert len(features_df) == len(sample_puzzle_df)
        
        # Check that the DataFrame has opening-related columns
        opening_columns = [col for col in features_df.columns if 'opening_' in col]
        assert len(opening_columns) > 0
        
        # Check that there are no NaN values in the DataFrame
        assert not features_df.isna().any().any()
    
    def test_engineer_chess_opening_features_with_missing_tags(self):
        """Test engineering opening features with missing tags."""
        # Create a DataFrame with some missing opening tags
        df = pd.DataFrame({
            'PuzzleId': [1, 2, 3, 4, 5],
            'OpeningTags': ['Kings_Pawn', None, 'Queens_Pawn', '', 'Italian_Game']
        })
        
        features_df = engineer_chess_opening_features(
            df,
            min_family_freq=1,
            min_variation_freq=1,
            min_keyword_freq=1
        )
        
        # Check that the function returns a DataFrame
        assert isinstance(features_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected number of rows
        assert len(features_df) == len(df)
        
        # Check that the DataFrame has opening-related columns
        opening_columns = [col for col in features_df.columns if 'opening_' in col]
        assert len(opening_columns) > 0
        
        # Check that there are no NaN values in the DataFrame
        assert not features_df.isna().any().any()
    
    def test_engineer_chess_opening_features_with_additional_columns(self):
        """Test engineering opening features with additional columns."""
        # Create a DataFrame with an additional column
        df = pd.DataFrame({
            'PuzzleId': [1, 2, 3, 4, 5],
            'OpeningTags': ['Kings_Pawn', 'Queens_Pawn', 'Kings_Knight', 'Italian_Game', 'Castling'],
            'AdditionalColumn': ['A', 'B', 'C', 'D', 'E']
        })
        
        features_df = engineer_chess_opening_features(
            df,
            min_family_freq=1,
            min_variation_freq=1,
            min_keyword_freq=1,
            additional_columns=['AdditionalColumn']
        )
        
        # Check that the function returns a DataFrame
        assert isinstance(features_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected number of rows
        assert len(features_df) == len(df)
        
        # Check that the DataFrame has opening-related columns
        opening_columns = [col for col in features_df.columns if 'opening_' in col]
        assert len(opening_columns) > 0
        
        # Check that there are columns related to the additional column
        additional_columns = [col for col in features_df.columns if 'additional_' in col]
        assert len(additional_columns) > 0
        
        # Check that there are no NaN values in the DataFrame
        assert not features_df.isna().any().any()
    
    def test_engineer_chess_opening_features_with_different_parameters(self):
        """Test engineering opening features with different parameters."""
        features_df = engineer_chess_opening_features(
            sample_puzzle_df,
            min_family_freq=1,
            min_variation_freq=1,
            min_keyword_freq=1,
            max_components_level1=50,
            max_components_level2=30,
            n_svd_components_family=5,
            n_svd_components_full=8,
            n_hash_features=10,
            max_tag_depth=3
        )
        
        # Check that the function returns a DataFrame
        assert isinstance(features_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected number of rows
        assert len(features_df) == len(sample_puzzle_df)
        
        # Check that the DataFrame has opening-related columns
        opening_columns = [col for col in features_df.columns if 'opening_' in col]
        assert len(opening_columns) > 0
        
        # Check that there are no NaN values in the DataFrame
        assert not features_df.isna().any().any()