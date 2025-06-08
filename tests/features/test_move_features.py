import pytest
import chess
import pandas as pd
import numpy as np
from chess_puzzle_rating.features.move_features import (
    extract_opening_move_features,
    infer_eco_codes,
    analyze_move_sequence,
    process_move_sequence_chunk
)

class TestOpeningMoveFeatures:
    """Tests for opening move features extraction."""
    
    def test_extract_opening_move_features_empty_df(self):
        """Test extracting opening move features from an empty DataFrame."""
        empty_df = pd.DataFrame(columns=['PuzzleId', 'Moves'])
        features_df = extract_opening_move_features(empty_df)
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 0
    
    def test_extract_opening_move_features(self, sample_puzzle_df):
        """Test extracting opening move features from a sample DataFrame."""
        features_df = extract_opening_move_features(sample_puzzle_df)
        
        # Check that the function returns a DataFrame
        assert isinstance(features_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected number of rows
        assert len(features_df) == len(sample_puzzle_df)
        
        # Check that the DataFrame has the expected columns
        expected_columns = [
            'first_move_pawn', 'first_move_e4_d4', 'first_move_center_control',
            'first_move_knight_development'
        ]
        for col in expected_columns:
            assert col in features_df.columns
        
        # Check specific features for known openings
        # For the first row (e2e4 e7e5), first_move_e4_d4 should be 1 (e4)
        assert features_df.iloc[0]['first_move_e4_d4'] == 1
        
        # For the second row (d2d4 d7d5), first_move_e4_d4 should be 1 (d4)
        assert features_df.iloc[1]['first_move_e4_d4'] == 1
        
        # Check that there are no NaN values in the DataFrame
        assert not features_df.isna().any().any()


class TestECOCodeInference:
    """Tests for ECO code inference."""
    
    def test_infer_eco_codes_empty_df(self):
        """Test inferring ECO codes from an empty DataFrame."""
        empty_df = pd.DataFrame(columns=['PuzzleId', 'Moves'])
        eco_df = infer_eco_codes(empty_df)
        assert isinstance(eco_df, pd.DataFrame)
        assert len(eco_df) == 0
    
    def test_infer_eco_codes(self, sample_puzzle_df):
        """Test inferring ECO codes from a sample DataFrame."""
        eco_df = infer_eco_codes(sample_puzzle_df)
        
        # Check that the function returns a DataFrame
        assert isinstance(eco_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected number of rows
        assert len(eco_df) == len(sample_puzzle_df)
        
        # Check that the DataFrame has ECO code related columns
        eco_columns = [col for col in eco_df.columns if 'eco_' in col]
        assert len(eco_columns) > 0
        
        # Check that there are no NaN values in the DataFrame
        assert not eco_df.isna().any().any()


class TestMoveSequenceAnalysis:
    """Tests for move sequence analysis."""
    
    def test_analyze_move_sequence_empty_df(self):
        """Test analyzing move sequences from an empty DataFrame."""
        empty_df = pd.DataFrame(columns=['PuzzleId', 'FEN', 'Moves'])
        features_df = analyze_move_sequence(empty_df)
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 0
    
    def test_analyze_move_sequence(self, sample_puzzle_df):
        """Test analyzing move sequences from a sample DataFrame."""
        features_df = analyze_move_sequence(sample_puzzle_df)
        
        # Check that the function returns a DataFrame
        assert isinstance(features_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected number of rows
        assert len(features_df) == len(sample_puzzle_df)
        
        # Check that the DataFrame has move sequence related columns
        expected_columns = [
            'move_forcing_factor', 'material_sacrifice_ratio', 'move_depth_complexity'
        ]
        for col in expected_columns:
            assert col in features_df.columns
        
        # Check that there are no NaN values in the DataFrame
        assert not features_df.isna().any().any()
    
    def test_process_move_sequence_chunk(self):
        """Test processing a chunk of move sequences."""
        # Create a small chunk of data for testing
        chunk_data = [
            (0, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'e2e4 e7e5'),
            (1, 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1', 'd2d4 d7d5')
        ]
        
        # Process the chunk
        result = process_move_sequence_chunk(chunk_data)
        
        # Check that the result is a list of dictionaries
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        
        # Check that each dictionary has the expected keys
        expected_keys = ['idx', 'move_forcing_factor', 'material_sacrifice_ratio', 'move_depth_complexity']
        for key in expected_keys:
            assert key in result[0]
        
        # Check that the indices match the input
        assert result[0]['idx'] == 0
        assert result[1]['idx'] == 1