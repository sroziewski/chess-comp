import pytest
import pandas as pd
import numpy as np
from chess_puzzle_rating.data.pipeline import run_data_pipeline

class TestDataValidation:
    """Tests for data validation."""
    
    def test_data_types(self, sample_puzzle_df):
        """Test that the data has the expected data types."""
        # Check that PuzzleId is an integer or string
        assert pd.api.types.is_integer_dtype(sample_puzzle_df['PuzzleId']) or pd.api.types.is_string_dtype(sample_puzzle_df['PuzzleId'])
        
        # Check that FEN is a string
        assert pd.api.types.is_string_dtype(sample_puzzle_df['FEN'])
        
        # Check that Moves is a string
        assert pd.api.types.is_string_dtype(sample_puzzle_df['Moves'])
        
        # Check that OpeningTags is a string
        assert pd.api.types.is_string_dtype(sample_puzzle_df['OpeningTags'])
        
        # Check that Rating is a numeric type
        assert pd.api.types.is_numeric_dtype(sample_puzzle_df['Rating'])
    
    def test_no_missing_values(self, sample_puzzle_df):
        """Test that there are no missing values in required columns."""
        required_columns = ['PuzzleId', 'FEN', 'Moves', 'Rating']
        for column in required_columns:
            assert not sample_puzzle_df[column].isna().any(), f"Column {column} has missing values"
    
    def test_valid_fen_strings(self, sample_puzzle_df):
        """Test that FEN strings are valid."""
        import chess
        
        # Check that each FEN string can be parsed by the chess library
        for fen in sample_puzzle_df['FEN']:
            try:
                chess.Board(fen)
            except ValueError:
                pytest.fail(f"Invalid FEN string: {fen}")
    
    def test_valid_moves_strings(self, sample_puzzle_df):
        """Test that Moves strings are valid."""
        import chess
        
        # Check that each Moves string contains valid moves
        for i, row in sample_puzzle_df.iterrows():
            board = chess.Board(row['FEN'])
            moves = row['Moves'].split()
            
            for move_str in moves:
                try:
                    move = chess.Move.from_uci(move_str)
                    assert move in board.legal_moves, f"Illegal move {move_str} in position {row['FEN']}"
                    board.push(move)
                except ValueError:
                    pytest.fail(f"Invalid move string: {move_str}")
    
    def test_rating_range(self, sample_puzzle_df):
        """Test that ratings are within a reasonable range."""
        # Chess puzzle ratings are typically between 500 and 3000
        min_rating = 500
        max_rating = 3000
        
        assert sample_puzzle_df['Rating'].min() >= min_rating, f"Rating below minimum: {sample_puzzle_df['Rating'].min()}"
        assert sample_puzzle_df['Rating'].max() <= max_rating, f"Rating above maximum: {sample_puzzle_df['Rating'].max()}"
    
    def test_unique_puzzle_ids(self, sample_puzzle_df):
        """Test that puzzle IDs are unique."""
        assert sample_puzzle_df['PuzzleId'].nunique() == len(sample_puzzle_df), "Duplicate puzzle IDs found"


class TestDataPipeline:
    """Tests for the data pipeline."""
    
    @pytest.mark.skip(reason="This test requires the full dataset and may take a long time to run")
    def test_run_data_pipeline(self):
        """Test running the data pipeline."""
        # Run the pipeline with a small subset of data
        X_train, X_test, y_train, test_ids = run_data_pipeline()
        
        # Check that the output has the expected shapes
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(test_ids, pd.Series)
        
        # Check that there are no missing values in the features
        assert not X_train.isna().any().any(), "Missing values in X_train"
        assert not X_test.isna().any().any(), "Missing values in X_test"
        
        # Check that there are no missing values in the target
        assert not y_train.isna().any(), "Missing values in y_train"
        
        # Check that the target has the expected range
        min_rating = 500
        max_rating = 3000
        assert y_train.min() >= min_rating, f"Rating below minimum: {y_train.min()}"
        assert y_train.max() <= max_rating, f"Rating above maximum: {y_train.max()}"
        
        # Check that the features have reasonable values
        assert X_train.min().min() >= -1000, f"Feature value below minimum: {X_train.min().min()}"
        assert X_train.max().max() <= 1000, f"Feature value above maximum: {X_train.max().max()}"
        
        # Check that the test IDs are unique
        assert test_ids.nunique() == len(test_ids), "Duplicate test IDs found"