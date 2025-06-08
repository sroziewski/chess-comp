import pytest
import pandas as pd
import numpy as np
import chess

@pytest.fixture
def sample_puzzle_df():
    """
    Create a small sample DataFrame of chess puzzles for testing.
    """
    return pd.DataFrame({
        'PuzzleId': [1, 2, 3, 4, 5],
        'FEN': [
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',  # Starting position
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',  # After 1. e4
            'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2',  # After 1. e4 e5
            'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2',  # After 1. e4 e5 2. Nf3
            'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',  # After 1. e4 e5 2. Nf3 Nc6
        ],
        'Moves': [
            'e2e4 e7e5',
            'd2d4 d7d5',
            'g1f3 b8c6',
            'f1c4 f8c5',
            'e1g1 e8g8',
        ],
        'OpeningTags': [
            'Kings_Pawn',
            'Queens_Pawn',
            'Kings_Knight',
            'Italian_Game',
            'Castling',
        ],
        'Rating': [1500, 1600, 1700, 1800, 1900]
    })

@pytest.fixture
def sample_puzzle_df_with_timestamps():
    """
    Create a sample DataFrame of chess puzzles with timestamps for time-based validation.
    """
    return pd.DataFrame({
        'PuzzleId': [1, 2, 3, 4, 5],
        'FEN': [
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',  # Starting position
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',  # After 1. e4
            'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2',  # After 1. e4 e5
            'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2',  # After 1. e4 e5 2. Nf3
            'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',  # After 1. e4 e5 2. Nf3 Nc6
        ],
        'Moves': [
            'e2e4 e7e5',
            'd2d4 d7d5',
            'g1f3 b8c6',
            'f1c4 f8c5',
            'e1g1 e8g8',
        ],
        'OpeningTags': [
            'Kings_Pawn',
            'Queens_Pawn',
            'Kings_Knight',
            'Italian_Game',
            'Castling',
        ],
        'Rating': [1500, 1600, 1700, 1800, 1900],
        'Timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D')
    })

@pytest.fixture
def sample_rating_ranges():
    """
    Create sample rating ranges for stratified sampling.
    """
    return [
        (0, 1200, 'Beginner'),
        (1201, 1600, 'Intermediate'),
        (1601, 2000, 'Advanced'),
        (2001, 2400, 'Expert'),
        (2401, 3000, 'Master')
    ]

@pytest.fixture
def sample_feature_df():
    """
    Create a sample DataFrame of extracted features for testing.
    """
    # Create a DataFrame with some common chess position features
    return pd.DataFrame({
        'material_balance': [0, 0, 0, 0, 0],
        'white_center_control': [4, 5, 4, 6, 5],
        'black_center_control': [4, 4, 5, 4, 5],
        'white_king_safety': [10, 10, 10, 10, 8],
        'black_king_safety': [10, 10, 10, 10, 8],
        'white_pawn_structure': [8, 7, 8, 8, 8],
        'black_pawn_structure': [8, 8, 7, 8, 8],
        'white_mobility': [20, 21, 20, 22, 21],
        'black_mobility': [20, 20, 21, 20, 21],
        'white_development': [0, 1, 0, 2, 3],
        'black_development': [0, 0, 1, 0, 1],
    })