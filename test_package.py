"""
Test script to verify that the chess_puzzle_rating package can be imported and used.
"""

import pandas as pd
import numpy as np
import chess
from tqdm import tqdm

# Import the package
import chess_puzzle_rating as cpr

# Create a small test DataFrame
test_df = pd.DataFrame({
    'PuzzleId': [1, 2, 3],
    'FEN': [
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',  # Starting position
        'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',  # After 1. e4
        'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2'  # After 1. e4 e5
    ],
    'Moves': [
        'e2e4 e7e5',
        'd2d4 d7d5',
        'g1f3 b8c6'
    ],
    'OpeningTags': [
        'Kings_Pawn',
        'Queens_Pawn',
        'Kings_Knight'
    ],
    'Rating': [1500, 1600, 1700]
})

print("Testing position features extraction...")
position_features = cpr.features.extract_fen_features(test_df)
print(f"Extracted {position_features.shape[1]} position features")

print("\nTesting move features extraction...")
move_features = cpr.features.extract_opening_move_features(test_df)
print(f"Extracted {move_features.shape[1]} move features")

print("\nTesting ECO code inference...")
eco_features = cpr.features.infer_eco_codes(test_df)
print(f"Inferred ECO codes with {eco_features.shape[1]} features")

print("\nTesting opening tag prediction...")
predictions, model, combined_features = cpr.features.predict_missing_opening_tags(test_df)
print(f"Made predictions for {len(predictions)} puzzles")

print("\nTesting opening features engineering...")
opening_features = cpr.features.engineer_chess_opening_features(
    test_df,
    min_family_freq=1,
    min_variation_freq=1,
    min_keyword_freq=1
)
print(f"Engineered {opening_features.shape[1]} opening features")

print("\nTesting complete feature engineering pipeline...")
final_features, model, predictions = cpr.features.complete_feature_engineering(test_df)
print(f"Created final feature set with {final_features.shape[1]} features")

print("\nAll tests passed successfully!")