Based on your columns, you have both `FEN` and `Moves` available, which are perfect for extracting position-related features. These can be used to infer missing opening tags. Here's how to extract the needed information:
## 1. Position Features from FEN
The FEN (Forsyth-Edwards Notation) contains all the information about a chess position. Let's extract useful features from it:

import chess
import pandas as pd
import numpy as np
from tqdm import tqdm


def extract_fen_features(df, fen_column='FEN'):
    """Extract rich positional features from FEN strings"""

    print("Extracting position features from FEN...")
    features = []

    for idx, fen in tqdm(df[fen_column].items(), total=len(df)):
        try:
            board = chess.Board(fen)
            feature_dict = {'idx': idx}

            # 1. Piece count features
            pieces = board.piece_map()

            # Initialize counters for each piece type
            for color in [chess.WHITE, chess.BLACK]:
                for piece_type in range(1, 7):  # 1=PAWN, 2=KNIGHT, 3=BISHOP, 4=ROOK, 5=QUEEN, 6=KING
                    color_name = 'white' if color == chess.WHITE else 'black'
                    piece_name = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king'][piece_type - 1]
                    feature_dict[f'piece_count_{color_name}_{piece_name}'] = 0

            # Count pieces
            for square, piece in pieces.items():
                color_name = 'white' if piece.color == chess.WHITE else 'black'
                piece_name = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king'][piece.piece_type - 1]
                feature_dict[f'piece_count_{color_name}_{piece_name}'] += 1

            # 2. Material balance
            material_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}  # Standard piece values (king=0)
            white_material = 0
            black_material = 0

            for square, piece in pieces.items():
                value = material_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

            feature_dict['material_balance'] = white_material - black_material
            feature_dict['total_material'] = white_material + black_material

            # 3. Pawn structure features
            white_pawns = [s for s, p in pieces.items() if p.piece_type == chess.PAWN and p.color == chess.WHITE]
            black_pawns = [s for s, p in pieces.items() if p.piece_type == chess.PAWN and p.color == chess.BLACK]

            # Pawn files (a-h -> 0-7)
            white_pawn_files = [chess.square_file(s) for s in white_pawns]
            black_pawn_files = [chess.square_file(s) for s in black_pawns]

            # Count doubled/isolated pawns
            from collections import Counter

            white_file_counts = Counter(white_pawn_files)
            black_file_counts = Counter(black_pawn_files)

            feature_dict['white_doubled_pawns'] = sum(1 for count in white_file_counts.values() if count > 1)
            feature_dict['black_doubled_pawns'] = sum(1 for count in black_file_counts.values() if count > 1)

            feature_dict['white_isolated_pawns'] = sum(1 for file in white_file_counts.keys()
                                                       if
                                                       file - 1 not in white_file_counts and file + 1 not in white_file_counts)
            feature_dict['black_isolated_pawns'] = sum(1 for file in black_file_counts.keys()
                                                       if
                                                       file - 1 not in black_file_counts and file + 1 not in black_file_counts)

            # 4. Center control
            central_squares = [chess.D4, chess.E4, chess.D5, chess.E5]  # d4, e4, d5, e5

            white_center_control = 0
            black_center_control = 0

            for sq in central_squares:
                white_attackers = len(board.attackers(chess.WHITE, sq))
                black_attackers = len(board.attackers(chess.BLACK, sq))
                white_center_control += white_attackers
                black_center_control += black_attackers

            feature_dict['white_center_control'] = white_center_control
            feature_dict['black_center_control'] = black_center_control
            feature_dict['center_control_balance'] = white_center_control - black_center_control

            # 5. King safety
            white_king_square = board.king(chess.WHITE)
            black_king_square = board.king(chess.BLACK)

            if white_king_square is not None:
                white_king_file = chess.square_file(white_king_square)
                feature_dict['white_king_castled_kingside'] = 1 if white_king_file >= 6 else 0  # g or h file
                feature_dict['white_king_castled_queenside'] = 1 if white_king_file <= 2 else 0  # a, b or c file
                feature_dict['white_king_center'] = 1 if 3 <= white_king_file <= 4 else 0  # d or e file
            else:
                feature_dict['white_king_castled_kingside'] = 0
                feature_dict['white_king_castled_queenside'] = 0
                feature_dict['white_king_center'] = 0

            if black_king_square is not None:
                black_king_file = chess.square_file(black_king_square)
                feature_dict['black_king_castled_kingside'] = 1 if black_king_file >= 6 else 0
                feature_dict['black_king_castled_queenside'] = 1 if black_king_file <= 2 else 0
                feature_dict['black_king_center'] = 1 if 3 <= black_king_file <= 4 else 0
            else:
                feature_dict['black_king_castled_kingside'] = 0
                feature_dict['black_king_castled_queenside'] = 0
                feature_dict['black_king_center'] = 0

            # 6. Development features
            white_knights_developed = sum(1 for s, p in pieces.items()
                                          if p.piece_type == chess.KNIGHT and p.color == chess.WHITE
                                          and not chess.square_rank(s) == 0)  # Not on first rank

            white_bishops_developed = sum(1 for s, p in pieces.items()
                                          if p.piece_type == chess.BISHOP and p.color == chess.WHITE
                                          and not chess.square_rank(s) == 0)  # Not on first rank

            black_knights_developed = sum(1 for s, p in pieces.items()
                                          if p.piece_type == chess.KNIGHT and p.color == chess.BLACK
                                          and not chess.square_rank(s) == 7)  # Not on eighth rank

            black_bishops_developed = sum(1 for s, p in pieces.items()
                                          if p.piece_type == chess.BISHOP and p.color == chess.BLACK
                                          and not chess.square_rank(s) == 7)  # Not on eighth rank

            feature_dict['white_minor_pieces_developed'] = white_knights_developed + white_bishops_developed
            feature_dict['black_minor_pieces_developed'] = black_knights_developed + black_bishops_developed

            # 7. Game phase estimation
            total_pieces = len(
                [p for p in pieces.values() if p.piece_type != chess.KING and p.piece_type != chess.PAWN])
            feature_dict['opening_phase'] = 1 if total_pieces >= 6 and feature_dict['total_material'] >= 24 else 0
            feature_dict['middlegame_phase'] = 1 if 3 <= total_pieces < 6 or (
                        total_pieces >= 6 and feature_dict['total_material'] < 24) else 0
            feature_dict['endgame_phase'] = 1 if total_pieces < 3 or feature_dict['total_material'] < 15 else 0

            # 8. Whose turn
            feature_dict['white_to_move'] = 1 if board.turn == chess.WHITE else 0

            features.append(feature_dict)

        except Exception as e:
            print(f"Error processing FEN {fen}: {e}")
            # Add empty features
            features.append({'idx': idx})

    # Create DataFrame from features
    fen_features_df = pd.DataFrame(features).set_index('idx')

    # Fill NaN values
    fen_features_df = fen_features_df.fillna(0)

    print(f"Extracted {fen_features_df.shape[1]} position features from FEN")
    return fen_features_df

## 2. Opening Move Features from Moves Column
The `Moves` column likely contains the move sequence. We can extract opening-specific features from it:

def extract_opening_move_features(df, moves_column='Moves'):
    """Extract features from the opening moves of each puzzle"""

    print("Extracting features from opening moves...")
    features = []

    # Common first moves and their frequencies in different openings
    common_first_moves = ['e4', 'd4', 'c4', 'Nf3', 'g3', 'f4', 'b3', 'Nc3']
    common_responses_e4 = ['e5', 'c5', 'e6', 'c6', 'd6', 'd5', 'g6', 'Nf6']  # After e4
    common_responses_d4 = ['d5', 'Nf6', 'f5', 'e6', 'c5', 'g6', 'd6', 'c6']  # After d4

    for idx, moves_str in tqdm(df[moves_column].items(), total=len(df)):
        feature_dict = {'idx': idx}

        try:
            # Parse moves
            if pd.isna(moves_str) or not moves_str:
                features.append(feature_dict)
                continue

            # Normalize move notation
            moves_str = moves_str.replace('O-O-O', 'Qa1').replace('O-O', 'Kg1')  # Replace castling for parsing
            moves = moves_str.split()

            # Initialize all move features to 0
            for move in common_first_moves:
                feature_dict[f'first_move_{move}'] = 0

            for move in common_responses_e4:
                feature_dict[f'response_to_e4_{move}'] = 0

            for move in common_responses_d4:
                feature_dict[f'response_to_d4_{move}'] = 0

            # Check first few moves
            if len(moves) >= 1:
                # First white move
                first_move = moves[0]
                for move in common_first_moves:
                    if move in first_move:  # Using 'in' to handle moves like 'e4+' or 'e4#'
                        feature_dict[f'first_move_{move}'] = 1
                        break

                # First black response
                if len(moves) >= 2:
                    response = moves[1]

                    # Check response to e4
                    if 'e4' in first_move:
                        for move in common_responses_e4:
                            if move in response:
                                feature_dict[f'response_to_e4_{move}'] = 1
                                break

                    # Check response to d4
                    elif 'd4' in first_move:
                        for move in common_responses_d4:
                            if move in response:
                                feature_dict[f'response_to_d4_{move}'] = 1
                                break

            # Common opening sequences
            opening_sequences = {
                'ruy_lopez': ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5'],
                'italian_game': ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4'],
                'sicilian_defense': ['e4', 'c5'],
                'french_defense': ['e4', 'e6'],
                'caro_kann': ['e4', 'c6'],
                'queens_gambit': ['d4', 'd5', 'c4'],
                'kings_indian': ['d4', 'Nf6', 'c4', 'g6'],
                'english_opening': ['c4'],
                'queens_pawn': ['d4', 'd5'],
                'dutch_defense': ['d4', 'f5'],
                'pirc_defense': ['e4', 'd6'],
                'scandinavian': ['e4', 'd5']
            }

            # Initialize sequence features
            for name in opening_sequences:
                feature_dict[f'opening_seq_{name}'] = 0

            # Check for each sequence
            for name, sequence in opening_sequences.items():
                if len(moves) >= len(sequence):
                    # Check if the first n moves match the sequence
                    matches = True
                    for i, move in enumerate(sequence):
                        if move not in moves[i]:  # Using 'in' to be more flexible
                            matches = False
                            break

                    if matches:
                        feature_dict[f'opening_seq_{name}'] = 1

            # Add number of moves
            feature_dict['num_moves'] = len(moves)

            # Check for early castling
            kingside_castle_white = False
            queenside_castle_white = False
            kingside_castle_black = False
            queenside_castle_black = False

            for i, move in enumerate(moves[:20]):  # Check first 20 moves
                if i % 2 == 0:  # White's move
                    if 'O-O-O' in move or 'Qa1' in move:
                        queenside_castle_white = True
                    elif 'O-O' in move or 'Kg1' in move:
                        kingside_castle_white = True
                else:  # Black's move
                    if 'O-O-O' in move or 'Qa8' in move:
                        queenside_castle_black = True
                    elif 'O-O' in move or 'Kg8' in move:
                        kingside_castle_black = True

            feature_dict['early_kingside_castle_white'] = 1 if kingside_castle_white and len(moves) <= 12 else 0
            feature_dict['early_queenside_castle_white'] = 1 if queenside_castle_white and len(moves) <= 12 else 0
            feature_dict['early_kingside_castle_black'] = 1 if kingside_castle_black and len(moves) <= 13 else 0
            feature_dict['early_queenside_castle_black'] = 1 if queenside_castle_black and len(moves) <= 13 else 0

            features.append(feature_dict)

        except Exception as e:
            print(f"Error processing moves {moves_str}: {e}")
            features.append({'idx': idx})

    # Create DataFrame from features
    moves_features_df = pd.DataFrame(features).set_index('idx')

    # Fill NaN values
    moves_features_df = moves_features_df.fillna(0)

    print(f"Extracted {moves_features_df.shape[1]} features from opening moves")
    return moves_features_df

## 3. ECO Code Inference
We can infer approximate ECO codes (Encyclopedia of Chess Openings) based on opening moves:


def infer_eco_codes(df, moves_column='Moves'):
    """Infer ECO codes from move sequences"""

    print("Inferring ECO codes from moves...")
    features = []

    # ECO code patterns (simplified)
    eco_patterns = {
        'A': {
            'moves': [
                ['c4'],  # English
                ['Nf3', 'd5', 'c4'],  # Reti
                ['g3'],  # King's Fianchetto
                ['Nf3', 'Nf6', 'c4', 'g6', 'b4'],  # Sokolsky
                ['f4']  # Bird's Opening
            ],
            'names': ['English', 'Reti', 'Kings_Fianchetto', 'Sokolsky', 'Birds']
        },
        'B': {
            'moves': [
                ['e4', 'c5'],  # Sicilian
                ['e4', 'c6'],  # Caro-Kann
                ['e4', 'd6'],  # Pirc/Modern
                ['e4', 'Nf6'],  # Alekhine
                ['e4', 'd5']  # Scandinavian
            ],
            'names': ['Sicilian', 'Caro_Kann', 'Pirc_Modern', 'Alekhine', 'Scandinavian']
        },
        'C': {
            'moves': [
                ['e4', 'e5'],  # Open Games
                ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4'],  # Italian
                ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5'],  # Ruy Lopez
                ['e4', 'e6'],  # French
                ['e4', 'e5', 'f4']  # King's Gambit
            ],
            'names': ['Open_Game', 'Italian', 'Ruy_Lopez', 'French', 'Kings_Gambit']
        },
        'D': {
            'moves': [
                ['d4', 'd5', 'c4'],  # Queen's Gambit
                ['d4', 'd5', 'c4', 'c6'],  # Slav
                ['d4', 'Nf6', 'c4', 'g6', 'Nc3', 'd5'],  # Grunfeld
                ['d4', 'd5', 'c4', 'e6', 'Nc3', 'c6']  # Semi-Slav
            ],
            'names': ['Queens_Gambit', 'Slav', 'Grunfeld', 'Semi_Slav']
        },
        'E': {
            'moves': [
                ['d4', 'Nf6', 'c4', 'e6'],  # Nimzo/Queen's Indian
                ['d4', 'Nf6', 'c4', 'g6'],  # King's Indian
                ['d4', 'Nf6', 'c4', 'e6', 'g3'],  # Catalan
                ['d4', 'Nf6', 'c4', 'c5']  # Benoni
            ],
            'names': ['Nimzo_Queens_Indian', 'Kings_Indian', 'Catalan', 'Benoni']
        }
    }

    for idx, moves_str in tqdm(df[moves_column].items(), total=len(df)):
        feature_dict = {'idx': idx}

        try:
            # Parse moves
            if pd.isna(moves_str) or not moves_str:
                for eco in eco_patterns:
                    feature_dict[f'eco_{eco}'] = 0
                    for name in eco_patterns[eco]['names']:
                        feature_dict[f'eco_{eco}_{name}'] = 0
                features.append(feature_dict)
                continue

            # Normalize move notation
            moves_str = moves_str.replace('O-O-O', 'Qa1').replace('O-O', 'Kg1')
            moves = moves_str.split()

            # Initialize ECO features
            for eco in eco_patterns:
                feature_dict[f'eco_{eco}'] = 0
                for name in eco_patterns[eco]['names']:
                    feature_dict[f'eco_{eco}_{name}'] = 0

            # Check each ECO category
            for eco, patterns in eco_patterns.items():
                for i, move_pattern in enumerate(patterns['moves']):
                    if len(moves) >= len(move_pattern):
                        matches = True
                        for j, expected_move in enumerate(move_pattern):
                            if expected_move not in moves[j]:
                                matches = False
                                break

                        if matches:
                            feature_dict[f'eco_{eco}'] = 1
                            feature_dict[f'eco_{eco}_{patterns["names"][i]}'] = 1
                            break

            features.append(feature_dict)

        except Exception as e:
            print(f"Error inferring ECO for {moves_str}: {e}")
            features.append({'idx': idx})

    # Create DataFrame from features
    eco_features_df = pd.DataFrame(features).set_index('idx')

    # Fill NaN values
    eco_features_df = eco_features_df.fillna(0)

    print(f"Inferred ECO codes with {eco_features_df.shape[1]} features")
    return eco_features_df

## 4. Combining All Features
Now let's combine all these features and use them to predict the missing opening tags:


def predict_missing_opening_tags(df, tag_column='OpeningTags'):
    """Predict missing opening tags using position and move features"""

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

    # Extract primary opening family from tags
    def extract_primary_family(tag_str):
        if pd.isna(tag_str) or not tag_str:
            return 'Unknown'
        # Get first tag if multiple tags exist
        first_tag = tag_str.split()[0]
        # Get first component of the tag (typically the family name)
        family = first_tag.split('_')[0] if '_' in first_tag else first_tag
        return family

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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

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

## 5. Creating Final Feature Set with Predicted Tags
Finally, we integrate the predictions with our opening feature engineering:


    def complete_feature_engineering(df, tag_column='OpeningTags'):
        """Complete pipeline for feature engineering with opening tag prediction"""

        # Step 1: Extract position and move features
        position_features = extract_fen_features(df)
        move_features = extract_opening_move_features(df)
        eco_features = infer_eco_codes(df)

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

        # Use your original opening feature engineering function
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
            eco_features
        ], axis=1)

        # Fill any remaining NaN values
        final_features = final_features.fillna(0)

        print(f"Created final feature set with {final_features.shape[1]} features")
        return final_features, model, predictions

# ## How This Addresses Your Dataset's Challenges
# 1. **FEN Position Analysis**: Extracts piece placement, material balance, pawn structure, and other positional features that strongly correlate with specific openings.
# 2. **Move Sequence Analysis**: Analyzes the actual moves played to identify common opening patterns and transpositions.
# 3. **ECO Code Inference**: Provides a structured way to categorize openings into the standard chess opening encyclopedia system.
# 4. **Confidence Scores**: Only uses high-confidence predictions to avoid introducing noise.
# 5. **Combined Features**: The final feature set combines original tags, predicted tags, position features, and move features to give your model the richest possible information.
#
# This approach should significantly increase the percentage of puzzles with opening tag information while maintaining high data quality through the confidence thresholds.

