"""
Module for extracting features from chess move sequences.
"""

import chess
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def extract_opening_move_features(df, moves_column='Moves'):
    """
    Extract features from the opening moves of each puzzle.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles with move sequences
    moves_column : str, optional
        Name of the column containing move sequences, by default 'Moves'

    Returns
    -------
    pandas.DataFrame
        DataFrame with extracted opening move features
    """
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

            feature_dict['white_kingside_castle'] = 1 if kingside_castle_white else 0
            feature_dict['white_queenside_castle'] = 1 if queenside_castle_white else 0
            feature_dict['black_kingside_castle'] = 1 if kingside_castle_black else 0
            feature_dict['black_queenside_castle'] = 1 if queenside_castle_black else 0

            features.append(feature_dict)

        except Exception as e:
            print(f"Error extracting move features for {moves_str}: {e}")
            features.append({'idx': idx})

    # Create DataFrame from features
    moves_features_df = pd.DataFrame(features).set_index('idx')

    # Fill NaN values
    moves_features_df = moves_features_df.fillna(0)

    print(f"Extracted {moves_features_df.shape[1]} features from opening moves")
    return moves_features_df


def infer_eco_codes(df, moves_column='Moves'):
    """
    Infer ECO codes from move sequences.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles with move sequences
    moves_column : str, optional
        Name of the column containing move sequences, by default 'Moves'

    Returns
    -------
    pandas.DataFrame
        DataFrame with inferred ECO code features
    """
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


def analyze_move_sequence(df, fen_column='FEN', moves_column='Moves'):
    """
    Analyze move sequences to extract advanced features:
    - Move forcing factors (checks, captures, threats)
    - Material sacrifices in solutions
    - Move depth complexity

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles with FEN positions and move sequences
    fen_column : str, optional
        Name of the column containing FEN strings, by default 'FEN'
    moves_column : str, optional
        Name of the column containing move sequences, by default 'Moves'

    Returns
    -------
    pandas.DataFrame
        DataFrame with extracted move sequence analysis features
    """
    print("Analyzing move sequences for forcing factors, sacrifices, and complexity...")
    features = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        feature_dict = {'idx': idx}

        try:
            # Get FEN and moves
            fen = row[fen_column] if fen_column in row and not pd.isna(row[fen_column]) else None
            moves_str = row[moves_column] if moves_column in row and not pd.isna(row[moves_column]) else None

            if fen is None or moves_str is None or not moves_str:
                features.append(feature_dict)
                continue

            # Initialize board from FEN
            board = chess.Board(fen)

            # Normalize move notation
            moves_str = moves_str.replace('O-O-O', 'Qa1').replace('O-O', 'Kg1')  # Replace castling for parsing
            moves = moves_str.split()

            # Initialize counters
            checks = 0
            captures = 0
            threats = 0
            material_sacrifices = 0
            max_depth = 0

            # Material values for sacrifice detection
            material_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}  # Standard piece values (king=0)

            # Track material balance changes
            material_balance_before = 0
            for square, piece in board.piece_map().items():
                value = material_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    material_balance_before += value
                else:
                    material_balance_before -= value

            # Process each move
            for i, move_str in enumerate(moves):
                # Try to parse the move
                try:
                    # Clean up move string (remove check/mate symbols)
                    clean_move_str = move_str.replace('+', '').replace('#', '')

                    # Parse move
                    move = None
                    for m in board.legal_moves:
                        if clean_move_str in board.san(m):
                            move = m
                            break

                    if move is None:
                        continue

                    # Check if move is a check
                    if board.gives_check(move):
                        checks += 1

                    # Check if move is a capture
                    if board.is_capture(move):
                        captures += 1

                    # Material balance before move
                    pre_move_balance = 0
                    for square, piece in board.piece_map().items():
                        value = material_values[piece.piece_type]
                        if piece.color == chess.WHITE:
                            pre_move_balance += value
                        else:
                            pre_move_balance -= value

                    # Make the move
                    board.push(move)

                    # Material balance after move
                    post_move_balance = 0
                    for square, piece in board.piece_map().items():
                        value = material_values[piece.piece_type]
                        if piece.color == chess.WHITE:
                            post_move_balance += value
                        else:
                            post_move_balance -= value

                    # Check for material sacrifice
                    # A sacrifice is when a player makes a move that loses material
                    # but potentially gains positional advantage
                    is_white_move = (i % 2 == 0)
                    if is_white_move and post_move_balance < pre_move_balance:
                        material_sacrifices += 1
                    elif not is_white_move and post_move_balance > pre_move_balance:
                        material_sacrifices += 1

                    # Check for threats (attacking opponent's pieces)
                    current_threats = 0
                    for square, piece in board.piece_map().items():
                        # Only count threats to opponent's pieces
                        if piece.color != board.turn:
                            attackers = board.attackers(board.turn, square)
                            if attackers:
                                current_threats += 1

                    if current_threats > 0:
                        threats += 1

                    # Update max depth
                    max_depth = max(max_depth, i + 1)

                except Exception as e:
                    # If we can't parse the move, just continue to the next one
                    continue

            # Calculate move forcing factor (weighted sum of checks, captures, threats)
            move_forcing_factor = (2 * checks + captures + threats) / max(1, len(moves))

            # Calculate material sacrifice ratio
            sacrifice_ratio = material_sacrifices / max(1, len(moves))

            # Calculate move depth complexity
            # This is a simple measure based on the number of moves and forcing factors
            depth_complexity = max_depth * (1 + move_forcing_factor)

            # Add features to dictionary
            feature_dict['move_checks_count'] = checks
            feature_dict['move_captures_count'] = captures
            feature_dict['move_threats_count'] = threats
            feature_dict['move_forcing_factor'] = move_forcing_factor
            feature_dict['material_sacrifices_count'] = material_sacrifices
            feature_dict['material_sacrifice_ratio'] = sacrifice_ratio
            feature_dict['move_depth'] = max_depth
            feature_dict['move_depth_complexity'] = depth_complexity

            # Add features for the distribution of forcing moves
            feature_dict['pct_moves_with_check'] = checks / max(1, len(moves))
            feature_dict['pct_moves_with_capture'] = captures / max(1, len(moves))
            feature_dict['pct_moves_with_threat'] = threats / max(1, len(moves))

        except Exception as e:
            print(f"Error analyzing move sequence for index {idx}: {e}")

        features.append(feature_dict)

    # Create DataFrame from features
    move_analysis_df = pd.DataFrame(features).set_index('idx')

    # Fill NaN values
    move_analysis_df = move_analysis_df.fillna(0)

    print(f"Extracted {move_analysis_df.shape[1]} features from move sequence analysis")
    return move_analysis_df
