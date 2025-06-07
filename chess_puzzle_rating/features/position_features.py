"""
Module for extracting features from chess positions in FEN format.
"""

import chess
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter


def extract_fen_features(df, fen_column='FEN'):
    """
    Extract rich positional features from FEN strings.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles with FEN strings
    fen_column : str, optional
        Name of the column containing FEN strings, by default 'FEN'
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with extracted position features
    """
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
            white_file_counts = Counter(white_pawn_files)
            black_file_counts = Counter(black_pawn_files)

            feature_dict['white_doubled_pawns'] = sum(1 for count in white_file_counts.values() if count > 1)
            feature_dict['black_doubled_pawns'] = sum(1 for count in black_file_counts.values() if count > 1)

            feature_dict['white_isolated_pawns'] = sum(1 for file in white_file_counts.keys()
                                                    if file - 1 not in white_file_counts and file + 1 not in white_file_counts)
            feature_dict['black_isolated_pawns'] = sum(1 for file in black_file_counts.keys()
                                                    if file - 1 not in black_file_counts and file + 1 not in black_file_counts)

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
                                        and not chess.square_rank(s) == 7)  # Not on last rank
            black_bishops_developed = sum(1 for s, p in pieces.items()
                                        if p.piece_type == chess.BISHOP and p.color == chess.BLACK
                                        and not chess.square_rank(s) == 7)  # Not on last rank

            feature_dict['white_minor_pieces_developed'] = white_knights_developed + white_bishops_developed
            feature_dict['black_minor_pieces_developed'] = black_knights_developed + black_bishops_developed

            # 7. Mobility features (simplified)
            white_mobility = 0
            black_mobility = 0

            # Count legal moves for each piece
            for square, piece in pieces.items():
                # Create a new board with just this piece to count its moves
                temp_board = chess.Board(None)
                temp_board.set_piece_at(square, piece)
                
                # Set the turn to match the piece color
                temp_board.turn = piece.color
                
                # Count legal moves
                moves = list(temp_board.legal_moves)
                if piece.color == chess.WHITE:
                    white_mobility += len(moves)
                else:
                    black_mobility += len(moves)

            feature_dict['white_mobility'] = white_mobility
            feature_dict['black_mobility'] = black_mobility
            feature_dict['mobility_balance'] = white_mobility - black_mobility

            features.append(feature_dict)

        except Exception as e:
            print(f"Error extracting features for FEN {fen}: {e}")
            features.append({'idx': idx})

    # Create DataFrame from features
    fen_features_df = pd.DataFrame(features).set_index('idx')

    # Fill NaN values
    fen_features_df = fen_features_df.fillna(0)

    print(f"Extracted {fen_features_df.shape[1]} position features from FEN")
    return fen_features_df