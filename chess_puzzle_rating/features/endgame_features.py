"""
Module for extracting endgame-specific features from chess positions.

This module provides functions to extract features specific to chess endgames:
1. Endgame pattern recognition (king and pawn endgames, rook endgames, etc.)
2. Tablebase distance-to-mate features for simple endgames
3. Features for common endgame motifs (opposition, zugzwang, etc.)

These features can be used to enhance the accuracy of chess puzzle rating prediction
in endgame positions.
"""

import chess
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
from ..utils.progress import get_logger


def is_endgame(board):
    """
    Determine if a position is an endgame.

    Parameters
    ----------
    board : chess.Board
        The chess board to analyze

    Returns
    -------
    bool
        True if the position is an endgame, False otherwise
    """
    # Count the total material on the board
    pieces = board.piece_map()

    # Count major pieces (queens and rooks)
    queens = sum(1 for _, p in pieces.items() if p.piece_type == chess.QUEEN)
    rooks = sum(1 for _, p in pieces.items() if p.piece_type == chess.ROOK)

    # Count minor pieces (bishops and knights)
    minor_pieces = sum(1 for _, p in pieces.items() 
                      if p.piece_type == chess.BISHOP or p.piece_type == chess.KNIGHT)

    # Endgame conditions:
    # 1. No queens, or
    # 2. Each side has at most one piece (excluding pawns and kings)
    if queens == 0 or (queens + rooks + minor_pieces <= 2):
        return True

    # Alternative definition: less than 25% of the material remains
    material_values = {
        chess.PAWN: 1, 
        chess.KNIGHT: 3, 
        chess.BISHOP: 3, 
        chess.ROOK: 5, 
        chess.QUEEN: 9
    }

    total_material = sum(material_values[p.piece_type] for _, p in pieces.items() 
                         if p.piece_type != chess.KING)

    # Full material would be 78 (16 pawns, 4 knights, 4 bishops, 4 rooks, 2 queens)
    return total_material <= 20  # Less than ~25% of material remains


def identify_endgame_type(board):
    """
    Identify the type of endgame.

    Parameters
    ----------
    board : chess.Board
        The chess board to analyze

    Returns
    -------
    dict
        Dictionary with boolean flags for different endgame types
    """
    if not is_endgame(board):
        return {
            'is_endgame': False,
            'king_pawn_endgame': False,
            'rook_endgame': False,
            'minor_piece_endgame': False,
            'queen_endgame': False,
            'opposite_colored_bishops': False,
            'same_colored_bishops': False
        }

    pieces = board.piece_map()

    # Count pieces by type
    piece_counts = defaultdict(int)
    for _, p in pieces.items():
        if p.piece_type != chess.KING:
            piece_counts[p.piece_type] += 1

    # Identify specific endgame types
    king_pawn_endgame = (piece_counts[chess.PAWN] > 0 and 
                         piece_counts[chess.KNIGHT] == 0 and
                         piece_counts[chess.BISHOP] == 0 and
                         piece_counts[chess.ROOK] == 0 and
                         piece_counts[chess.QUEEN] == 0)

    rook_endgame = piece_counts[chess.ROOK] > 0 and piece_counts[chess.QUEEN] == 0

    minor_piece_endgame = ((piece_counts[chess.KNIGHT] > 0 or piece_counts[chess.BISHOP] > 0) and
                          piece_counts[chess.ROOK] == 0 and
                          piece_counts[chess.QUEEN] == 0)

    queen_endgame = piece_counts[chess.QUEEN] > 0

    # Check for opposite-colored bishops
    white_bishops = [s for s, p in pieces.items() 
                    if p.piece_type == chess.BISHOP and p.color == chess.WHITE]
    black_bishops = [s for s, p in pieces.items() 
                    if p.piece_type == chess.BISHOP and p.color == chess.BLACK]

    opposite_colored_bishops = False
    same_colored_bishops = False

    if len(white_bishops) == 1 and len(black_bishops) == 1:
        white_bishop_square_color = (chess.square_rank(white_bishops[0]) + chess.square_file(white_bishops[0])) % 2
        black_bishop_square_color = (chess.square_rank(black_bishops[0]) + chess.square_file(black_bishops[0])) % 2

        opposite_colored_bishops = white_bishop_square_color != black_bishop_square_color
        same_colored_bishops = white_bishop_square_color == black_bishop_square_color

    return {
        'is_endgame': True,
        'king_pawn_endgame': king_pawn_endgame,
        'rook_endgame': rook_endgame,
        'minor_piece_endgame': minor_piece_endgame,
        'queen_endgame': queen_endgame,
        'opposite_colored_bishops': opposite_colored_bishops,
        'same_colored_bishops': same_colored_bishops
    }


def analyze_king_position(board):
    """
    Analyze king positions in endgames.

    Parameters
    ----------
    board : chess.Board
        The chess board to analyze

    Returns
    -------
    dict
        Dictionary with king position features
    """
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)

    if white_king_square is None or black_king_square is None:
        return {}

    white_king_file = chess.square_file(white_king_square)
    white_king_rank = chess.square_rank(white_king_square)
    black_king_file = chess.square_file(black_king_square)
    black_king_rank = chess.square_rank(black_king_square)

    # Distance to center
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    white_king_center_distance = min(chess.square_distance(white_king_square, sq) for sq in center_squares)
    black_king_center_distance = min(chess.square_distance(black_king_square, sq) for sq in center_squares)

    # Distance between kings
    king_distance = chess.square_distance(white_king_square, black_king_square)

    # Check for opposition
    has_opposition = (king_distance == 2 and 
                     ((white_king_file == black_king_file) or 
                      (white_king_rank == black_king_rank)))

    # Distance to edge
    white_king_edge_distance = min(
        white_king_file, 7 - white_king_file,
        white_king_rank, 7 - white_king_rank
    )
    black_king_edge_distance = min(
        black_king_file, 7 - black_king_file,
        black_king_rank, 7 - black_king_rank
    )

    return {
        'white_king_center_distance': white_king_center_distance,
        'black_king_center_distance': black_king_center_distance,
        'king_distance': king_distance,
        'has_opposition': int(has_opposition),
        'white_king_edge_distance': white_king_edge_distance,
        'black_king_edge_distance': black_king_edge_distance
    }


def analyze_pawn_structure_endgame(board):
    """
    Analyze pawn structure in endgames.

    Parameters
    ----------
    board : chess.Board
        The chess board to analyze

    Returns
    -------
    dict
        Dictionary with pawn structure features specific to endgames
    """
    pieces = board.piece_map()

    # Get pawns
    white_pawns = [s for s, p in pieces.items() if p.piece_type == chess.PAWN and p.color == chess.WHITE]
    black_pawns = [s for s, p in pieces.items() if p.piece_type == chess.PAWN and p.color == chess.BLACK]

    # Check for passed pawns
    white_passed_pawns = 0
    for pawn in white_pawns:
        pawn_file = chess.square_file(pawn)
        pawn_rank = chess.square_rank(pawn)
        is_passed = True

        # Check if any black pawns can block this pawn
        for file_offset in [-1, 0, 1]:
            check_file = pawn_file + file_offset
            if 0 <= check_file <= 7:
                for black_pawn in black_pawns:
                    black_pawn_file = chess.square_file(black_pawn)
                    black_pawn_rank = chess.square_rank(black_pawn)

                    if black_pawn_file == check_file and black_pawn_rank > pawn_rank:
                        is_passed = False
                        break

            if not is_passed:
                break

        if is_passed:
            white_passed_pawns += 1

    black_passed_pawns = 0
    for pawn in black_pawns:
        pawn_file = chess.square_file(pawn)
        pawn_rank = chess.square_rank(pawn)
        is_passed = True

        # Check if any white pawns can block this pawn
        for file_offset in [-1, 0, 1]:
            check_file = pawn_file + file_offset
            if 0 <= check_file <= 7:
                for white_pawn in white_pawns:
                    white_pawn_file = chess.square_file(white_pawn)
                    white_pawn_rank = chess.square_rank(white_pawn)

                    if white_pawn_file == check_file and white_pawn_rank < pawn_rank:
                        is_passed = False
                        break

            if not is_passed:
                break

        if is_passed:
            black_passed_pawns += 1

    # Calculate most advanced pawn for each side
    white_most_advanced = max([chess.square_rank(p) for p in white_pawns]) if white_pawns else 0
    black_most_advanced = 7 - min([chess.square_rank(p) for p in black_pawns]) if black_pawns else 0

    # Check for connected passed pawns
    white_connected_passed = 0
    black_connected_passed = 0

    # This would require more complex analysis, simplified version:
    if white_passed_pawns >= 2:
        white_passed_pawn_files = [chess.square_file(p) for p in white_pawns if any(
            chess.square_file(bp) != chess.square_file(p) - 1 and
            chess.square_file(bp) != chess.square_file(p) and
            chess.square_file(bp) != chess.square_file(p) + 1
            for bp in black_pawns
        )]

        for i in range(len(white_passed_pawn_files) - 1):
            if abs(white_passed_pawn_files[i] - white_passed_pawn_files[i+1]) == 1:
                white_connected_passed += 1

    if black_passed_pawns >= 2:
        black_passed_pawn_files = [chess.square_file(p) for p in black_pawns if any(
            chess.square_file(wp) != chess.square_file(p) - 1 and
            chess.square_file(wp) != chess.square_file(p) and
            chess.square_file(wp) != chess.square_file(p) + 1
            for wp in white_pawns
        )]

        for i in range(len(black_passed_pawn_files) - 1):
            if abs(black_passed_pawn_files[i] - black_passed_pawn_files[i+1]) == 1:
                black_connected_passed += 1

    return {
        'white_passed_pawns': white_passed_pawns,
        'black_passed_pawns': black_passed_pawns,
        'white_most_advanced_pawn_rank': white_most_advanced,
        'black_most_advanced_pawn_rank': black_most_advanced,
        'white_connected_passed_pawns': white_connected_passed,
        'black_connected_passed_pawns': black_connected_passed
    }


def detect_zugzwang_potential(board):
    """
    Detect potential zugzwang situations in endgames.

    Parameters
    ----------
    board : chess.Board
        The chess board to analyze

    Returns
    -------
    dict
        Dictionary with zugzwang potential features
    """
    # This is a simplified heuristic for zugzwang potential
    # True zugzwang detection would require engine analysis

    # Indicators of potential zugzwang:
    # 1. Limited material (endgame)
    # 2. Kings in close proximity
    # 3. Limited mobility for the side to move

    if not is_endgame(board):
        return {'zugzwang_potential': 0}

    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)

    if white_king_square is None or black_king_square is None:
        return {'zugzwang_potential': 0}

    # Check king proximity
    king_distance = chess.square_distance(white_king_square, black_king_square)

    # Count legal moves for the side to move
    legal_moves = list(board.legal_moves)
    move_count = len(legal_moves)

    # Heuristic for zugzwang potential
    zugzwang_potential = 0

    if king_distance <= 2 and move_count <= 5:
        zugzwang_potential = 3  # High potential
    elif king_distance <= 3 and move_count <= 8:
        zugzwang_potential = 2  # Medium potential
    elif king_distance <= 4 and move_count <= 10:
        zugzwang_potential = 1  # Low potential

    return {'zugzwang_potential': zugzwang_potential}


def estimate_tablebase_distance(board):
    """
    Estimate distance to mate for simple endgames that would be in tablebases.

    This is a simplified estimation as actual tablebase access would require
    external libraries or API calls.

    Parameters
    ----------
    board : chess.Board
        The chess board to analyze

    Returns
    -------
    dict
        Dictionary with tablebase distance features
    """
    pieces = board.piece_map()
    piece_count = len(pieces)

    # We can only estimate for positions that would be in 5-piece or fewer tablebases
    if piece_count > 5:
        return {
            'in_tablebase': 0,
            'tablebase_dtm_estimate': 0,
            'theoretical_win': 0,
            'theoretical_draw': 0
        }

    # Count material by type and color
    white_material = defaultdict(int)
    black_material = defaultdict(int)

    for _, p in pieces.items():
        if p.color == chess.WHITE:
            white_material[p.piece_type] += 1
        else:
            black_material[p.piece_type] += 1

    # Check if this is a theoretical win or draw
    theoretical_win = False
    theoretical_draw = False

    # KQ vs K is a win
    if (white_material[chess.QUEEN] == 1 and sum(white_material.values()) == 2 and sum(black_material.values()) == 1) or \
       (black_material[chess.QUEEN] == 1 and sum(black_material.values()) == 2 and sum(white_material.values()) == 1):
        theoretical_win = True

    # KR vs K is a win
    elif (white_material[chess.ROOK] == 1 and sum(white_material.values()) == 2 and sum(black_material.values()) == 1) or \
         (black_material[chess.ROOK] == 1 and sum(black_material.values()) == 2 and sum(white_material.values()) == 1):
        theoretical_win = True

    # KB+KN vs K is a win (with proper technique)
    elif ((white_material[chess.BISHOP] == 1 and white_material[chess.KNIGHT] == 1 and sum(white_material.values()) == 3 and sum(black_material.values()) == 1) or
          (black_material[chess.BISHOP] == 1 and black_material[chess.KNIGHT] == 1 and sum(black_material.values()) == 3 and sum(white_material.values()) == 1)):
        theoretical_win = True

    # K+P vs K can be a win depending on position
    elif ((white_material[chess.PAWN] == 1 and sum(white_material.values()) == 2 and sum(black_material.values()) == 1) or
          (black_material[chess.PAWN] == 1 and sum(black_material.values()) == 2 and sum(white_material.values()) == 1)):
        # Simplified check - would need more analysis for accurate assessment
        if white_material[chess.PAWN] == 1:
            pawn_squares = [s for s, p in pieces.items() if p.piece_type == chess.PAWN and p.color == chess.WHITE]
            if pawn_squares:
                pawn_rank = chess.square_rank(pawn_squares[0])
                if pawn_rank >= 5:  # Advanced pawn
                    theoretical_win = True
                else:
                    # Would need king position analysis for accuracy
                    theoretical_draw = True
        else:
            pawn_squares = [s for s, p in pieces.items() if p.piece_type == chess.PAWN and p.color == chess.BLACK]
            if pawn_squares:
                pawn_rank = chess.square_rank(pawn_squares[0])
                if pawn_rank <= 2:  # Advanced pawn
                    theoretical_win = True
                else:
                    # Would need king position analysis for accuracy
                    theoretical_draw = True

    # KN vs K or KB vs K is a draw
    elif ((white_material[chess.KNIGHT] == 1 and sum(white_material.values()) == 2 and sum(black_material.values()) == 1) or
          (black_material[chess.KNIGHT] == 1 and sum(black_material.values()) == 2 and sum(white_material.values()) == 1) or
          (white_material[chess.BISHOP] == 1 and sum(white_material.values()) == 2 and sum(black_material.values()) == 1) or
          (black_material[chess.BISHOP] == 1 and sum(black_material.values()) == 2 and sum(white_material.values()) == 1)):
        theoretical_draw = True

    # K vs K is a draw
    elif sum(white_material.values()) == 1 and sum(black_material.values()) == 1:
        theoretical_draw = True

    # Estimate DTM (distance to mate)
    dtm_estimate = 0

    if theoretical_win:
        # Rough estimates based on typical DTM for these endgames
        if white_material[chess.QUEEN] == 1 or black_material[chess.QUEEN] == 1:
            dtm_estimate = 10
        elif white_material[chess.ROOK] == 1 or black_material[chess.ROOK] == 1:
            dtm_estimate = 16
        elif (white_material[chess.BISHOP] == 1 and white_material[chess.KNIGHT] == 1) or \
             (black_material[chess.BISHOP] == 1 and black_material[chess.KNIGHT] == 1):
            dtm_estimate = 33
        elif white_material[chess.PAWN] == 1 or black_material[chess.PAWN] == 1:
            dtm_estimate = 15

    return {
        'in_tablebase': 1 if piece_count <= 5 else 0,
        'tablebase_dtm_estimate': dtm_estimate,
        'theoretical_win': 1 if theoretical_win else 0,
        'theoretical_draw': 1 if theoretical_draw else 0
    }


def extract_endgame_features(df, fen_column='FEN'):
    """
    Extract endgame-specific features from FEN strings.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess puzzles with FEN strings
    fen_column : str, optional
        Name of the column containing FEN strings, by default 'FEN'

    Returns
    -------
    pandas.DataFrame
        DataFrame with extracted endgame features
    """
    logger = get_logger()
    logger.info("Extracting endgame features from FEN...")
    features = []

    for idx, fen in tqdm(df[fen_column].items(), total=len(df)):
        try:
            board = chess.Board(fen)
            feature_dict = {'idx': idx}

            # Check if this is an endgame
            endgame_type = identify_endgame_type(board)
            feature_dict.update(endgame_type)

            # Only extract detailed endgame features if this is an endgame
            if endgame_type['is_endgame']:
                # King position analysis
                king_features = analyze_king_position(board)
                feature_dict.update(king_features)

                # Pawn structure analysis for endgames
                pawn_features = analyze_pawn_structure_endgame(board)
                feature_dict.update(pawn_features)

                # Zugzwang potential
                zugzwang_features = detect_zugzwang_potential(board)
                feature_dict.update(zugzwang_features)

                # Tablebase distance estimation
                tablebase_features = estimate_tablebase_distance(board)
                feature_dict.update(tablebase_features)
            else:
                # Add default values for endgame features
                feature_dict.update({
                    'white_king_center_distance': 0,
                    'black_king_center_distance': 0,
                    'king_distance': 0,
                    'has_opposition': 0,
                    'white_king_edge_distance': 0,
                    'black_king_edge_distance': 0,
                    'white_passed_pawns': 0,
                    'black_passed_pawns': 0,
                    'white_most_advanced_pawn_rank': 0,
                    'black_most_advanced_pawn_rank': 0,
                    'white_connected_passed_pawns': 0,
                    'black_connected_passed_pawns': 0,
                    'zugzwang_potential': 0,
                    'in_tablebase': 0,
                    'tablebase_dtm_estimate': 0,
                    'theoretical_win': 0,
                    'theoretical_draw': 0
                })

            features.append(feature_dict)

        except Exception as e:
            logger.info(f"Error extracting endgame features for FEN {fen}: {e}")
            features.append({'idx': idx})

    # Create DataFrame from features
    endgame_features_df = pd.DataFrame(features).set_index('idx')

    # Fill NaN values
    endgame_features_df = endgame_features_df.fillna(0)

    logger.info(f"Created endgame feature set with {endgame_features_df.shape[1]} features")
    return endgame_features_df
