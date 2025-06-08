"""
Module for extracting features from chess positions in FEN format.

This module provides functions to extract rich positional features from chess positions
represented in FEN (Forsyth-Edwards Notation) format. The features include:

1. Basic piece counts and material balance
2. Advanced pawn structure analysis (doubled/isolated pawns, chains, islands)
3. Center control metrics
4. Enhanced king safety evaluation with attack pattern detection
5. Development features for minor pieces
6. Detailed piece mobility metrics
7. Tactical pattern recognition (pins, forks, discovered attacks)

These features can be used for chess position evaluation, puzzle rating prediction,
and other chess-related machine learning tasks.
"""

import chess
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
import concurrent.futures
import os
from ..utils.config import get_config


def identify_pawn_chains(pawns):
    """
    Identify pawn chains in a position.
    A pawn chain is a diagonal connection of pawns supporting each other.

    Parameters
    ----------
    pawns : list
        List of squares containing pawns of one color

    Returns
    -------
    list
        List of pawn chains, where each chain is a list of connected pawn squares
    """
    if not pawns:
        return []

    # Create a graph representation of pawn connections
    pawn_graph = defaultdict(list)

    for p1 in pawns:
        file1 = chess.square_file(p1)
        rank1 = chess.square_rank(p1)

        for p2 in pawns:
            if p1 == p2:
                continue

            file2 = chess.square_file(p2)
            rank2 = chess.square_rank(p2)

            # Check if pawns are diagonally adjacent (supporting each other)
            if abs(file1 - file2) == 1 and abs(rank1 - rank2) == 1:
                pawn_graph[p1].append(p2)

    # Find connected components (chains)
    visited = set()
    chains = []

    for pawn in pawns:
        if pawn not in visited:
            chain = []
            queue = [pawn]

            while queue:
                current = queue.pop(0)
                if current not in visited:
                    visited.add(current)
                    chain.append(current)

                    for neighbor in pawn_graph[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)

            if len(chain) > 1:  # Only consider chains with at least 2 pawns
                chains.append(chain)

    return chains


def identify_pawn_islands(pawns):
    """
    Identify pawn islands in a position.
    A pawn island is a group of pawns on adjacent files with no pawns on neighboring files.

    Parameters
    ----------
    pawns : list
        List of squares containing pawns of one color

    Returns
    -------
    int
        Number of pawn islands
    """
    if not pawns:
        return 0

    # Get the files where pawns are located
    files = sorted([chess.square_file(p) for p in pawns])

    if not files:
        return 0

    # Count islands (groups of consecutive files)
    islands = 1
    for i in range(1, len(files)):
        if files[i] > files[i-1] + 1:  # Gap in files
            islands += 1

    return islands


def detect_pins(board, color):
    """
    Detect pinned pieces on the board.

    Parameters
    ----------
    board : chess.Board
        The chess board
    color : chess.Color
        The color of the pieces to check for pins

    Returns
    -------
    dict
        Dictionary with counts of pinned pieces by type
    """
    pins = {
        'pawn': 0,
        'knight': 0,
        'bishop': 0,
        'rook': 0,
        'queen': 0,
        'total': 0
    }

    king_square = board.king(color)
    if king_square is None:
        return pins

    # Piece type names for mapping
    piece_names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']

    # Check for pins along ranks, files, and diagonals
    for attacker_type in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
        # Get squares that the attacker type can attack from
        if attacker_type == chess.ROOK:
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, right, down, left
        elif attacker_type == chess.BISHOP:
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # Diagonals
        else:  # QUEEN
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)

        for d_file, d_rank in directions:
            potential_pin = None

            # Look along the direction from the king
            for distance in range(1, 8):
                file = king_file + d_file * distance
                rank = king_rank + d_rank * distance

                if not (0 <= file <= 7 and 0 <= rank <= 7):
                    break  # Off the board

                square = chess.square(file, rank)
                piece = board.piece_at(square)

                if piece is None:
                    continue  # Empty square

                if piece.color == color:
                    if potential_pin is None:
                        potential_pin = (square, piece)  # First piece of our color
                    else:
                        break  # Second piece of our color, no pin possible
                else:
                    # Enemy piece
                    if potential_pin is not None:
                        # Check if this enemy piece can attack along this direction
                        can_pin = False
                        if piece.piece_type == chess.QUEEN:
                            can_pin = True
                        elif piece.piece_type == chess.ROOK and attacker_type in [chess.ROOK, chess.QUEEN]:
                            can_pin = True
                        elif piece.piece_type == chess.BISHOP and attacker_type in [chess.BISHOP, chess.QUEEN]:
                            can_pin = True

                        if can_pin:
                            # We have a pin!
                            pinned_square, pinned_piece = potential_pin
                            if pinned_piece.piece_type < 6:  # Not the king
                                pins[piece_names[pinned_piece.piece_type - 1]] += 1
                                pins['total'] += 1
                    break  # Enemy piece, stop looking in this direction

    return pins


def detect_forks(board, color):
    """
    Detect forks in the position.
    A fork is when one piece attacks two or more valuable pieces simultaneously.

    Parameters
    ----------
    board : chess.Board
        The chess board
    color : chess.Color
        The color of the attacking pieces

    Returns
    -------
    dict
        Dictionary with counts of different types of forks
    """
    forks = {
        'knight_forks': 0,
        'pawn_forks': 0,
        'bishop_forks': 0,
        'rook_forks': 0,
        'queen_forks': 0,
        'total_forks': 0
    }

    enemy_color = not color

    # Get all pieces of the attacking color
    for square, piece in board.piece_map().items():
        if piece.color != color:
            continue

        # Create a temporary board with just this piece to find its legal moves
        temp_board = chess.Board(None)
        temp_board.set_piece_at(square, piece)
        temp_board.turn = color

        # Get all squares this piece attacks
        attacked_squares = set()
        for move in temp_board.legal_moves:
            attacked_squares.add(move.to_square)

        # Count valuable enemy pieces being attacked
        valuable_targets = []
        for attacked in attacked_squares:
            target = board.piece_at(attacked)
            if target and target.color == enemy_color and target.piece_type != chess.PAWN:
                valuable_targets.append((attacked, target))

        # If attacking two or more valuable pieces, it's a fork
        if len(valuable_targets) >= 2:
            piece_type = piece.piece_type
            if piece_type == chess.KNIGHT:
                forks['knight_forks'] += 1
            elif piece_type == chess.PAWN:
                forks['pawn_forks'] += 1
            elif piece_type == chess.BISHOP:
                forks['bishop_forks'] += 1
            elif piece_type == chess.ROOK:
                forks['rook_forks'] += 1
            elif piece_type == chess.QUEEN:
                forks['queen_forks'] += 1

            forks['total_forks'] += 1

    return forks


def detect_discovered_attacks(board, color):
    """
    Detect potential discovered attacks in the position.

    Parameters
    ----------
    board : chess.Board
        The chess board
    color : chess.Color
        The color of the attacking pieces

    Returns
    -------
    int
        Number of potential discovered attacks
    """
    discovered_attacks = 0
    enemy_color = not color
    enemy_king_square = board.king(enemy_color)

    if enemy_king_square is None:
        return 0

    # Check for potential discovered attacks along ranks, files, and diagonals
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),  # Rook directions
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # Bishop directions
    ]

    king_file = chess.square_file(enemy_king_square)
    king_rank = chess.square_rank(enemy_king_square)

    for d_file, d_rank in directions:
        blocker = None
        attacker = None

        # Look along the direction from the enemy king
        for distance in range(1, 8):
            file = king_file + d_file * distance
            rank = king_rank + d_rank * distance

            if not (0 <= file <= 7 and 0 <= rank <= 7):
                break  # Off the board

            square = chess.square(file, rank)
            piece = board.piece_at(square)

            if piece is None:
                continue  # Empty square

            if piece.color == color:
                if blocker is None:
                    blocker = (square, piece)  # First piece of our color
                elif attacker is None:
                    # Check if this piece can attack along this direction
                    can_attack = False
                    if piece.piece_type == chess.QUEEN:
                        can_attack = True
                    elif piece.piece_type == chess.ROOK and d_file * d_rank == 0:  # Rook direction
                        can_attack = True
                    elif piece.piece_type == chess.BISHOP and d_file * d_rank != 0:  # Bishop direction
                        can_attack = True

                    if can_attack:
                        attacker = (square, piece)
                        discovered_attacks += 1
                    break
            else:
                break  # Enemy piece, stop looking in this direction

    return discovered_attacks


def process_single_fen(idx_fen):
    """
    Process a single FEN string and extract features.

    Parameters
    ----------
    idx_fen : tuple
        Tuple containing (idx, fen)

    Returns
    -------
    dict
        Dictionary with extracted features
    """
    idx, fen = idx_fen
    feature_dict = {'idx': idx}

    try:
        board = chess.Board(fen)
        # The rest of the feature extraction code will be executed in the main function
        return idx, board
    except Exception as e:
        print(f"Error creating board for FEN {fen}: {e}")
        return idx, None


def extract_features_from_board(idx_board_fen):
    """
    Extract features from a single board object.

    Parameters
    ----------
    idx_board_fen : tuple
        Tuple containing (idx, board, fen)

    Returns
    -------
    dict
        Dictionary with extracted features
    """
    idx, board, fen = idx_board_fen
    feature_dict = {'idx': idx}

    try:
        if board is None:
            # Skip this FEN if we couldn't create a board
            return feature_dict

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

        # 2. Material balance - using values from configuration
        config_values = get_config()['feature_engineering']['material_values']
        material_values = {
            chess.PAWN: config_values['pawn'],
            chess.KNIGHT: config_values['knight'],
            chess.BISHOP: config_values['bishop'],
            chess.ROOK: config_values['rook'],
            chess.QUEEN: config_values['queen'],
            chess.KING: config_values['king']
        }
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

        # Advanced pawn structure analysis
        # Pawn chains
        white_pawn_chains = identify_pawn_chains(white_pawns)
        black_pawn_chains = identify_pawn_chains(black_pawns)

        feature_dict['white_pawn_chains'] = len(white_pawn_chains)
        feature_dict['black_pawn_chains'] = len(black_pawn_chains)

        # Longest chain
        feature_dict['white_longest_chain'] = max([len(chain) for chain in white_pawn_chains]) if white_pawn_chains else 0
        feature_dict['black_longest_chain'] = max([len(chain) for chain in black_pawn_chains]) if black_pawn_chains else 0

        # Total pawns in chains
        feature_dict['white_pawns_in_chains'] = sum(len(chain) for chain in white_pawn_chains)
        feature_dict['black_pawns_in_chains'] = sum(len(chain) for chain in black_pawn_chains)

        # Pawn islands
        feature_dict['white_pawn_islands'] = identify_pawn_islands(white_pawns)
        feature_dict['black_pawn_islands'] = identify_pawn_islands(black_pawns)

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

        # 5. Enhanced king safety with attack pattern detection
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)

        # Basic king position features
        if white_king_square is not None:
            white_king_file = chess.square_file(white_king_square)
            white_king_rank = chess.square_rank(white_king_square)
            feature_dict['white_king_castled_kingside'] = 1 if white_king_file >= 6 else 0  # g or h file
            feature_dict['white_king_castled_queenside'] = 1 if white_king_file <= 2 else 0  # a, b or c file
            feature_dict['white_king_center'] = 1 if 3 <= white_king_file <= 4 else 0  # d or e file

            # King zone (squares around the king)
            white_king_zone = []
            for file_offset in [-1, 0, 1]:
                for rank_offset in [-1, 0, 1]:
                    if file_offset == 0 and rank_offset == 0:
                        continue  # Skip the king's square itself

                    zone_file = white_king_file + file_offset
                    zone_rank = white_king_rank + rank_offset

                    if 0 <= zone_file <= 7 and 0 <= zone_rank <= 7:
                        zone_square = chess.square(zone_file, zone_rank)
                        white_king_zone.append(zone_square)

            # Count attackers to king zone
            white_king_attackers = 0
            for zone_square in white_king_zone + [white_king_square]:
                attackers = board.attackers(chess.BLACK, zone_square)
                white_king_attackers += len(attackers)

            feature_dict['white_king_attackers'] = white_king_attackers

            # Pawn shield analysis
            white_pawn_shield = 0
            for file_offset in [-1, 0, 1]:
                shield_file = white_king_file + file_offset
                if 0 <= shield_file <= 7:
                    for rank_offset in [1, 2]:  # 1-2 squares in front of king
                        shield_rank = white_king_rank + rank_offset if white_king_rank <= 5 else white_king_rank - rank_offset
                        if 0 <= shield_rank <= 7:
                            shield_square = chess.square(shield_file, shield_rank)
                            piece = board.piece_at(shield_square)
                            if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                                white_pawn_shield += 1

            feature_dict['white_pawn_shield'] = white_pawn_shield

            # Open files near king
            white_king_open_files = 0
            for file_offset in [-1, 0, 1]:
                check_file = white_king_file + file_offset
                if 0 <= check_file <= 7:
                    file_has_pawn = False
                    for rank in range(8):
                        square = chess.square(check_file, rank)
                        piece = board.piece_at(square)
                        if piece and piece.piece_type == chess.PAWN:
                            file_has_pawn = True
                            break
                    if not file_has_pawn:
                        white_king_open_files += 1

            feature_dict['white_king_open_files'] = white_king_open_files
        else:
            feature_dict['white_king_castled_kingside'] = 0
            feature_dict['white_king_castled_queenside'] = 0
            feature_dict['white_king_center'] = 0
            feature_dict['white_king_attackers'] = 0
            feature_dict['white_pawn_shield'] = 0
            feature_dict['white_king_open_files'] = 0

        # Repeat for black king
        if black_king_square is not None:
            black_king_file = chess.square_file(black_king_square)
            black_king_rank = chess.square_rank(black_king_square)
            feature_dict['black_king_castled_kingside'] = 1 if black_king_file >= 6 else 0
            feature_dict['black_king_castled_queenside'] = 1 if black_king_file <= 2 else 0
            feature_dict['black_king_center'] = 1 if 3 <= black_king_file <= 4 else 0

            # King zone (squares around the king)
            black_king_zone = []
            for file_offset in [-1, 0, 1]:
                for rank_offset in [-1, 0, 1]:
                    if file_offset == 0 and rank_offset == 0:
                        continue  # Skip the king's square itself

                    zone_file = black_king_file + file_offset
                    zone_rank = black_king_rank + rank_offset

                    if 0 <= zone_file <= 7 and 0 <= zone_rank <= 7:
                        zone_square = chess.square(zone_file, zone_rank)
                        black_king_zone.append(zone_square)

            # Count attackers to king zone
            black_king_attackers = 0
            for zone_square in black_king_zone + [black_king_square]:
                attackers = board.attackers(chess.WHITE, zone_square)
                black_king_attackers += len(attackers)

            feature_dict['black_king_attackers'] = black_king_attackers

            # Pawn shield analysis
            black_pawn_shield = 0
            for file_offset in [-1, 0, 1]:
                shield_file = black_king_file + file_offset
                if 0 <= shield_file <= 7:
                    for rank_offset in [1, 2]:  # 1-2 squares in front of king
                        shield_rank = black_king_rank - rank_offset if black_king_rank >= 2 else black_king_rank + rank_offset
                        if 0 <= shield_rank <= 7:
                            shield_square = chess.square(shield_file, shield_rank)
                            piece = board.piece_at(shield_square)
                            if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                                black_pawn_shield += 1

            feature_dict['black_pawn_shield'] = black_pawn_shield

            # Open files near king
            black_king_open_files = 0
            for file_offset in [-1, 0, 1]:
                check_file = black_king_file + file_offset
                if 0 <= check_file <= 7:
                    file_has_pawn = False
                    for rank in range(8):
                        square = chess.square(check_file, rank)
                        piece = board.piece_at(square)
                        if piece and piece.piece_type == chess.PAWN:
                            file_has_pawn = True
                            break
                    if not file_has_pawn:
                        black_king_open_files += 1

            feature_dict['black_king_open_files'] = black_king_open_files
        else:
            feature_dict['black_king_castled_kingside'] = 0
            feature_dict['black_king_castled_queenside'] = 0
            feature_dict['black_king_center'] = 0
            feature_dict['black_king_attackers'] = 0
            feature_dict['black_pawn_shield'] = 0
            feature_dict['black_king_open_files'] = 0

        # King safety score (higher is safer) - using weights from configuration
        king_safety_weights = get_config()['feature_engineering']['king_safety_weights']
        pawn_shield_weight = king_safety_weights['pawn_shield']
        king_attackers_weight = king_safety_weights['king_attackers']
        king_open_files_weight = king_safety_weights['king_open_files']
        castling_bonus = king_safety_weights['castling_bonus']

        if white_king_square is not None:
            feature_dict['white_king_safety'] = (
                (feature_dict['white_pawn_shield'] * pawn_shield_weight) + 
                (feature_dict['white_king_attackers'] * king_attackers_weight) + 
                (feature_dict['white_king_open_files'] * king_open_files_weight) +
                (castling_bonus if feature_dict['white_king_castled_kingside'] or feature_dict['white_king_castled_queenside'] else 0)
            )
        else:
            feature_dict['white_king_safety'] = 0

        if black_king_square is not None:
            feature_dict['black_king_safety'] = (
                (feature_dict['black_pawn_shield'] * pawn_shield_weight) + 
                (feature_dict['black_king_attackers'] * king_attackers_weight) + 
                (feature_dict['black_king_open_files'] * king_open_files_weight) +
                (castling_bonus if feature_dict['black_king_castled_kingside'] or feature_dict['black_king_castled_queenside'] else 0)
            )
        else:
            feature_dict['black_king_safety'] = 0

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

        # 7. Enhanced mobility features
        # Initialize mobility counters for each piece type
        white_mobility = {
            'pawn': 0,
            'knight': 0,
            'bishop': 0,
            'rook': 0,
            'queen': 0,
            'king': 0,
            'total': 0
        }
        black_mobility = {
            'pawn': 0,
            'knight': 0,
            'bishop': 0,
            'rook': 0,
            'queen': 0,
            'king': 0,
            'total': 0
        }

        # Piece type names for mapping
        piece_names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']

        # Count legal moves for each piece with context
        for square, piece in pieces.items():
            # Create a new board with just this piece to count its moves
            temp_board = chess.Board(None)
            temp_board.set_piece_at(square, piece)

            # Set the turn to match the piece color
            temp_board.turn = piece.color

            # Count legal moves
            moves = list(temp_board.legal_moves)
            piece_name = piece_names[piece.piece_type - 1]

            if piece.color == chess.WHITE:
                white_mobility[piece_name] += len(moves)
                white_mobility['total'] += len(moves)
            else:
                black_mobility[piece_name] += len(moves)
                black_mobility['total'] += len(moves)

        # Add individual piece type mobility to features
        for piece_name in piece_names:
            feature_dict[f'white_{piece_name}_mobility'] = white_mobility[piece_name]
            feature_dict[f'black_{piece_name}_mobility'] = black_mobility[piece_name]

        # Add total mobility
        feature_dict['white_mobility'] = white_mobility['total']
        feature_dict['black_mobility'] = black_mobility['total']
        feature_dict['mobility_balance'] = white_mobility['total'] - black_mobility['total']

        # Calculate mobility ratios for major pieces
        feature_dict['white_minor_piece_mobility_ratio'] = (
            (white_mobility['knight'] + white_mobility['bishop']) / 
            max(1, white_mobility['total'] - white_mobility['pawn'] - white_mobility['king'])
        )
        feature_dict['black_minor_piece_mobility_ratio'] = (
            (black_mobility['knight'] + black_mobility['bishop']) / 
            max(1, black_mobility['total'] - black_mobility['pawn'] - black_mobility['king'])
        )

        # Calculate mobility advantage by piece type
        for piece_name in piece_names:
            feature_dict[f'{piece_name}_mobility_advantage'] = (
                white_mobility[piece_name] - black_mobility[piece_name]
            )

        # 8. Tactical pattern recognition
        # Pins
        white_pins = detect_pins(board, chess.WHITE)
        black_pins = detect_pins(board, chess.BLACK)

        # Add pin features
        for piece_name in ['pawn', 'knight', 'bishop', 'rook', 'queen']:
            feature_dict[f'white_pinned_{piece_name}s'] = white_pins[piece_name]
            feature_dict[f'black_pinned_{piece_name}s'] = black_pins[piece_name]

        feature_dict['white_total_pinned_pieces'] = white_pins['total']
        feature_dict['black_total_pinned_pieces'] = black_pins['total']

        # Forks
        white_forks = detect_forks(board, chess.WHITE)
        black_forks = detect_forks(board, chess.BLACK)

        # Add fork features
        for fork_type in ['knight_forks', 'pawn_forks', 'bishop_forks', 'rook_forks', 'queen_forks', 'total_forks']:
            feature_dict[f'white_{fork_type}'] = white_forks[fork_type]
            feature_dict[f'black_{fork_type}'] = black_forks[fork_type]

        # Discovered attacks
        feature_dict['white_discovered_attacks'] = detect_discovered_attacks(board, chess.WHITE)
        feature_dict['black_discovered_attacks'] = detect_discovered_attacks(board, chess.BLACK)

        # Tactical advantage metrics
        feature_dict['tactical_advantage_pins'] = white_pins['total'] - black_pins['total']
        feature_dict['tactical_advantage_forks'] = white_forks['total_forks'] - black_forks['total_forks']
        feature_dict['tactical_advantage_discovered'] = feature_dict['white_discovered_attacks'] - feature_dict['black_discovered_attacks']

        # Overall tactical advantage - using weights from configuration
        tactical_weights = get_config()['feature_engineering']['tactical_advantage_weights']
        feature_dict['overall_tactical_advantage'] = (
            feature_dict['tactical_advantage_pins'] * tactical_weights['pins'] + 
            feature_dict['tactical_advantage_forks'] * tactical_weights['forks'] + 
            feature_dict['tactical_advantage_discovered'] * tactical_weights['discovered_attacks']
        )

        return feature_dict

    except Exception as e:
        print(f"Error extracting features for FEN {fen}: {e}")
        return feature_dict


def extract_fen_features(df, fen_column='FEN'):
    """
    Extract rich positional features from FEN strings.
    Uses parallel processing for improved performance.

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

    # Get configuration for internal parallelization
    config = get_config()
    performance_config = config.get('performance', {})
    parallel_config = performance_config.get('parallel', {})

    # Determine the number of threads to use for internal parallelization
    # Use all available CPU cores for maximum parallelization
    n_threads = os.cpu_count() or 4  # Default to 4 if os.cpu_count() returns None

    # Prepare the list of (idx, fen) tuples
    idx_fen_tuples = list(df[fen_column].items())

    # Process FEN strings in parallel using ThreadPoolExecutor to create board objects
    print(f"Using {n_threads} threads for parallel FEN processing")
    boards = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Process results as they complete
        for idx, board in tqdm(executor.map(process_single_fen, idx_fen_tuples), total=len(idx_fen_tuples), desc="Creating board objects"):
            boards[idx] = board

    # Prepare input for parallel feature extraction
    idx_board_fen_tuples = [(idx, boards[idx], fen) for idx, fen in idx_fen_tuples]

    # Process feature extraction in parallel
    features = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Process results as they complete
        for feature_dict in tqdm(executor.map(extract_features_from_board, idx_board_fen_tuples), 
                                total=len(idx_board_fen_tuples), desc="Extracting features"):
            features.append(feature_dict)

    # Create DataFrame from features
    fen_features_df = pd.DataFrame(features).set_index('idx')

    # Fill NaN values
    fen_features_df = fen_features_df.fillna(0)

    print(f"Extracted {fen_features_df.shape[1]} position features from FEN")
    return fen_features_df
