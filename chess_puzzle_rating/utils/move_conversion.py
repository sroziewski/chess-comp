"""
Utility functions for converting between different chess move formats.
"""

import chess
import concurrent.futures
from typing import List, Tuple, Union, Optional

def uci_to_pgn(uci_moves, fen=None):
    """
    Convert a sequence of UCI moves to PGN format.

    Parameters
    ----------
    uci_moves : str or list
        A string of space-separated UCI moves or a list of UCI move strings
    fen : str, optional
        The starting FEN position. If None, the standard starting position is used.

    Returns
    -------
    str
        A space-separated string of moves in PGN format
    """
    # Initialize the board with the given FEN or the starting position
    board = chess.Board(fen) if fen else chess.Board()

    # If uci_moves is a string, split it into a list
    if isinstance(uci_moves, str):
        uci_moves = uci_moves.split()

    # Convert each UCI move to PGN
    pgn_moves = []
    for uci_move in uci_moves:
        try:
            # Parse the UCI move
            move = chess.Move.from_uci(uci_move)

            # Check if the move is legal
            if move in board.legal_moves:
                # Convert to SAN (Standard Algebraic Notation, used in PGN)
                pgn_move = board.san(move)
                pgn_moves.append(pgn_move)

                # Make the move on the board
                board.push(move)
            else:
                # If the move is illegal, stop processing
                break
        except ValueError:
            # If the move can't be parsed, stop processing
            break

    # Return the PGN moves as a space-separated string
    return ' '.join(pgn_moves)

def uci_to_pgn_parallel(moves_list: List[Union[str, List[str]]], 
                        fens_list: Optional[List[str]] = None, 
                        max_workers: int = None) -> List[str]:
    """
    Convert multiple sequences of UCI moves to PGN format in parallel.

    Parameters
    ----------
    moves_list : List[Union[str, List[str]]]
        A list of UCI move sequences, where each sequence can be a string of 
        space-separated UCI moves or a list of UCI move strings
    fens_list : List[str], optional
        A list of starting FEN positions corresponding to each move sequence.
        If None, the standard starting position is used for all sequences.
    max_workers : int, optional
        Maximum number of worker threads to use. If None, it will default to 
        the number of processors on the machine, multiplied by 5.

    Returns
    -------
    List[str]
        A list of space-separated strings of moves in PGN format
    """
    # Prepare the arguments for parallel processing
    if fens_list is None:
        fens_list = [None] * len(moves_list)

    # Make sure the lists have the same length
    if len(moves_list) != len(fens_list):
        raise ValueError("moves_list and fens_list must have the same length")

    # Define a worker function for processing a single item
    def process_item(args: Tuple[Union[str, List[str]], Optional[str]]) -> str:
        moves, fen = args
        return uci_to_pgn(moves, fen)

    # Process items in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_item, zip(moves_list, fens_list)))

    return results
