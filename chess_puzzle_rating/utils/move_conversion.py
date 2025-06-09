"""
Utility functions for converting between different chess move formats.
"""

import chess
import concurrent.futures
import multiprocessing
import math
from typing import List, Tuple, Union, Optional, Iterator
from ..utils.progress import track_progress, get_logger

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

def chunk_list(lst: List, chunk_size: int) -> Iterator[List]:
    """
    Split a list into chunks of specified size.

    Parameters
    ----------
    lst : List
        The list to split
    chunk_size : int
        The size of each chunk

    Returns
    -------
    Iterator[List]
        An iterator of list chunks
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def uci_to_pgn_parallel(moves_list: List[Union[str, List[str]]], 
                        fens_list: Optional[List[str]] = None, 
                        max_workers: int = None,
                        chunk_size: int = None,
                        show_progress: bool = True) -> List[str]:
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
        Maximum number of worker processes to use. If None, it will default to 
        the number of CPU cores.
    chunk_size : int, optional
        Size of chunks to process in parallel. If None, it will be calculated based
        on the number of items and workers.
    show_progress : bool, optional
        Whether to show a progress bar, by default True

    Returns
    -------
    List[str]
        A list of space-separated strings of moves in PGN format
    """
    logger = get_logger()
    logger.info(f"Converting {len(moves_list)} UCI move sequences to PGN format in parallel")

    # Prepare the arguments for parallel processing
    if fens_list is None:
        fens_list = [None] * len(moves_list)

    # Make sure the lists have the same length
    if len(moves_list) != len(fens_list):
        raise ValueError("moves_list and fens_list must have the same length")

    # Determine the number of worker processes
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    logger.info(f"Using {max_workers} worker processes")

    # Calculate chunk size if not provided
    if chunk_size is None:
        # Aim for at least 10 chunks per worker for better load balancing
        chunk_size = max(1, math.ceil(len(moves_list) / (max_workers * 10)))

    logger.info(f"Using chunk size of {chunk_size} items")

    # Create chunks of data
    moves_chunks = list(chunk_list(moves_list, chunk_size))
    fens_chunks = list(chunk_list(fens_list, chunk_size))

    # Define a worker function for processing a chunk
    def process_chunk(chunk_data: Tuple[List[Union[str, List[str]]], List[Optional[str]]]) -> List[str]:
        chunk_moves, chunk_fens = chunk_data
        results = []

        # Process each item in the chunk
        for moves, fen in zip(chunk_moves, chunk_fens):
            results.append(uci_to_pgn(moves, fen))

        return results

    # Process chunks in parallel
    all_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for processing
        futures = [executor.submit(process_chunk, (moves_chunk, fens_chunk)) 
                  for moves_chunk, fens_chunk in zip(moves_chunks, fens_chunks)]

        # Process results as they complete
        if show_progress:
            for future in track_progress(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                description="Converting UCI to PGN",
                logger=logger
            ):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
        else:
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")

    # Make sure results are in the same order as the input
    if len(all_results) != len(moves_list):
        logger.warning(f"Expected {len(moves_list)} results, but got {len(all_results)}")

    logger.info(f"Completed conversion of {len(all_results)} UCI move sequences to PGN format")

    return all_results
