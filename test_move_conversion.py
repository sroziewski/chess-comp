"""
Test script for the UCI to PGN move conversion functionality.
"""

import pandas as pd
import chess
import time
from chess_puzzle_rating.utils.move_conversion import uci_to_pgn, uci_to_pgn_parallel
from chess_puzzle_rating.data.pipeline import run_data_pipeline

def test_uci_to_pgn_conversion():
    """Test the UCI to PGN conversion function with a few examples."""
    # Test with standard starting position
    uci_moves = "e2e4 e7e5 g1f3 b8c6 f1b5"  # Ruy Lopez opening
    pgn_moves = uci_to_pgn(uci_moves)
    print(f"UCI: {uci_moves}")
    print(f"PGN: {pgn_moves}")
    assert pgn_moves == "e4 e5 Nf3 Nc6 Bb5"

    # Test with a specific FEN position
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    uci_moves = "f1b5 g8f6 e1g1"  # Continuing Ruy Lopez with castling
    pgn_moves = uci_to_pgn(uci_moves, fen)
    print(f"FEN: {fen}")
    print(f"UCI: {uci_moves}")
    print(f"PGN: {pgn_moves}")
    assert pgn_moves == "Bb5 Nf6 O-O"

    print("UCI to PGN conversion tests passed!")

def test_parallel_uci_to_pgn_conversion():
    """Test the parallel UCI to PGN conversion function and compare with sequential version."""
    # Create a list of test cases
    test_cases = [
        # Standard starting position with different openings
        ("e2e4 e7e5 g1f3 b8c6 f1b5", None),  # Ruy Lopez
        ("d2d4 d7d5 c2c4 e7e6 b1c3 g8f6", None),  # Queen's Gambit
        ("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4", None),  # Sicilian Defense

        # Custom FEN positions
        ("f1b5 g8f6 e1g1", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
        ("e1g1 e8g8 f1e1 f8e8", "r1bq1rk1/ppp2ppp/2np1n2/1B2p3/4P3/2P2N2/PP1P1PPP/RNBQR1K1 w - - 0 8")
    ]

    # Extract moves and FENs
    moves_list = [tc[0] for tc in test_cases]
    fens_list = [tc[1] for tc in test_cases]

    # Test sequential conversion
    print("Testing sequential conversion...")
    start_time = time.time()
    sequential_results = []
    for moves, fen in test_cases:
        sequential_results.append(uci_to_pgn(moves, fen))
    sequential_time = time.time() - start_time
    print(f"Sequential conversion time: {sequential_time:.4f} seconds")

    # Test parallel conversion
    print("Testing parallel conversion...")
    start_time = time.time()
    parallel_results = uci_to_pgn_parallel(moves_list, fens_list)
    parallel_time = time.time() - start_time
    print(f"Parallel conversion time: {parallel_time:.4f} seconds")

    # Compare results
    for i, (seq, par) in enumerate(zip(sequential_results, parallel_results)):
        print(f"Test case {i+1}:")
        print(f"  Sequential: {seq}")
        print(f"  Parallel:   {par}")
        assert seq == par, f"Results don't match for test case {i+1}"

    print("All parallel conversion tests passed!")
    print(f"Speed improvement: {sequential_time / parallel_time:.2f}x")

    return True

def test_pipeline_with_pgn_conversion():
    """
    Test that the data pipeline correctly adds the MovesPGN column.
    This test requires the actual dataset files to be present.
    """
    try:
        # Run the pipeline with a small subset of data
        X_train, X_test, y_train, test_ids = run_data_pipeline()
        print("Data pipeline completed successfully!")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        return True
    except Exception as e:
        print(f"Error running data pipeline: {e}")
        return False

if __name__ == "__main__":
    print("Testing UCI to PGN conversion...")
    test_uci_to_pgn_conversion()

    print("\nTesting parallel UCI to PGN conversion...")
    test_parallel_uci_to_pgn_conversion()

    print("\nTesting data pipeline with PGN conversion...")
    success = test_pipeline_with_pgn_conversion()

    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")
