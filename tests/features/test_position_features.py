import pytest
import chess
import pandas as pd
import numpy as np
from chess_puzzle_rating.features.position_features import (
    identify_pawn_chains,
    identify_pawn_islands,
    detect_pins,
    detect_forks,
    detect_discovered_attacks,
    extract_fen_features
)

class TestPawnStructureAnalysis:
    """Tests for pawn structure analysis functions."""
    
    def test_identify_pawn_chains_empty(self):
        """Test identifying pawn chains with no pawns."""
        pawns = {}
        chains = identify_pawn_chains(pawns)
        assert chains == []
    
    def test_identify_pawn_chains_single(self):
        """Test identifying pawn chains with a single pawn."""
        pawns = {chess.E2}
        chains = identify_pawn_chains(pawns)
        assert chains == [[chess.E2]]
    
    def test_identify_pawn_chains_diagonal(self):
        """Test identifying pawn chains with diagonal pawns."""
        # Create a diagonal chain e2, d3, c4
        pawns = {chess.E2, chess.D3, chess.C4}
        chains = identify_pawn_chains(pawns)
        assert len(chains) == 1
        assert set(chains[0]) == {chess.E2, chess.D3, chess.C4}
    
    def test_identify_pawn_chains_multiple(self):
        """Test identifying multiple pawn chains."""
        # Create two separate chains: e2, d3, c4 and a2, b3
        pawns = {chess.E2, chess.D3, chess.C4, chess.A2, chess.B3}
        chains = identify_pawn_chains(pawns)
        assert len(chains) == 2
        # Check that each chain is correctly identified
        chains_as_sets = [set(chain) for chain in chains]
        assert {chess.E2, chess.D3, chess.C4} in chains_as_sets
        assert {chess.A2, chess.B3} in chains_as_sets
    
    def test_identify_pawn_islands_empty(self):
        """Test identifying pawn islands with no pawns."""
        pawns = {}
        islands = identify_pawn_islands(pawns)
        assert islands == []
    
    def test_identify_pawn_islands_single(self):
        """Test identifying pawn islands with a single pawn."""
        pawns = {chess.E2}
        islands = identify_pawn_islands(pawns)
        assert islands == [[chess.E2]]
    
    def test_identify_pawn_islands_adjacent(self):
        """Test identifying pawn islands with adjacent pawns."""
        # Create adjacent pawns on the same rank: e2, f2, g2
        pawns = {chess.E2, chess.F2, chess.G2}
        islands = identify_pawn_islands(pawns)
        assert len(islands) == 1
        assert set(islands[0]) == {chess.E2, chess.F2, chess.G2}
    
    def test_identify_pawn_islands_separate(self):
        """Test identifying separate pawn islands."""
        # Create two separate islands: e2, f2 and a2, b2
        pawns = {chess.E2, chess.F2, chess.A2, chess.B2}
        islands = identify_pawn_islands(pawns)
        assert len(islands) == 2
        # Check that each island is correctly identified
        islands_as_sets = [set(island) for island in islands]
        assert {chess.E2, chess.F2} in islands_as_sets
        assert {chess.A2, chess.B2} in islands_as_sets


class TestTacticalPatternDetection:
    """Tests for tactical pattern detection functions."""
    
    def test_detect_pins_starting_position(self):
        """Test pin detection in the starting position (should be none)."""
        board = chess.Board()
        white_pins = detect_pins(board, chess.WHITE)
        black_pins = detect_pins(board, chess.BLACK)
        assert len(white_pins) == 0
        assert len(black_pins) == 0
    
    def test_detect_pins_bishop_pin(self):
        """Test detecting a bishop pin."""
        # Set up a position with a bishop pinning a knight to the king
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B5/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
        pins = detect_pins(board, chess.WHITE)
        # The bishop on c4 is pinning the knight on c6 to the king on e8
        assert len(pins) == 1
        assert pins[0]['pinner'] == chess.C4
        assert pins[0]['pinned'] == chess.C6
        assert pins[0]['king'] == chess.E8
    
    def test_detect_forks_starting_position(self):
        """Test fork detection in the starting position (should be none)."""
        board = chess.Board()
        white_forks = detect_forks(board, chess.WHITE)
        black_forks = detect_forks(board, chess.BLACK)
        assert len(white_forks) == 0
        assert len(black_forks) == 0
    
    def test_detect_forks_knight_fork(self):
        """Test detecting a knight fork."""
        # Set up a position with a knight forking king and rook
        board = chess.Board("r1bqkbnr/pppp1ppp/8/4p3/3n4/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1")
        forks = detect_forks(board, chess.BLACK)
        # The knight on d4 is forking the king on e1 and the rook on a1
        assert len(forks) > 0
        # Find the fork that includes the knight on d4
        knight_forks = [f for f in forks if f['forking_piece'] == chess.D4]
        assert len(knight_forks) > 0
        # Check that the forked pieces include the king and rook
        forked_pieces = knight_forks[0]['forked_pieces']
        assert chess.E1 in forked_pieces  # King
        assert chess.A1 in forked_pieces  # Rook
    
    def test_detect_discovered_attacks_starting_position(self):
        """Test discovered attack detection in the starting position (should be none)."""
        board = chess.Board()
        white_discovered = detect_discovered_attacks(board, chess.WHITE)
        black_discovered = detect_discovered_attacks(board, chess.BLACK)
        assert len(white_discovered) == 0
        assert len(black_discovered) == 0
    
    def test_detect_discovered_attacks_bishop_discovered(self):
        """Test detecting a discovered attack by a bishop."""
        # Set up a position with a potential discovered attack
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B5/8/PPPP1PPP/RNBQK2R w KQkq - 0 1")
        # Move the knight to create a discovered attack by the bishop
        board.push_san("Nf3")
        discovered = detect_discovered_attacks(board, chess.WHITE)
        # The bishop on c1 now has a discovered attack through the knight on f3
        assert len(discovered) > 0
        # Find the discovered attack that includes the bishop on c1
        bishop_discovered = [d for d in discovered if d['attacking_piece'] == chess.C1]
        assert len(bishop_discovered) > 0


class TestFeatureExtraction:
    """Tests for the main feature extraction function."""
    
    def test_extract_fen_features(self, sample_puzzle_df):
        """Test extracting features from a DataFrame of FEN strings."""
        features_df = extract_fen_features(sample_puzzle_df)
        
        # Check that the function returns a DataFrame
        assert isinstance(features_df, pd.DataFrame)
        
        # Check that the DataFrame has the expected number of rows
        assert len(features_df) == len(sample_puzzle_df)
        
        # Check that the DataFrame has the expected columns
        expected_columns = [
            'material_balance', 'white_center_control', 'black_center_control',
            'white_king_safety', 'black_king_safety'
        ]
        for col in expected_columns:
            assert col in features_df.columns
        
        # Check that there are no NaN values in the DataFrame
        assert not features_df.isna().any().any()