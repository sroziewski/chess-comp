# Using Polyglot Opening Books to Predict Missing Tags
These opening book files contain a wealth of information about chess openings, and you can use them to help predict missing opening tags in your dataset. Here's how to do it:
## 1. Reading Polyglot Opening Books
First, you'll need to read these .bin files. The `python-chess` library has built-in support for Polyglot books:

import chess
import chess.polyglot
import pandas as pd
from tqdm import tqdm


def load_opening_book(book_path):
    """Load a Polyglot opening book and create a lookup dictionary"""
    print(f"Loading opening book: {book_path}")

    # Dictionary to store positions and their names
    position_to_opening = {}

    try:
        # Open the book
        with chess.polyglot.open_reader(book_path) as reader:
            # Get all entries
            entries = list(reader.find_all(chess.Board()))

            print(f"Found {len(entries)} entries in opening book")

            # Process each entry
            for entry in tqdm(entries, desc="Processing book entries"):
                # Get the position (after moves are played)
                board = chess.Board()
                for move in entry.line:
                    board.push(move)

                # Store the FEN (just the position part)
                fen = board.fen().split(' ')[0]

                # Store entry information
                if fen not in position_to_opening:
                    position_to_opening[fen] = {
                        'weight': entry.weight,
                        'learn': entry.learn,
                        'line': ' '.join([board.san(move) for move in entry.line])
                    }
                elif entry.weight > position_to_opening[fen]['weight']:
                    # Update if this entry has a higher weight
                    position_to_opening[fen] = {
                        'weight': entry.weight,
                        'learn': entry.learn,
                        'line': ' '.join([board.san(move) for move in entry.line])
                    }

        return position_to_opening

    except Exception as e:
        print(f"Error loading book: {e}")
        return {}


## 2. Mapping Book Entries to Opening Names
The Polyglot books don't contain opening names directly, but we can create a mapping using an ECO database:


def create_eco_mapping():
    """Create a mapping from move sequences to opening names using ECO codes"""

    # This is a simplified mapping - ideally, you'd use a complete ECO database
    eco_mapping = {
        "e4": {"name": "King's Pawn", "eco": "B"},
        "e4 e5": {"name": "Open Game", "eco": "C"},
        "e4 e5 Nf3": {"name": "King's Knight", "eco": "C"},
        "e4 e5 Nf3 Nc6": {"name": "Two Knights", "eco": "C"},
        "e4 e5 Nf3 Nc6 Bc4": {"name": "Italian Game", "eco": "C"},
        "e4 e5 Nf3 Nc6 Bb5": {"name": "Ruy Lopez", "eco": "C"},
        "e4 c5": {"name": "Sicilian Defense", "eco": "B"},
        "e4 c6": {"name": "Caro-Kann Defense", "eco": "B"},
        "e4 e6": {"name": "French Defense", "eco": "C"},
        "e4 d5": {"name": "Scandinavian Defense", "eco": "B"},
        "d4": {"name": "Queen's Pawn", "eco": "D"},
        "d4 d5": {"name": "Closed Game", "eco": "D"},
        "d4 d5 c4": {"name": "Queen's Gambit", "eco": "D"},
        "d4 d5 c4 e6": {"name": "Queen's Gambit Declined", "eco": "D"},
        "d4 d5 c4 c6": {"name": "Slav Defense", "eco": "D"},
        "d4 Nf6": {"name": "Indian Defense", "eco": "E"},
        "d4 Nf6 c4": {"name": "Indian Defense", "eco": "E"},
        "d4 Nf6 c4 g6": {"name": "King's Indian Defense", "eco": "E"},
        "d4 Nf6 c4 e6": {"name": "Queen's Indian Defense", "eco": "E"},
        "c4": {"name": "English Opening", "eco": "A"},
        "Nf3": {"name": "Reti Opening", "eco": "A"},
        # Add more mappings as needed
    }

    return eco_mapping


## 3. Combining Multiple Opening Books
Since you have multiple books, you can combine them for better coverage:

def combine_opening_books(book_paths):
    """Combine multiple opening books into one comprehensive dictionary"""

    combined_positions = {}

    for book_path in book_paths:
        book_positions = load_opening_book(book_path)

        # Merge into combined dictionary
        for fen, data in book_positions.items():
            if fen not in combined_positions or data['weight'] > combined_positions[fen]['weight']:
                combined_positions[fen] = data

    print(f"Combined book contains {len(combined_positions)} unique positions")
    return combined_positions

## 4. Using Books to Predict Opening Tags
Now, let's use these books to predict missing opening tags in your dataset:


def predict_tags_using_books(df, book_paths, fen_column='FEN', moves_column='Moves'):
    """
    Use opening books to predict missing opening tags.

    Args:
        df: DataFrame with chess puzzles
        book_paths: List of paths to polyglot opening book files
        fen_column: Column containing FEN strings
        moves_column: Column containing moves

    Returns:
        DataFrame with predicted opening tags and confidence scores
    """
    print("Predicting opening tags using opening books...")

    # Load all books
    combined_book = combine_opening_books(book_paths)

    # Create ECO mapping
    eco_mapping = create_eco_mapping()

    # Initialize results
    results = pd.DataFrame(index=df.index)
    results['predicted_opening'] = ""
    results['book_confidence'] = 0.0
    results['predicted_eco'] = ""

    # Identify puzzles without tags
    has_tags = ~df['OpeningTags'].isna() & (df['OpeningTags'] != '')
    puzzles_without_tags = df[~has_tags].index

    # Function to preprocess FEN string
    def preprocess_fen(fen):
        if pd.isna(fen) or not fen:
            return ""
        # Just get the piece placement part
        return fen.split(' ')[0]

    # Function to match position with book
    def match_with_book(fen, moves):
        # Direct FEN match
        fen_processed = preprocess_fen(fen)
        if fen_processed in combined_book:
            return combined_book[fen_processed], 1.0  # Perfect confidence

        # Try to reconstruct position from moves
        if not pd.isna(moves) and moves:
            try:
                board = chess.Board()

                # Parse moves
                move_list = moves.split()

                # Track all positions we reach
                positions = []

                for move_idx, move_str in enumerate(move_list):
                    try:
                        # Try to parse as SAN (Standard Algebraic Notation)
                        move = board.parse_san(move_str)
                        board.push(move)

                        # Store this position
                        pos_fen = board.fen().split(' ')[0]
                        positions.append((pos_fen, move_idx))
                    except:
                        # If parsing fails, stop
                        break

                # Check if any position is in our book
                matching_positions = []
                for pos_fen, move_idx in positions:
                    if pos_fen in combined_book:
                        # Calculate confidence based on move number
                        # Earlier positions (higher move_idx) get lower confidence
                        confidence = max(0.4, 1.0 - (move_idx * 0.05))
                        matching_positions.append((combined_book[pos_fen], confidence))

                if matching_positions:
                    # Return the match with highest confidence
                    return max(matching_positions, key=lambda x: x[1])
            except:
                pass

        # No match found
        return None, 0.0

    # Function to infer opening name from move sequence
    def infer_opening_name(move_sequence):
        # Look for the longest matching prefix in our ECO mapping
        best_match = None
        best_match_length = 0

        for eco_seq, info in eco_mapping.items():
            if move_sequence.startswith(eco_seq) and len(eco_seq) > best_match_length:
                best_match = info
                best_match_length = len(eco_seq)

        if best_match:
            return best_match

        # No match found
        return {"name": "Unknown", "eco": ""}

    # Process each puzzle without tags
    for idx in tqdm(puzzles_without_tags, desc="Matching positions with books"):
        fen = df.loc[idx, fen_column]
        moves = df.loc[idx, moves_column] if moves_column in df.columns else None

        # Match with book
        book_match, confidence = match_with_book(fen, moves)

        if book_match:
            # Get the move sequence from the book
            move_sequence = book_match['line']

            # Infer opening name
            opening_info = infer_opening_name(move_sequence)

            # Store results
            results.loc[idx, 'predicted_opening'] = opening_info['name']
            results.loc[idx, 'book_confidence'] = confidence
            results.loc[idx, 'predicted_eco'] = opening_info['eco']

    # Count matches
    matches_count = (results['book_confidence'] > 0).sum()
    high_conf_count = (results['book_confidence'] >= 0.7).sum()

    print(f"Found book matches for {matches_count} puzzles, {high_conf_count} with high confidence")

    return results

## 5. Integration with Previous Code
Now, let's integrate this with our previous approach for a comprehensive solution:


def comprehensive_tag_prediction(df, book_paths, fen_column='FEN', moves_column='Moves'):
    """
    Comprehensive approach to predict missing opening tags using all available methods.

    Args:
        df: DataFrame with chess puzzles
        book_paths: List of paths to polyglot opening book files
        fen_column: Column containing FEN strings
        moves_column: Column containing moves

    Returns:
        DataFrame with predicted opening tags and complete feature set
    """
    # 1. Extract basic position features
    position_features = extract_fen_features(df, fen_column)
    move_features = extract_opening_move_features(df, moves_column)
    eco_features = infer_eco_codes(df, moves_column)

    # 2. Use opening books for prediction
    book_predictions = predict_tags_using_books(df, book_paths, fen_column, moves_column)

    # 3. Use machine learning prediction from position features
    # Combine all features for ML
    ml_features = pd.concat([position_features, move_features, eco_features], axis=1)
    ml_features = ml_features.fillna(0)

    # Apply ML prediction
    ml_predictions, model, _ = predict_missing_opening_tags(df, ml_features)

    # 4. Combine predictions
    final_predictions = pd.DataFrame(index=df.index)
    final_predictions['has_original_tag'] = (~df['OpeningTags'].isna() & (df['OpeningTags'] != '')).astype(int)

    # For puzzles without original tags
    no_tags_mask = final_predictions['has_original_tag'] == 0

    # Default to ML prediction
    final_predictions['predicted_opening'] = ""
    final_predictions['prediction_confidence'] = 0.0
    final_predictions['prediction_method'] = ""

    # Add ML predictions
    ml_mask = no_tags_mask & (ml_predictions['prediction_confidence'] > 0)
    final_predictions.loc[ml_mask, 'predicted_opening'] = ml_predictions.loc[ml_mask, 'predicted_family']
    final_predictions.loc[ml_mask, 'prediction_confidence'] = ml_predictions.loc[ml_mask, 'prediction_confidence']
    final_predictions.loc[ml_mask, 'prediction_method'] = "ml"

    # Override with book predictions where they have higher confidence
    book_mask = no_tags_mask & (book_predictions['book_confidence'] > 0)
    book_better_mask = book_mask & (
            (book_predictions['book_confidence'] > final_predictions['prediction_confidence']) |
            (final_predictions['prediction_confidence'] == 0)
    )

    final_predictions.loc[book_better_mask, 'predicted_opening'] = book_predictions.loc[
        book_better_mask, 'predicted_opening']
    final_predictions.loc[book_better_mask, 'prediction_confidence'] = book_predictions.loc[
        book_better_mask, 'book_confidence']
    final_predictions.loc[book_better_mask, 'prediction_method'] = "book"

    # 5. Create enhanced tags column
    enhanced_tags = df['OpeningTags'].copy()

    # Add high-confidence predictions
    high_conf_mask = final_predictions['prediction_confidence'] >= 0.7
    for idx in final_predictions[high_conf_mask & no_tags_mask].index:
        opening = final_predictions.loc[idx, 'predicted_opening']
        method = final_predictions.loc[idx, 'prediction_method']
        enhanced_tags.loc[idx] = f"{opening} (predicted-{method})"

    # 6. Create final feature set
    df_enhanced = df.copy()
    df_enhanced['EnhancedOpeningTags'] = enhanced_tags

    opening_features = engineer_chess_opening_features(
        df_enhanced,
        tag_column='EnhancedOpeningTags',
        min_family_freq=20,
        min_variation_freq=10,
        min_keyword_freq=50
    )

    # Add prediction metadata
    opening_features['has_original_tag'] = final_predictions['has_original_tag']
    opening_features['has_predicted_tag'] = (
            final_predictions['has_original_tag'] |
            (final_predictions['prediction_confidence'] >= 0.7)
    ).astype(int)
    opening_features['tag_prediction_confidence'] = final_predictions['prediction_confidence']
    opening_features['tag_prediction_method'] = final_predictions['prediction_method']

    # Set confidence to 1.0 for original tags
    original_tag_mask = final_predictions['has_original_tag'] == 1
    opening_features.loc[original_tag_mask, 'tag_prediction_confidence'] = 1.0

    # Combine with position features
    final_features = pd.concat([
        opening_features,
        position_features,
        move_features,
        eco_features
    ], axis=1)

    # Fill any remaining NaN values
    final_features = final_features.fillna(0)

    # Calculate statistics
    original_tag_count = final_predictions['has_original_tag'].sum()
    enhanced_tag_count = opening_features['has_predicted_tag'].sum()

    print(f"Original puzzles with tags: {original_tag_count} ({original_tag_count / len(df) * 100:.2f}%)")
    print(f"Puzzles with tags after prediction: {enhanced_tag_count} ({enhanced_tag_count / len(df) * 100:.2f}%)")

    return final_features, final_predictions, enhanced_tags

## Usage with Your Opening Books
Here's how to use this with the chess opening books you mentioned:

# List of opening book paths
book_paths = [
    "Cerebellum_Light_3Merge_200916.bin",
    "gm2001.bin",
    "komodo.bin",
    "rodent.bin",
    "Titans.bin",
    "Human.bin"
]

# Filter to only include books that exist
import os

book_paths = [path for path in book_paths if os.path.exists(path)]

if not book_paths:
    print("Warning: No opening books found. Please check the file paths.")
else:
    # Load your data
    df = pd.read_csv('chess_puzzles.csv')

    # Run the comprehensive prediction
    features, predictions, enhanced_tags = comprehensive_tag_prediction(
        df,
        book_paths=book_paths,
        fen_column='FEN',
        moves_column='Moves'
    )

    # Save results
    features.to_csv('chess_features_with_predicted_tags.csv')
    predictions.to_csv('opening_tag_predictions.csv')

    # Create a version of the original dataset with enhanced tags
    df_enhanced = df.copy()
    df_enhanced['EnhancedOpeningTags'] = enhanced_tags
    df_enhanced.to_csv('chess_puzzles_with_enhanced_tags.csv')

## Benefits of Using These Opening Books
1. **High-Quality Opening Knowledge**: These books contain professionally curated opening theory.
2. **Transposition Handling**: Opening books understand how different move orders can transpose to the same position.
3. **Varied Coverage**: Having multiple books increases the coverage of different opening lines:
    - `Human.bin` likely contains common human opening choices
    - `gm2001.bin` probably focuses on grandmaster-level theory
    - `Titans.bin` and `komodo.bin` may include engine-preferred variations

4. **Higher Confidence**: When a position matches an opening book, we can have higher confidence in the opening classification compared to pure machine learning approaches.
5. **ECO Code Information**: These books can help map positions to the standardized ECO (Encyclopedia of Chess Openings) classification system.

By integrating these opening books with the feature extraction and machine learning approach, you should be able to significantly increase the percentage of puzzles with reliable opening tag information.
