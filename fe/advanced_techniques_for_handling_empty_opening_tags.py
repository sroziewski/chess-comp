Since 78.48% of your puzzles don't have opening tags, this is a critical issue to address. Here are several sophisticated techniques that go beyond basic regression trees:

You can train a model to predict opening tags for positions that don't have them using self-supervised learning:


def predict_missing_tags(df, feature_df, tag_column='OpeningTags',
                         position_features=None, confidence_threshold=0.7):
    """
    Use self-supervised learning to predict missing opening tags.

    Args:
        df: Original DataFrame with chess puzzles
        feature_df: DataFrame with engineered opening features
        tag_column: Column containing opening tags
        position_features: Chess position features (FEN-derived, material, etc.)
        confidence_threshold: Minimum confidence to apply predictions

    Returns:
        DataFrame with predicted tags and confidence scores
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import classification_report
    import numpy as np

    # Ensure we have position features
    if position_features is None or not all(col in df.columns for col in position_features):
        print("Warning: Position features not available for tag prediction")
        return feature_df

    # Identify puzzles with and without tags
    has_tags = ~df[tag_column].isna() & (df[tag_column] != '')

    if has_tags.sum() < 100:
        print("Too few puzzles with tags for reliable prediction")
        return feature_df

    # Extract main opening family from tags (first component, e.g., "Sicilian_Defense")
    def extract_main_family(tag_str):
        if pd.isna(tag_str) or not tag_str:
            return "Unknown"
        first_tag = tag_str.split()[0]  # Get first tag if multiple
        components = first_tag.split('_')
        if len(components) >= 2:
            return '_'.join(components[:2])
        return first_tag

    # Create target variable for opening families
    df_with_tags = df[has_tags].copy()
    df_without_tags = df[~has_tags].copy()

    df_with_tags['main_family'] = df_with_tags[tag_column].apply(extract_main_family)

    # Encode opening families
    le = LabelEncoder()
    y = le.fit_transform(df_with_tags['main_family'])

    # Use position features to predict openings
    X = df_with_tags[position_features]

    # Train model with cross-validation to assess reliability
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

    # Get cross-validated predictions to evaluate model quality
    cv_preds = cross_val_predict(model, X, y, cv=5)
    print("Cross-validation classification report:")
    print(classification_report(y, cv_preds))

    # Train on all data with tags
    model.fit(X, y)

    # Get predictions and probabilities for puzzles without tags
    X_predict = df_without_tags[position_features]
    predictions = model.predict(X_predict)
    probabilities = model.predict_proba(X_predict)

    # Get confidence scores (max probability)
    confidence_scores = np.max(probabilities, axis=1)

    # Add predictions to feature DataFrame
    indices_without_tags = df_without_tags.index
    feature_df.loc[indices_without_tags, 'predicted_family'] = le.inverse_transform(predictions)
    feature_df.loc[indices_without_tags, 'prediction_confidence'] = confidence_scores

    # For high-confidence predictions, add synthetic features
    high_confidence = confidence_scores >= confidence_threshold
    high_conf_indices = indices_without_tags[high_confidence]
    high_conf_families = le.inverse_transform(predictions[high_confidence])

    # Create synthetic features for high-confidence predictions
    for i, idx in enumerate(high_conf_indices):
        family = high_conf_families[i]

        # Update family one-hot encodings
        for fam_col in [col for col in feature_df.columns if col.startswith('open_fam_')]:
            feature_df.loc[idx, fam_col] = 1 if fam_col == f'open_fam_{family}' else 0

        # Update ECO code features based on predicted family
        eco_patterns = {
            'A': ['english', 'reti', 'bird', 'larsen', 'dutch', 'benoni'],
            'B': ['sicilian', 'caro-kann', 'pirc', 'modern', 'alekhine'],
            'C': ['ruy_lopez', 'italian', 'french', 'petrov', 'kings_gambit'],
            'D': ['queens_gambit', 'slav', 'grunfeld', 'tarrasch'],
            'E': ['indian', 'nimzo', 'queens_indian', 'kings_indian', 'catalan']
        }

        for eco, patterns in eco_patterns.items():
            if any(pat in family.lower() for pat in patterns):
                feature_df.loc[idx, f'eco_{eco}'] = 1

        # Set has_opening_tags_predicted flag
        feature_df.loc[idx, 'has_opening_tags_predicted'] = 1

    # Set flag for all puzzles
    feature_df['has_opening_tags_predicted'] = feature_df.get('has_opening_tags_predicted', 0).fillna(0)

    print(f"Added predictions for {len(indices_without_tags)} puzzles, "
          f"{high_confidence.sum()} with high confidence (≥{confidence_threshold})")

    return feature_df

## 2. Clustering-Based Tag Inference
Using clustering to group similar positions and transfer opening tags:


def cluster_based_tag_inference(df, feature_df, position_features, n_clusters=50):
    """
    Use clustering to infer opening tags for puzzles without them.

    Args:
        df: Original DataFrame with chess puzzles
        feature_df: DataFrame with engineered opening features
        position_features: Chess position features for clustering
        n_clusters: Number of clusters to create

    Returns:
        DataFrame with inferred opening tags based on clusters
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Identify puzzles with and without tags
    has_tags = feature_df['has_opening_tags'] == 1

    if has_tags.sum() < 100:
        print("Too few puzzles with tags for reliable clustering")
        return feature_df

    # Standardize features for clustering
    scaler = StandardScaler()
    X = scaler.fit_transform(df[position_features])

    # Perform clustering (use MiniBatchKMeans for very large datasets)
    if len(X) > 100000:
        print("Using MiniBatchKMeans for large dataset")
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    else:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    # Fit clusters
    cluster_labels = clusterer.fit_predict(X)

    # Add cluster labels to feature DataFrame
    feature_df['position_cluster'] = cluster_labels

    # For each cluster, analyze opening tag distribution
    cluster_tag_distribution = {}

    for cluster_id in range(n_clusters):
        # Get indices of puzzles in this cluster that have tags
        cluster_with_tags = feature_df.index[(feature_df['position_cluster'] == cluster_id) & has_tags]

        if len(cluster_with_tags) > 0:
            # Get opening families for this cluster
            cluster_families = [extract_main_family(df.loc[idx, 'OpeningTags'])
                                for idx in cluster_with_tags]

            # Count family frequencies
            from collections import Counter
            family_counts = Counter(cluster_families)

            # Store dominant family and its proportion
            dominant_family, count = family_counts.most_common(1)[0]
            proportion = count / len(cluster_families)

            cluster_tag_distribution[cluster_id] = {
                'dominant_family': dominant_family,
                'proportion': proportion,
                'sample_size': len(cluster_families),
                'all_families': dict(family_counts)
            }

    # For puzzles without tags, add cluster-based family features
    for idx in feature_df.index[~has_tags]:
        cluster_id = feature_df.loc[idx, 'position_cluster']

        if cluster_id in cluster_tag_distribution:
            cluster_info = cluster_tag_distribution[cluster_id]

            # Only use clusters with sufficient sample size and confidence
            if (cluster_info['sample_size'] >= 5 and
                    cluster_info['proportion'] >= 0.5):

                dominant_family = cluster_info['dominant_family']
                confidence = cluster_info['proportion']

                # Add inferred features
                feature_df.loc[idx, 'cluster_inferred_family'] = dominant_family
                feature_df.loc[idx, 'cluster_inference_confidence'] = confidence

                # Optionally set synthetic features for high-confidence inferences
                if confidence >= 0.7:
                    # Update family one-hot encodings (similar to previous function)
                    for fam_col in [col for col in feature_df.columns if col.startswith('open_fam_')]:
                        if fam_col == f'open_fam_{dominant_family}':
                            feature_df.loc[idx, fam_col] = confidence  # Scale by confidence

                    # Update ECO and style features as well
                    # (Code similar to previous function)

                    feature_df.loc[idx, 'has_cluster_inferred_tags'] = 1

    # Set flag for all puzzles
    feature_df['has_cluster_inferred_tags'] = feature_df.get('has_cluster_inferred_tags', 0).fillna(0)

    # Count how many puzzles got inferred tags
    inferred_count = (feature_df['has_cluster_inferred_tags'] == 1).sum()
    print(f"Added cluster-inferred tags for {inferred_count} puzzles without tags")

    return feature_df

## 3. Fuzzy Position Matching with Opening Books
Use opening book data to find the most similar known opening positions:


def fuzzy_position_matching(df, feature_df, fen_column='FEN',
                            opening_book_path='opening_book.csv', max_dist=2):
    """
    Match puzzle positions to opening book positions to infer tags.

    Args:
        df: Original DataFrame with chess puzzles
        feature_df: DataFrame with engineered opening features
        fen_column: Column containing FEN position string
        opening_book_path: Path to CSV file with opening book positions and tags
        max_dist: Maximum edit distance/move distance for matching

    Returns:
        DataFrame with opening tags inferred from position matching
    """
    import pandas as pd
    import chess
    import chess.pgn
    from tqdm import tqdm

    # Prepare FEN processing function
    def normalize_fen(fen_str):
        """Extract the piece placement part of FEN and normalize it"""
        if pd.isna(fen_str) or not fen_str:
            return ""
        return fen_str.split()[0]  # Take just the piece placement part

    # Load opening book
    try:
        opening_book = pd.read_csv(opening_book_path)
        print(f"Loaded opening book with {len(opening_book)} positions")
    except Exception as e:
        print(f"Error loading opening book: {e}")
        return feature_df

    # Create dictionaries for fast lookup
    book_positions = {}
    for _, row in opening_book.iterrows():
        pos_fen = normalize_fen(row['FEN'])
        opening_name = row['OpeningName']
        eco = row.get('ECO', '')

        if pos_fen:
            book_positions[pos_fen] = {'name': opening_name, 'eco': eco}

    # Function to compare two positions
    def position_similarity(fen1, fen2):
        """Calculate similarity between two positions (0-1)"""
        try:
            board1 = chess.Board(fen1)
            board2 = chess.Board(fen2)

            # Count differences in piece placement
            diff_count = 0
            for square in chess.SQUARES:
                piece1 = board1.piece_at(square)
                piece2 = board2.piece_at(square)
                if piece1 != piece2:
                    diff_count += 1

            # Calculate similarity score (0-1)
            similarity = 1 - (diff_count / 64)
            return similarity
        except:
            return 0

    # Identify puzzles without tags
    no_tags = feature_df['has_opening_tags'] == 0
    puzzles_without_tags = df[no_tags].index

    # Match positions to opening book
    matches_found = 0

    for idx in tqdm(puzzles_without_tags, desc="Matching Positions"):
        if idx not in df.index or fen_column not in df.columns:
            continue

        puzzle_fen = df.loc[idx, fen_column]
        norm_fen = normalize_fen(puzzle_fen)

        if not norm_fen:
            continue

        # Check for exact match first
        if norm_fen in book_positions:
            opening_info = book_positions[norm_fen]
            feature_df.loc[idx, 'book_matched_opening'] = opening_info['name']
            feature_df.loc[idx, 'book_match_confidence'] = 1.0
            feature_df.loc[idx, 'book_matched_eco'] = opening_info['eco']
            feature_df.loc[idx, 'has_book_matched_tag'] = 1
            matches_found += 1
            continue

        # If no exact match, find close matches
        best_match = None
        best_similarity = 0.85  # Minimum threshold

        # Check a sample of positions for performance reasons
        for book_fen in list(book_positions.keys())[:1000]:  # Limit to 1000 for performance
            similarity = position_similarity(norm_fen, book_fen)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = book_positions[book_fen]

        if best_match:
            feature_df.loc[idx, 'book_matched_opening'] = best_match['name']
            feature_df.loc[idx, 'book_match_confidence'] = best_similarity
            feature_df.loc[idx, 'book_matched_eco'] = best_match['eco']
            feature_df.loc[idx, 'has_book_matched_tag'] = 1
            matches_found += 1

    # Set flag for all puzzles
    feature_df['has_book_matched_tag'] = feature_df.get('has_book_matched_tag', 0).fillna(0)

    print(f"Matched {matches_found} positions to opening book")

    # Update feature columns based on matched openings
    match_cols = ['book_matched_opening', 'book_match_confidence']
    if all(col in feature_df.columns for col in match_cols):
        # Process matched openings to update feature columns
        for idx in feature_df.index[feature_df['has_book_matched_tag'] == 1]:
            opening_name = feature_df.loc[idx, 'book_matched_opening']
            confidence = feature_df.loc[idx, 'book_match_confidence']

            # Extract family name
            components = opening_name.split()
            family = components[0] if components else ""

            # Update family columns with confidence-weighted values
            for fam_col in [col for col in feature_df.columns if col.startswith('open_fam_')]:
                if family in fam_col:
                    feature_df.loc[idx, fam_col] = confidence

            # Update ECO code features based on matched ECO
            eco = feature_df.loc[idx, 'book_matched_eco']
            if eco and len(eco) >= 1:
                eco_letter = eco[0]
                feature_df.loc[idx, f'eco_{eco_letter}'] = confidence

    return feature_df

## 4. Move Sequence Analysis for Opening Identification
Analyze the initial moves of the puzzle to identify potential openings:


def move_sequence_analysis(df, feature_df, moves_column='Moves'):
    """
    Analyze puzzle move sequences to identify opening patterns.

    Args:
        df: Original DataFrame with chess puzzles
        feature_df: DataFrame with engineered opening features
        moves_column: Column containing puzzle moves

    Returns:
        DataFrame with opening features inferred from move sequences
    """
    import chess
    import chess.pgn
    import io
    from collections import defaultdict

    # Define common opening move sequences
    opening_patterns = {
        'e4 e5': {'family': 'Kings_Pawn', 'eco': 'C'},
        'e4 c5': {'family': 'Sicilian_Defense', 'eco': 'B'},
        'e4 e6': {'family': 'French_Defense', 'eco': 'C'},
        'e4 c6': {'family': 'Caro-Kann_Defense', 'eco': 'B'},
        'd4 d5': {'family': 'Queens_Pawn', 'eco': 'D'},
        'd4 Nf6': {'family': 'Indian_Defense', 'eco': 'E'},
        'c4 e5': {'family': 'English_Opening', 'eco': 'A'},
        'Nf3 d5': {'family': 'Reti_Opening', 'eco': 'A'},

        # More detailed patterns
        'e4 e5 Nf3': {'family': 'Kings_Pawn', 'eco': 'C'},
        'e4 e5 Nf3 Nc6': {'family': 'Open_Game', 'eco': 'C'},
        'e4 e5 Nf3 Nc6 Bc4': {'family': 'Italian_Game', 'eco': 'C'},
        'e4 e5 Nf3 Nc6 Bb5': {'family': 'Ruy_Lopez', 'eco': 'C'},
        'd4 d5 c4': {'family': 'Queens_Gambit', 'eco': 'D'},
        'd4 d5 c4 e6': {'family': 'Queens_Gambit_Declined', 'eco': 'D'},
        'd4 d5 c4 c6': {'family': 'Slav_Defense', 'eco': 'D'},
        'd4 Nf6 c4 g6': {'family': 'Kings_Indian_Defense', 'eco': 'E'},
        'd4 Nf6 c4 e6': {'family': 'Queens_Indian_Defense', 'eco': 'E'},
        'e4 c5 Nf3': {'family': 'Sicilian_Defense', 'eco': 'B'},
        'e4 c5 Nf3 d6': {'family': 'Sicilian_Defense', 'eco': 'B'},
        'e4 c5 Nf3 Nc6': {'family': 'Sicilian_Defense', 'eco': 'B'}
    }

    # Function to extract initial move sequence
    def extract_initial_moves(moves_str, max_moves=8):
        """Extract the first few moves from a move sequence"""
        if pd.isna(moves_str) or not moves_str:
            return ""

        # Try to parse as PGN
        try:
            pgn = io.StringIO(
                f"[Event \"?\"]\n[Site \"?\"]\n[Date \"????.??.??\"]\n[Round \"?\"]\n[White \"?\"]\n[Black \"?\"]\n[Result \"*\"]\n\n{moves_str}\n*\n")
            game = chess.pgn.read_game(pgn)

            moves = []
            board = game.board()

            # Extract the first few moves
            for i, move in enumerate(game.mainline_moves()):
                if i >= max_moves:
                    break
                san = board.san(move)
                moves.append(san)
                board.push(move)

            # Create move sequence string
            move_sequence = ' '.join(moves)
            return move_sequence

        except Exception:
            # Fall back to simple parsing if PGN parsing fails
            return ' '.join(moves_str.split()[:max_moves])

    # Process puzzles without tags
    no_tags = feature_df['has_opening_tags'] == 0
    puzzles_without_tags = df[no_tags].index

    patterns_found = 0

    for idx in puzzles_without_tags:
        if idx not in df.index or moves_column not in df.columns:
            continue

        moves_str = df.loc[idx, moves_column]
        initial_moves = extract_initial_moves(moves_str)

        if not initial_moves:
            continue

        # Look for matching patterns
        matched_family = None
        matched_eco = None
        match_length = 0

        # Try to match the longest possible sequence first
        for pattern, info in sorted(opening_patterns.items(),
                                    key=lambda x: len(x[0]), reverse=True):
            if initial_moves.startswith(pattern):
                # Found a match
                matched_family = info['family']
                matched_eco = info['eco']
                match_length = len(pattern.split())
                break

        if matched_family:
            # Calculate confidence based on match length
            confidence = min(0.95, 0.5 + (match_length * 0.1))

            feature_df.loc[idx, 'moves_inferred_family'] = matched_family
            feature_df.loc[idx, 'moves_match_confidence'] = confidence
            feature_df.loc[idx, 'moves_matched_eco'] = matched_eco
            feature_df.loc[idx, 'has_moves_inferred_tag'] = 1

            # Update family columns
            for fam_col in [col for col in feature_df.columns if col.startswith('open_fam_')]:
                if matched_family.lower() in fam_col.lower():
                    feature_df.loc[idx, fam_col] = confidence

            # Update ECO code
            if matched_eco:
                feature_df.loc[idx, f'eco_{matched_eco}'] = confidence

            patterns_found += 1

    # Set flag for all puzzles
    feature_df['has_moves_inferred_tag'] = feature_df.get('has_moves_inferred_tag', 0).fillna(0)

    print(f"Found opening patterns in {patterns_found} puzzles without tags")
    return feature_df

## 5. Ensemble Method for Tag Inference
Combine multiple methods for more robust tag inference:


    def ensemble_tag_inference(df, feature_df, position_features=None, fen_column='FEN',
                               moves_column='Moves', opening_book_path=None):
        """
        Ensemble method combining multiple tag inference techniques.

        Args:
            df: Original DataFrame with chess puzzles
            feature_df: DataFrame with engineered opening features
            position_features: Features for machine learning/clustering
            fen_column: Column containing FEN position string
            moves_column: Column containing puzzle moves
            opening_book_path: Path to opening book file

        Returns:
            DataFrame with ensemble inferred opening features
        """
        # Track original tag status
        feature_df['original_has_tags'] = feature_df['has_opening_tags'].copy()

        # Apply each method if the required inputs are available
        if position_features is not None and len(position_features) > 0:
            # ML-based prediction
            feature_df = predict_missing_tags(df, feature_df,
                                              position_features=position_features)

            # Clustering-based inference
            feature_df = cluster_based_tag_inference(df, feature_df,
                                                     position_features=position_features)

        # Position matching if FEN and opening book available
        if fen_column in df.columns and opening_book_path is not None:
            feature_df = fuzzy_position_matching(df, feature_df,
                                                 fen_column=fen_column,
                                                 opening_book_path=opening_book_path)

        # Move sequence analysis if moves column available
        if moves_column in df.columns:
            feature_df = move_sequence_analysis(df, feature_df,
                                                moves_column=moves_column)

        # Create ensemble confidence and family
        methods = []
        if 'prediction_confidence' in feature_df.columns:
            methods.append(('predicted_family', 'prediction_confidence'))
        if 'cluster_inference_confidence' in feature_df.columns:
            methods.append(('cluster_inferred_family', 'cluster_inference_confidence'))
        if 'book_match_confidence' in feature_df.columns:
            methods.append(('book_matched_opening', 'book_match_confidence'))
        if 'moves_match_confidence' in feature_df.columns:
            methods.append(('moves_inferred_family', 'moves_match_confidence'))

        # Combine methods with weighted voting
        if methods:
            for idx in feature_df.index[feature_df['original_has_tags'] == 0]:
                votes = {}
                total_weight = 0

                for family_col, conf_col in methods:
                    if family_col in feature_df.columns and conf_col in feature_df.columns:
                        family = feature_df.loc[idx, family_col]
                        confidence = feature_df.loc[idx, conf_col]

                        if pd.notna(family) and pd.notna(confidence) and family and confidence > 0:
                            if family not in votes:
                                votes[family] = 0
                            votes[family] += confidence
                            total_weight += confidence

                if votes and total_weight > 0:
                    # Get top vote-getter
                    top_family = max(votes.items(), key=lambda x: x[1])[0]
                    ensemble_confidence = votes[top_family] / total_weight

                    feature_df.loc[idx, 'ensemble_family'] = top_family
                    feature_df.loc[idx, 'ensemble_confidence'] = ensemble_confidence

                    # Add high-confidence inferred features
                    if ensemble_confidence >= 0.6:
                        # Update family columns
                        for fam_col in [col for col in feature_df.columns if col.startswith('open_fam_')]:
                            if top_family.lower().replace(' ', '_') in fam_col.lower():
                                feature_df.loc[idx, fam_col] = ensemble_confidence

                        # Update has tags flag for model training
                        feature_df.loc[idx, 'has_inferred_tags'] = 1

        # Set flag for all puzzles
        feature_df['has_inferred_tags'] = feature_df.get('has_inferred_tags', 0).fillna(0)

        # Count inference results
        inference_count = (feature_df['has_inferred_tags'] == 1).sum()
        print(f"Added ensemble tag inference for {inference_count} puzzles without tags")

        return feature_df

## 6. Transposition Awareness for Opening Identification
Identify transpositions between opening move orders:


def transposition_aware_inference(df, feature_df, moves_column='Moves'):
    """
    Identify opening transpositions to improve tag inference.

    Args:
        df: Original DataFrame with chess puzzles
        feature_df: DataFrame with engineered opening features
        moves_column: Column containing puzzle moves

    Returns:
        DataFrame with transposition-aware opening features
    """
    import chess
    import chess.pgn
    import io
    from collections import defaultdict

    # Function to get position signature after N moves
    def get_position_signature(moves_str, n_moves=4):
        """Get a unique signature for a position after n_moves"""
        if pd.isna(moves_str) or not moves_str:
            return None

        try:
            pgn = io.StringIO(
                f"[Event \"?\"]\n[Site \"?\"]\n[Date \"????.??.??\"]\n[Round \"?\"]\n[White \"?\"]\n[Black \"?\"]\n[Result \"*\"]\n\n{moves_str}\n*\n")
            game = chess.pgn.read_game(pgn)

            board = game.board()
            move_count = 0

            for move in game.mainline_moves():
                board.push(move)
                move_count += 1
                if move_count >= n_moves * 2:  # 2 ply per move
                    break

            # Return FEN as position signature
            return board.fen().split(' ')[0]  # Just the piece placement
        except:
            return None

    # Build dictionary of position signatures from puzzles with tags
    print("Building position signature database...")
    position_openings = defaultdict(list)

    for idx in feature_df.index[feature_df['has_opening_tags'] == 1]:
        if idx not in df.index or moves_column not in df.columns:
            continue

        moves_str = df.loc[idx, moves_column]
        opening_tag = df.loc[idx, 'OpeningTags'].split()[0] if not pd.isna(df.loc[idx, 'OpeningTags']) else None

        if not opening_tag:
            continue

        # Get position signatures at different move depths
        for n_moves in [3, 4, 5, 6]:
            sig = get_position_signature(moves_str, n_moves)
            if sig:
                position_openings[sig].append(opening_tag)

    # For each signature, identify the most common opening
    signature_to_opening = {}
    for sig, openings in position_openings.items():
        from collections import Counter
        counts = Counter(openings)
        if counts:
            most_common = counts.most_common(1)[0]
            # Store opening and its proportion of occurrences
            signature_to_opening[sig] = {
                'opening': most_common[0],
                'confidence': most_common[1] / len(openings)
            }

    print(f"Built database with {len(signature_to_opening)} position signatures")

    # Apply to puzzles without tags
    transpositions_found = 0

    for idx in feature_df.index[feature_df['has_opening_tags'] == 0]:
        if idx not in df.index or moves_column not in df.columns:
            continue

        moves_str = df.loc[idx, moves_column]

        # Try different move depths
        for n_moves in [6, 5, 4, 3]:  # Try deeper positions first
            sig = get_position_signature(moves_str, n_moves)
            if sig and sig in signature_to_opening:
                info = signature_to_opening[sig]
                opening = info['opening']
                confidence = info['confidence'] * 0.9  # Slight reduction for transpositions

                feature_df.loc[idx, 'transposition_opening'] = opening
                feature_df.loc[idx, 'transposition_confidence'] = confidence
                feature_df.loc[idx, 'has_transposition_tag'] = 1

                # Update feature columns if confidence is high enough
                if confidence >= 0.7:
                    # Extract family name
                    family = opening.split('_')[0] if '_' in opening else opening

                    # Update family columns
                    for fam_col in [col for col in feature_df.columns if col.startswith('open_fam_')]:
                        if family.lower() in fam_col.lower():
                            feature_df.loc[idx, fam_col] = confidence

                transpositions_found += 1
                break  # Stop after first match

    # Set flag for all puzzles
    feature_df['has_transposition_tag'] = feature_df.get('has_transposition_tag', 0).fillna(0)

    print(f"Found {transpositions_found} opening transpositions")
    return feature_df

## 7. Integrated Solution for Empty Tags
Here's how to combine all these methods into a comprehensive solution:


def handle_empty_opening_tags(df, feature_df, position_features=None,
                              fen_column='FEN', moves_column='Moves',
                              opening_book_path=None):
    """
    Comprehensive solution for handling puzzles without opening tags.

    Args:
        df: Original DataFrame with chess puzzles
        feature_df: DataFrame with engineered opening features
        position_features: Features for ML and clustering
        fen_column: Column containing FEN strings
        moves_column: Column containing moves
        opening_book_path: Path to opening book file

    Returns:
        DataFrame with comprehensive tag inference
    """
    print("Handling puzzles without opening tags...")

    # Start with ensemble method that integrates multiple approaches
    feature_df = ensemble_tag_inference(
        df, feature_df,
        position_features=position_features,
        fen_column=fen_column,
        moves_column=moves_column,
        opening_book_path=opening_book_path
    )

    # Add transposition awareness
    if moves_column in df.columns:
        feature_df = transposition_aware_inference(df, feature_df, moves_column=moves_column)

    # Create flag for high-confidence inferences
    high_conf_cols = [
        ('prediction_confidence', 'predicted_family'),
        ('cluster_inference_confidence', 'cluster_inferred_family'),
        ('book_match_confidence', 'book_matched_opening'),
        ('moves_match_confidence', 'moves_inferred_family'),
        ('transposition_confidence', 'transposition_opening'),
        ('ensemble_confidence', 'ensemble_family')
    ]

    feature_df['high_confidence_inference'] = 0

    for conf_col, fam_col in high_conf_cols:
        if conf_col in feature_df.columns and fam_col in feature_df.columns:
            # Mark as high confidence if any method has >= 0.8 confidence
            high_conf = (feature_df[conf_col] >= 0.8) & (feature_df['has_opening_tags'] == 0)
            feature_df.loc[high_conf, 'high_confidence_inference'] = 1

    # Create synthetic "has tags" column that includes high-confidence inferences
    feature_df['effective_has_tags'] = (
            (feature_df['has_opening_tags'] == 1) |
            (feature_df['high_confidence_inference'] == 1)
    ).astype(int)

    # Calculate statistics
    original_tag_pct = feature_df['has_opening_tags'].mean() * 100
    effective_tag_pct = feature_df['effective_has_tags'].mean() * 100

    print(f"Original puzzles with tags: {original_tag_pct:.1f}%")
    print(f"Puzzles with tags after inference: {effective_tag_pct:.1f}%")

    return feature_df

## 8. Training Models with Weighted Samples
Once you've inferred tags, use weighted training to balance the confidence:


def train_weighted_model(feature_df, target_column, feature_columns):
    """
    Train a model using weights based on tag confidence.

    Args:
        feature_df: DataFrame with features including inferred tags
        target_column: Target variable to predict
        feature_columns: List of feature columns to use

    Returns:
        Trained model and performance metrics
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import classification_report
    import numpy as np

    # Calculate sample weights based on tag confidence
    def get_sample_weights(df):
        weights = np.ones(len(df))

        # Original tags get weight 1.0
        original_tags = df['has_opening_tags'] == 1

        # High confidence inferences get weight 0.8
        high_conf = df['high_confidence_inference'] == 1
        weights[high_conf] = 0.8

        # Medium confidence inferences get weight 0.5
        medium_conf = (~original_tags & ~high_conf &
                       ((df.get('ensemble_confidence', 0) >= 0.6) |
                        (df.get('prediction_confidence', 0) >= 0.7)))
        weights[medium_conf] = 0.5

        # Low confidence or no inference get weight 0.3
        low_conf = (~original_tags & ~high_conf & ~medium_conf)
        weights[low_conf] = 0.3

        return weights

    # Prepare data
    X = feature_df[feature_columns]
    y = feature_df[target_column]

    # Get sample weights
    sample_weights = get_sample_weights(feature_df)

    # Split data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42
    )

    # Train model with sample weights
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train, sample_weight=w_train)

    # Evaluate
    print("Model performance:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Compare performance on subsets with different tag confidence
    def evaluate_subset(mask, name):
        if mask.sum() > 0:
            subset_pred = model.predict(X_test[mask])
            print(f"\nPerformance on {name} (n={mask.sum()}):")
            print(classification_report(y_test[mask], subset_pred))

    # Evaluate on original tags vs inferred tags
    original_tags_mask = feature_df.loc[X_test.index, 'has_opening_tags'] == 1
    inferred_tags_mask = feature_df.loc[X_test.index, 'high_confidence_inference'] == 1
    no_tags_mask = (~original_tags_mask & ~inferred_tags_mask)

    evaluate_subset(original_tags_mask, "puzzles with original tags")
    evaluate_subset(inferred_tags_mask, "puzzles with inferred tags")
    evaluate_subset(no_tags_mask, "puzzles without reliable tags")

    return model
## Recommended Approach for Your Chess Dataset
Given your specific situation with 78.48% of puzzles lacking opening tags, I recommend:
1. **Start with the ensemble method** that combines multiple inference techniques.
2. **Use confidence thresholds** to categorize puzzles:
    - Original tags (21.52%)
    - High-confidence inferences (maybe ~20-30%)
    - Medium-confidence inferences (maybe ~20%)
    - Low/no confidence (remaining puzzles)

3. **Apply weighted training** that uses these confidence levels.
4. **Create dual models** for puzzles with and without reliable opening information.

This approach will make the most of the available data while handling the inherent uncertainty in the inferred tags.


# I ma not sure how you can predit missing value based of what, what is your assumption?
    
