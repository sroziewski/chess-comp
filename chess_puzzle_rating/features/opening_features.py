"""
Module for engineering features from chess opening tags.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from collections import Counter
import re
from tqdm import tqdm

from chess_puzzle_rating.features.opening_tags import OPENING_TAGS


def engineer_chess_opening_features(df, tag_column='OpeningTags',
                                   min_family_freq=50,
                                   min_variation_freq=25,
                                   min_keyword_freq=100,
                                   max_components_level1=100,
                                   max_components_level2=75,
                                   n_svd_components_family=10,
                                   n_svd_components_full=15,
                                   n_hash_features=20,
                                   max_tag_depth=5,
                                   additional_columns=None,
                                   use_predefined_tags=True):
    """
    Comprehensive feature engineering for chess opening tags.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing chess data
    tag_column : str, optional
        Column name containing opening tags, by default 'OpeningTags'
    min_family_freq : int, optional
        Minimum frequency for an opening family to be one-hot encoded, by default 50
    min_variation_freq : int, optional
        Minimum frequency for a primary variation to be one-hot encoded, by default 25
    min_keyword_freq : int, optional
        Minimum frequency for a keyword to be one-hot encoded, by default 100
    max_components_level1 : int, optional
        Max number of most frequent level 1 openings to one-hot encode, by default 100
    max_components_level2 : int, optional
        Max number of most frequent level 2 openings to one-hot encode, by default 75
    n_svd_components_family : int, optional
        Number of SVD components for opening family embeddings, by default 10
    n_svd_components_full : int, optional
        Number of SVD components for full tag embeddings, by default 15
    n_hash_features : int, optional
        Number of features to use for hashing vectorizer, by default 20
    max_tag_depth : int, optional
        Maximum depth of tag hierarchy to consider, by default 5
    additional_columns : list, optional
        List of additional columns to use for interaction features, by default None
    use_predefined_tags : bool, optional
        Whether to use the predefined list of opening tags from opening_tags.py, by default True

    Returns
    -------
    pandas.DataFrame
        DataFrame with engineered opening features
    """
    print("Engineering Chess Opening Features...")

    # Use a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    df_copy[tag_column] = df_copy[tag_column].fillna('')

    # --- Helper Functions ---
    def parse_opening_tag(tag_string):
        """
        Parses opening tags with enhanced chess domain knowledge.

        Parameters
        ----------
        tag_string : str
            String containing opening tags

        Returns
        -------
        tuple
            (raw_tags_list, parsed_tags_components_list)
        """
        if pd.isna(tag_string) or not tag_string.strip():
            return [], []

        individual_tags = tag_string.split(' ')  # Assuming tags are space-separated
        parsed_tags_components = []

        for tag in individual_tags:
            # Split by underscore, then by potential CamelCase if underscores are missing
            components = []
            for part in tag.split('_'):
                # Split CamelCase: 'McDonnellGambit' -> ['Mc', 'Donnell', 'Gambit']
                camel_split_parts = re.findall(r'[A-Z][a-z0-9\'\-]*|[A-Z]+(?![a-z])', part)
                if camel_split_parts:
                    components.extend(camel_split_parts)
                elif part:  # if not empty and not camel case (e.g. "Accepted")
                    components.append(part)

            if components:
                # Apply chess-specific parsing rules
                processed_components = []
                i = 0
                while i < len(components):
                    # Common prefixes that should be kept with the following word
                    prefixes = ['King', 'Kings', 'Queen', 'Queens', 'Ruy', 'Semi', 'Neo', 'Nimzo', 'Bogo']
                    if i < len(components) - 1 and components[i] in prefixes:
                        processed_components.append(f"{components[i]}_{components[i + 1]}")
                        i += 2
                    else:
                        processed_components.append(components[i])
                        i += 1

                parsed_tags_components.append([c.lower() for c in processed_components])

        return individual_tags, parsed_tags_components

    def standardize_opening_family(components):
        """
        Standardize opening family names based on chess theory.

        Parameters
        ----------
        components : list
            List of components from a parsed opening tag

        Returns
        -------
        str
            Standardized opening family name
        """
        if not components:
            return "unknown"

        # Handle special cases and normalize common variations
        if len(components) >= 1:
            # Check for common opening families that might be parsed differently
            first_comp = components[0].lower()

            # Handle common prefixed openings
            if first_comp.startswith(('king', 'kings')):
                return 'kings' + first_comp[5:] if first_comp.startswith('kings') else 'kings_' + first_comp[4:]
            elif first_comp.startswith(('queen', 'queens')):
                return 'queens' + first_comp[6:] if first_comp.startswith('queens') else 'queens_' + first_comp[5:]

            # Other common normalization cases
            if first_comp == 'ruy' and len(components) > 1 and components[1].lower() == 'lopez':
                return 'ruy_lopez'
            elif first_comp == 'nimzo' and len(components) > 1 and 'indian' in components[1].lower():
                return 'nimzo_indian'

            return first_comp

        return "unknown"

    # --- Pre-computation: Parse all tags and gather statistics ---
    all_parsed_tags_info = []  # Store (original_index, raw_tags_list, parsed_components_list)
    all_level1_families = Counter()
    all_level2_variations = Counter()
    all_keywords = Counter()
    all_full_tags_for_svd = []  # For SVD on common full tags

    # Create a set of predefined tags for faster lookup
    predefined_tags_set = set(OPENING_TAGS) if use_predefined_tags else set()

    # Extract families and variations from predefined tags
    predefined_families = set()
    predefined_variations = {}

    if use_predefined_tags:
        for tag in OPENING_TAGS:
            parts = tag.split('_')
            family = parts[0]
            predefined_families.add(family)

            if len(parts) > 1:
                variation = '_'.join(parts[1:])
                if family not in predefined_variations:
                    predefined_variations[family] = set()
                predefined_variations[family].add(variation)

    for idx, tag_string in tqdm(df_copy[tag_column].items(), desc="Parsing Tags"):
        raw_tags = []
        parsed_tags_list = []

        # If using predefined tags, validate against the list
        if use_predefined_tags and tag_string:
            # Split by spaces to handle multiple tags
            for tag in tag_string.split():
                if tag in predefined_tags_set:
                    raw_tags.append(tag)
                    # Parse the tag into components
                    components = tag.split('_')
                    parsed_components = [comp.lower() for comp in components]
                    parsed_tags_list.append(parsed_components)
        else:
            # Use the original parsing logic
            raw_tags, parsed_tags_list = parse_opening_tag(tag_string)

        all_parsed_tags_info.append({'idx': idx, 'raw_tags': raw_tags, 'parsed_tags': parsed_tags_list})

        # For simplicity, consider the first tag in a multi-tag string as the "primary"
        primary_tag_components = parsed_tags_list[0] if parsed_tags_list else []

        if not primary_tag_components:
            all_full_tags_for_svd.append("")  # For SVD alignment
            continue

        # Get standardized family name
        family_name = standardize_opening_family(primary_tag_components)
        all_level1_families[family_name] += 1

        # Determine variation level components
        if len(primary_tag_components) >= 2:
            # If there are designated types like Defense, Attack, etc.
            variation_types = ['defense', 'attack', 'variation', 'gambit', 'system', 'opening']
            type_indices = [i for i, comp in enumerate(primary_tag_components) if comp.lower() in variation_types]

            if type_indices:
                first_type_idx = type_indices[0]
                if first_type_idx > 0:  # There's a family name before the type
                    level2_components = primary_tag_components[:first_type_idx + 1]
                    variation_name = "_".join(level2_components).lower()
                    all_level2_variations[variation_name] += 1

                    # Add another level with one more component if available
                    if first_type_idx + 1 < len(primary_tag_components):
                        level3_components = primary_tag_components[:first_type_idx + 2]
                        level3_variation = "_".join(level3_components).lower()
                        all_level2_variations[level3_variation] += 1
            else:
                # No type designation, take first 2-3 components as variation
                if len(primary_tag_components) >= 3:
                    variation_name = "_".join(primary_tag_components[:3]).lower()
                    all_level2_variations[variation_name] += 1
                elif len(primary_tag_components) >= 2:
                    variation_name = "_".join(primary_tag_components[:2]).lower()
                    all_level2_variations[variation_name] += 1

        # Count keyword occurrences
        for component_word in primary_tag_components:
            all_keywords[component_word.lower()] += 1

        # Store the full first tag for SVD processing
        all_full_tags_for_svd.append(raw_tags[0] if raw_tags else "")

    # --- Select Top N for One-Hot Encoding ---
    if use_predefined_tags:
        # When using predefined tags, prioritize them in the feature selection
        # Convert predefined families to lowercase for consistent comparison
        predefined_families_lower = {fam.lower() for fam in predefined_families}

        # For families, include all predefined families that meet the frequency threshold
        predefined_families_with_counts = []
        for fam_lower in predefined_families_lower:
            if fam_lower in all_level1_families and all_level1_families[fam_lower] >= min_family_freq:
                predefined_families_with_counts.append((fam_lower, all_level1_families[fam_lower]))

        predefined_families_with_counts.sort(key=lambda x: x[1], reverse=True)

        # Take top N predefined families
        top_predefined_families = [fam for fam, _ in predefined_families_with_counts[:max_components_level1]]

        # Add remaining families from frequency count if needed
        remaining_slots = max_components_level1 - len(top_predefined_families)
        if remaining_slots > 0:
            # Get families not already included
            remaining_families = [fam for fam, count in all_level1_families.most_common() 
                                 if fam.lower() not in [f.lower() for f in top_predefined_families] 
                                 and count >= min_family_freq]
            top_families = top_predefined_families + remaining_families[:remaining_slots]
        else:
            top_families = top_predefined_families

        # Similar approach for variations
        all_predefined_variations = []
        for family, variations in predefined_variations.items():
            family_lower = family.lower()
            for variation in variations:
                variation_lower = variation.lower()
                full_variation = f"{family_lower}_{variation_lower}"
                # Check if this variation exists in the data
                if full_variation in all_level2_variations and all_level2_variations[full_variation] >= min_variation_freq:
                    all_predefined_variations.append((full_variation, all_level2_variations[full_variation]))

        all_predefined_variations.sort(key=lambda x: x[1], reverse=True)
        top_predefined_variations = [var for var, _ in all_predefined_variations[:max_components_level2]]

        remaining_slots = max_components_level2 - len(top_predefined_variations)
        if remaining_slots > 0:
            # Get variations not already included
            remaining_variations = [var for var, count in all_level2_variations.most_common() 
                                   if var.lower() not in [v.lower() for v in top_predefined_variations] 
                                   and count >= min_variation_freq]
            top_variations = top_predefined_variations + remaining_variations[:remaining_slots]
        else:
            top_variations = top_predefined_variations
    else:
        # Original selection logic
        top_families = [fam for fam, count in all_level1_families.most_common(max_components_level1)
                       if count >= min_family_freq]
        top_variations = [var for var, count in all_level2_variations.most_common(max_components_level2)
                         if count >= min_variation_freq]

    # Keywords selection remains the same
    top_keywords = [kw for kw, count in all_keywords.most_common(200)
                   if count >= min_keyword_freq and len(kw) > 2]  # Avoid short words like "of"

    # --- Feature Generation Loop ---
    features_list = []

    for entry in tqdm(all_parsed_tags_info, desc="Generating Features"):
        idx = entry['idx']
        parsed_tags_list = entry['parsed_tags']

        # Basic features
        current_features = {
            'opening_num_tags': len(parsed_tags_list),
            'has_opening_tags': 1 if parsed_tags_list else 0
        }

        # Primary tag features (first tag in the list)
        primary_tag_components = parsed_tags_list[0] if parsed_tags_list else []

        if not primary_tag_components:
            current_features['opening_depth'] = 0
            # Fill with 0 for one-hot encoded features
            for fam in top_families:
                current_features[f'open_fam_{fam}'] = 0
            for var in top_variations:
                current_features[f'open_var_{var}'] = 0
            for kw in top_keywords:
                current_features[f'open_kw_{kw}'] = 0

            # Add strategic features
            current_features.update({
                'opening_is_gambit': 0,
                'opening_is_defense': 0,
                'opening_is_attack': 0,
                'opening_is_system': 0,
                'opening_has_variation': 0
            })

            features_list.append({'idx': idx, **current_features})
            continue

        # 1. Tag depth (complexity metric)
        current_features['opening_depth'] = len(primary_tag_components)

        # 2. Opening family features (standardized)
        family_name = standardize_opening_family(primary_tag_components)
        for fam in top_families:
            current_features[f'open_fam_{fam}'] = 1 if fam == family_name else 0

        # 3. Variation level features
        variation_name = ""
        if len(primary_tag_components) >= 2:
            # Look for defense, attack, variation markers
            variation_types = ['defense', 'attack', 'variation', 'gambit', 'system', 'opening']
            type_indices = [i for i, comp in enumerate(primary_tag_components) if comp.lower() in variation_types]

            if type_indices:
                first_type_idx = type_indices[0]
                if first_type_idx > 0:  # There's a family name before the type
                    variation_name = "_".join(primary_tag_components[:first_type_idx + 1]).lower()
            else:
                # No type designation found, use first 3 components if available
                comp_count = min(3, len(primary_tag_components))
                variation_name = "_".join(primary_tag_components[:comp_count]).lower()

        for var in top_variations:
            current_features[f'open_var_{var}'] = 1 if var == variation_name else 0

        # 4. Keyword features
        tag_keywords_set = {comp.lower() for comp in primary_tag_components}
        for kw in top_keywords:
            current_features[f'open_kw_{kw}'] = 1 if kw in tag_keywords_set else 0

        # 5. Strategic pattern features
        current_features['opening_is_gambit'] = 1 if 'gambit' in tag_keywords_set else 0
        current_features['opening_is_defense'] = 1 if 'defense' in tag_keywords_set else 0
        current_features['opening_is_attack'] = 1 if 'attack' in tag_keywords_set else 0
        current_features['opening_is_system'] = 1 if 'system' in tag_keywords_set else 0
        current_features['opening_has_variation'] = 1 if 'variation' in tag_keywords_set else 0

        features_list.append({'idx': idx, **current_features})

    # Create DataFrame from features list
    openings_df = pd.DataFrame(features_list).set_index('idx')

    # --- Add ECO Code Approximation Features ---
    # Map common opening families to approximate ECO codes
    eco_patterns = {
        'A': ['english', 'reti', 'bird', 'larsen', 'dutch', 'benoni', 'zukertort', 'sokolsky'],
        'B': ['sicilian', 'caro-kann', 'pirc', 'modern', 'alekhine', 'scandinavian'],
        'C': ['ruy_lopez', 'italian', 'french', 'petrov', 'petroff', 'kings_gambit', 'vienna', 'scotch'],
        'D': ['queens_gambit', 'slav', 'grunfeld', 'tarrasch', 'semi-slav'],
        'E': ['indian', 'nimzo', 'queens_indian', 'kings_indian', 'catalan', 'bogo']
    }

    # Create ECO code features
    for eco, patterns in eco_patterns.items():
        pattern_cols = [col for col in openings_df.columns if col.startswith('open_fam_') and
                       any(pattern in col.lower() for pattern in patterns)]

        if pattern_cols:
            openings_df[f'eco_{eco}'] = openings_df[pattern_cols].max(axis=1)

    # --- Add Strategic Characteristic Features ---
    strategic_patterns = {
        'aggressive': ['sicilian', 'kings_gambit', 'dragon', 'najdorf', 'alekhine', 'benoni', 'smith-morra'],
        'positional': ['caro-kann', 'queens_gambit', 'french', 'ruy_lopez', 'slav', 'london', 'catalan'],
        'hypermodern': ['reti', 'alekhine', 'modern', 'pirc', 'kings_indian', 'catalan', 'larsen'],
        'classical': ['ruy_lopez', 'italian', 'vienna', 'scotch', 'queens_gambit', 'giuoco_piano'],
        'solid': ['caro-kann', 'french', 'slav', 'berlin', 'petroff', 'petrovs', 'queens_gambit_declined'],
        'theoretical': ['sicilian_najdorf', 'sicilian_dragon', 'ruy_lopez_marshall', 'kings_indian', 'grunfeld']
    }

    # Create strategic features
    for style, patterns in strategic_patterns.items():
        pattern_cols = [col for col in openings_df.columns if any(pattern in col.lower() for pattern in patterns)]
        if pattern_cols:
            openings_df[f'style_{style}'] = openings_df[pattern_cols].max(axis=1)

    # --- Add First Move Pattern Features ---
    first_move_patterns = {
        'e4_e5': ['kings_pawn', 'kings_gambit', 'vienna', 'italian', 'ruy_lopez', 'scotch', 'petrov', 'petroff'],
        'e4_c5': ['sicilian'],
        'e4_e6': ['french'],
        'e4_c6': ['caro-kann'],
        'e4_d6': ['pirc', 'modern', 'philidor'],
        'e4_nf6': ['alekhine', 'petrov', 'petroff'],
        'd4_d5': ['queens_gambit', 'slav', 'semi-slav', 'tarrasch'],
        'd4_nf6': ['indian', 'nimzo', 'kings_indian', 'grunfeld', 'benoni', 'budapest'],
        'c4_e5': ['english_reversed_sicilian'],
        'c4_c5': ['english_symmetrical'],
        'nf3_d5': ['reti', 'zukertort'],
        'nf3_nf6': ['reti_symmetrical']
    }

    # Create first move features
    for move_seq, patterns in first_move_patterns.items():
        pattern_cols = [col for col in openings_df.columns if any(pattern in col.lower() for pattern in patterns)]
        if pattern_cols:
            openings_df[f'move_{move_seq}'] = openings_df[pattern_cols].max(axis=1)

    # --- TF-IDF and SVD Features ---
    # Only proceed if we have enough puzzles with opening tags
    has_tags_mask = openings_df['has_opening_tags'] == 1
    if has_tags_mask.sum() > 100 and n_svd_components_family > 0:
        try:
            # Process opening families for SVD
            family_texts = []
            for entry in all_parsed_tags_info:
                parsed_tags_list = entry['parsed_tags']
                if parsed_tags_list:
                    family_name = standardize_opening_family(parsed_tags_list[0])
                    family_texts.append(family_name)
                else:
                    family_texts.append("")

            # TF-IDF on opening families
            family_vectorizer = TfidfVectorizer(min_df=5, max_features=500)
            family_matrix = family_vectorizer.fit_transform(family_texts)

            n_comp_fam = min(n_svd_components_family, family_matrix.shape[1] - 1, family_matrix.shape[0] - 1)
            if n_comp_fam > 1:
                svd_family = TruncatedSVD(n_components=n_comp_fam, random_state=42)
                family_svd_features = svd_family.fit_transform(family_matrix)

                family_svd_df = pd.DataFrame(
                    family_svd_features,
                    columns=[f'open_fam_svd_{i}' for i in range(n_comp_fam)],
                    index=openings_df.index
                )
                openings_df = pd.concat([openings_df, family_svd_df], axis=1)
                print(f"Added {n_comp_fam} SVD features for opening families")

            # TF-IDF on full opening tags
            if n_svd_components_full > 0:
                # Process tags for better semantic representation
                processed_tags = [tag.replace('_', ' ') for tag in all_full_tags_for_svd]

                full_vectorizer = TfidfVectorizer(min_df=3, max_features=1000)
                full_matrix = full_vectorizer.fit_transform(processed_tags)

                n_comp_full = min(n_svd_components_full, full_matrix.shape[1] - 1, full_matrix.shape[0] - 1)
                if n_comp_full > 1:
                    svd_full = TruncatedSVD(n_components=n_comp_full, random_state=42)
                    full_svd_features = svd_full.fit_transform(full_matrix)

                    full_svd_df = pd.DataFrame(
                        full_svd_features,
                        columns=[f'open_full_svd_{i}' for i in range(n_comp_full)],
                        index=openings_df.index
                    )
                    openings_df = pd.concat([openings_df, full_svd_df], axis=1)
                    print(f"Added {n_comp_full} SVD features for full opening tags")
        except Exception as e:
            print(f"Error in SVD processing: {str(e)}")

    # --- Hashing Features for High-Cardinality ---
    if n_hash_features > 0:
        try:
            # Process tags for hashing
            processed_tags = [tag.replace('_', ' ') if tag else '' for tag in all_full_tags_for_svd]

            hasher = HashingVectorizer(n_features=n_hash_features, alternate_sign=False)
            hash_features = hasher.fit_transform(processed_tags).toarray()

            hash_df = pd.DataFrame(
                hash_features,
                columns=[f'open_hash_{i}' for i in range(n_hash_features)],
                index=openings_df.index
            )
            openings_df = pd.concat([openings_df, hash_df], axis=1)
            print(f"Added {n_hash_features} hashing features for opening tags")
        except Exception as e:
            print(f"Error in hashing: {str(e)}")

    # --- Handle the imbalance between puzzles with and without tags ---
    # Create features to interact with other puzzle attributes
    if additional_columns is not None and len(additional_columns) > 0:
        for col in additional_columns:
            if col in df_copy.columns:
                # Interaction features: separate signal from puzzles with/without tags
                openings_df[f'no_tag_{col}'] = (1 - openings_df['has_opening_tags']) * df_copy[col]
                openings_df[f'with_tag_{col}'] = openings_df['has_opening_tags'] * df_copy[col]

    # --- Topic Modeling with LDA ---
    # Only proceed if we have enough puzzles with opening tags
    if has_tags_mask.sum() > 100 and len(top_keywords) > 10:
        try:
            # Create document-term matrix for keywords
            keyword_docs = []
            for entry in all_parsed_tags_info:
                parsed_tags_list = entry['parsed_tags']
                if parsed_tags_list:
                    # Join all keywords from all tags into one document
                    keywords = [comp.lower() for tag_comps in parsed_tags_list for comp in tag_comps]
                    keyword_docs.append(' '.join(keywords))
                else:
                    keyword_docs.append('')

            # Apply CountVectorizer to create document-term matrix
            count_vec = CountVectorizer(vocabulary=top_keywords)
            dtm = count_vec.fit_transform(keyword_docs)

            # Apply LDA
            lda = LatentDirichletAllocation(n_components=8, random_state=42)
            topic_results = lda.fit_transform(dtm)

            # Add topic features
            topic_df = pd.DataFrame(
                topic_results,
                columns=[f'open_topic_{i}' for i in range(topic_results.shape[1])],
                index=openings_df.index
            )
            openings_df = pd.concat([openings_df, topic_df], axis=1)

            # Add dominant topic
            openings_df['dominant_topic'] = topic_results.argmax(axis=1)

            print(f"Added {topic_results.shape[1]} LDA topic features")
        except Exception as e:
            print(f"Error in LDA topic modeling: {str(e)}")

    # --- Pawn Structure Patterns ---
    pawn_structures = {
        'central_pawns': ['queens_gambit', 'kings_pawn', 'italian', 'spanish', 'vienna'],
        'fianchetto': ['kings_indian', 'catalan', 'modern', 'pirc', 'reti', 'sicilian_dragon'],
        'isolated_queens_pawn': ['tarrasch', 'queens_gambit_accepted'],
        'hanging_pawns': ['tarrasch', 'queens_gambit_declined'],
        'pawn_chain': ['french', 'kings_indian'],
    }

    for structure, patterns in pawn_structures.items():
        pattern_cols = [col for col in openings_df.columns if
                       any(any(pattern in part for part in col.lower().split('_'))
                          for pattern in patterns)]
        if pattern_cols:
            openings_df[f'structure_{structure}'] = openings_df[pattern_cols].max(axis=1)

    # --- Final cleaning and normalization ---
    # Fill NaNs that might have been introduced
    openings_df = openings_df.fillna(0)

    # Normalize embedding features if desired
    svd_cols = [col for col in openings_df.columns if ('svd' in col or 'hash' in col)]
    if svd_cols:
        openings_df[svd_cols] = (openings_df[svd_cols] - openings_df[svd_cols].mean()) / openings_df[svd_cols].std()
        openings_df[svd_cols] = openings_df[svd_cols].fillna(0)  # Handle any NaNs from division by zero

    print(f"Engineered {openings_df.shape[1]} chess opening features")
    return openings_df
