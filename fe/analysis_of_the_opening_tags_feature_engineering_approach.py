def enhanced_parse_opening_tag(tag_string):
    """Enhanced parsing with chess-specific domain knowledge"""
    # Existing parsing logic
    individual_tags, parsed_tags_components = parse_opening_tag(tag_string)

    # Enhanced family identification
    for i, components in enumerate(parsed_tags_components):
        # Handle special cases with known family patterns
        if len(components) >= 2:
            # Common prefixes that should be kept with the following word
            prefixes = ['king', 'queen', 'ruy', 'sicilian', 'french', 'nimzo', 'bogo', 'semi']
            if components[0] in prefixes and len(components) > 1:
                # Merge prefix with next component to form proper family name
                parsed_tags_components[i] = [f"{components[0]}_{components[1]}"] + components[2:]

    return individual_tags, parsed_tags_components


def add_eco_code_features(features_df):
    """Add approximate ECO code features based on opening patterns"""
    eco_patterns = {
        'A': ['english', 'reti', 'bird', 'larsen', 'dutch', 'benoni'],
        'B': ['sicilian', 'caro-kann', 'pirc', 'modern', 'alekhine'],
        'C': ['ruy_lopez', 'italian', 'french', 'petroff', 'kings_gambit'],
        'D': ['queens_gambit', 'slav', 'grunfeld', 'tarrasch'],
        'E': ['indian', 'nimzo', 'queens_indian', 'kings_indian', 'catalan']
    }

    for eco, patterns in eco_patterns.items():
        features_df[f'eco_{eco}'] = features_df.apply(
            lambda row: 1 if any(pattern in col and row[col] == 1
                                 for pattern in patterns
                                 for col in row.index if col.startswith('open_fam_')),
            axis=1
        )

    return features_df


def add_strategic_features(features_df):
    """Add features capturing strategic characteristics of openings"""
    strategic_patterns = {
        'aggressive': ['sicilian', 'kings_gambit', 'dragon', 'najdorf', 'alekhine', 'benoni'],
        'positional': ['caro-kann', 'queens_gambit', 'french', 'ruy_lopez', 'slav'],
        'hypermodern': ['reti', 'alekhine', 'modern', 'pirc', 'kings_indian'],
        'classical': ['ruy_lopez', 'italian', 'vienna', 'scotch', 'queens_gambit']
    }

    for style, patterns in strategic_patterns.items():
        features_df[f'style_{style}'] = features_df.apply(
            lambda row: 1 if any(pattern in col and row[col] == 1
                                 for pattern in patterns
                                 for col in row.index if col.startswith('open_')),
            axis=1
        )

    return features_df


from sklearn.feature_extraction.text import HashingVectorizer


def add_hashing_features(tag_strings, n_features=20):
    """Use HashingVectorizer for memory-efficient representation of many openings"""
    # Process tags to replace underscores with spaces for better tokenization
    processed_tags = [tag.replace('_', ' ') if not pd.isna(tag) else '' for tag in tag_strings]

    # Use HashingVectorizer for efficient representation
    hasher = HashingVectorizer(n_features=n_features, alternate_sign=False, norm=None)
    hash_features = hasher.fit_transform(processed_tags).toarray()

    hash_df = pd.DataFrame(
        hash_features,
        columns=[f'open_hash_{i}' for i in range(n_features)],
        index=tag_strings.index
    )

    return hash_df

1. **Handle imbalance between puzzles with and without tags**:

    def create_tag_presence_features(features_df, additional_columns):
        """Create features to handle the imbalance of puzzles with/without tags"""
        # Tag presence indicator
        features_df['has_opening_tags'] = (features_df['opening_depth'] > 0).astype(int)

        # For puzzles without tags, add interaction features with other puzzle attributes
        for col in additional_columns:
            features_df[f'no_tag_{col}'] = (1 - features_df['has_opening_tags']) * features_df[col]
            features_df[f'with_tag_{col}'] = features_df['has_opening_tags'] * features_df[col]

        return features_df

