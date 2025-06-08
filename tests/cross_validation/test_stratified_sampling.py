import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from chess_puzzle_rating.models.stacking import LightGBMModel

class TestStratifiedSamplingByRatingRange:
    """Tests for stratified sampling by rating range."""
    
    def test_stratified_sampling(self, sample_puzzle_df_with_timestamps, sample_rating_ranges):
        """Test stratified sampling by rating range."""
        # Define rating ranges
        rating_ranges = sample_rating_ranges
        
        # Create a function to assign a rating range label to each rating
        def get_rating_range_label(rating):
            for min_rating, max_rating, label in rating_ranges:
                if min_rating <= rating <= max_rating:
                    return label
            return "Unknown"
        
        # Assign rating range labels to the sample data
        sample_puzzle_df_with_timestamps['RatingRange'] = sample_puzzle_df_with_timestamps['Rating'].apply(get_rating_range_label)
        
        # Check that each rating has been assigned a range
        assert not sample_puzzle_df_with_timestamps['RatingRange'].eq("Unknown").any(), "Some ratings were not assigned a range"
        
        # Perform stratified sampling
        X = sample_puzzle_df_with_timestamps.drop(['Rating', 'RatingRange'], axis=1)
        y = sample_puzzle_df_with_timestamps['Rating']
        stratify = sample_puzzle_df_with_timestamps['RatingRange']
        
        X_train, X_test, y_train, y_test, stratify_train, stratify_test = train_test_split(
            X, y, stratify, test_size=0.2, random_state=42
        )
        
        # Check that the stratification has been preserved
        train_distribution = stratify_train.value_counts(normalize=True)
        test_distribution = stratify_test.value_counts(normalize=True)
        
        # Check that the distributions are similar (within a reasonable tolerance)
        for label in train_distribution.index:
            assert abs(train_distribution[label] - test_distribution[label]) < 0.2, \
                f"Distribution mismatch for {label}: train={train_distribution[label]}, test={test_distribution[label]}"
    
    def test_stratified_cross_validation(self, sample_puzzle_df_with_timestamps, sample_rating_ranges):
        """Test stratified cross-validation by rating range."""
        # Define rating ranges
        rating_ranges = sample_rating_ranges
        
        # Create a function to assign a rating range label to each rating
        def get_rating_range_label(rating):
            for min_rating, max_rating, label in rating_ranges:
                if min_rating <= rating <= max_rating:
                    return label
            return "Unknown"
        
        # Assign rating range labels to the sample data
        sample_puzzle_df_with_timestamps['RatingRange'] = sample_puzzle_df_with_timestamps['Rating'].apply(get_rating_range_label)
        
        # Check that each rating has been assigned a range
        assert not sample_puzzle_df_with_timestamps['RatingRange'].eq("Unknown").any(), "Some ratings were not assigned a range"
        
        # Prepare data for cross-validation
        X = sample_puzzle_df_with_timestamps.drop(['Rating', 'RatingRange'], axis=1)
        y = sample_puzzle_df_with_timestamps['Rating']
        stratify = sample_puzzle_df_with_timestamps['RatingRange']
        
        # Create a stratified k-fold cross-validator
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        
        # Perform stratified cross-validation
        for train_index, test_index in skf.split(X, stratify):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            stratify_train, stratify_test = stratify.iloc[train_index], stratify.iloc[test_index]
            
            # Check that the stratification has been preserved
            train_distribution = stratify_train.value_counts(normalize=True)
            test_distribution = stratify_test.value_counts(normalize=True)
            
            # Check that the distributions are similar (within a reasonable tolerance)
            for label in train_distribution.index:
                assert abs(train_distribution[label] - test_distribution[label]) < 0.2, \
                    f"Distribution mismatch for {label}: train={train_distribution[label]}, test={test_distribution[label]}"
    
    def test_rating_range_specific_metrics(self, sample_puzzle_df_with_timestamps, sample_rating_ranges):
        """Test calculating metrics for specific rating ranges."""
        # Define rating ranges
        rating_ranges = sample_rating_ranges
        
        # Create a function to assign a rating range label to each rating
        def get_rating_range_label(rating):
            for min_rating, max_rating, label in rating_ranges:
                if min_rating <= rating <= max_rating:
                    return label
            return "Unknown"
        
        # Assign rating range labels to the sample data
        sample_puzzle_df_with_timestamps['RatingRange'] = sample_puzzle_df_with_timestamps['Rating'].apply(get_rating_range_label)
        
        # Create dummy features for testing
        X = pd.DataFrame({
            'feature1': np.random.rand(len(sample_puzzle_df_with_timestamps)),
            'feature2': np.random.rand(len(sample_puzzle_df_with_timestamps))
        })
        y = sample_puzzle_df_with_timestamps['Rating']
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a model
        model = LightGBMModel(name="test_model", model_params={"n_estimators": 10})
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        predictions = model.predict(X_test)
        
        # Calculate metrics for each rating range
        range_metrics = {}
        
        for min_rating, max_rating, label in rating_ranges:
            # Filter test data for this rating range
            range_mask = (y_test >= min_rating) & (y_test <= max_rating)
            if range_mask.sum() > 0:  # Only calculate metrics if there are samples in this range
                range_y_test = y_test[range_mask]
                range_predictions = predictions[range_mask]
                
                # Calculate mean squared error for this range
                range_mse = np.mean((range_predictions - range_y_test) ** 2)
                range_metrics[label] = range_mse
        
        # Check that metrics have been calculated for each range
        for _, _, label in rating_ranges:
            assert label in range_metrics or sum((y_test >= min_rating) & (y_test <= max_rating)) == 0, \
                f"No metrics calculated for range {label}"
        
        # Check that the metrics are reasonable (this is a dummy test, so we just check that they're not NaN or infinite)
        for label, mse in range_metrics.items():
            assert np.isfinite(mse), f"MSE for range {label} is not finite: {mse}"