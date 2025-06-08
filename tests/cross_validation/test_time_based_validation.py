import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from chess_puzzle_rating.models.stacking import LightGBMModel

class TestTimeBasedValidation:
    """Tests for time-based validation splits."""
    
    def test_time_based_split(self, sample_puzzle_df_with_timestamps):
        """Test creating a time-based train-test split."""
        # Sort the data by timestamp
        sorted_df = sample_puzzle_df_with_timestamps.sort_values('Timestamp')
        
        # Split the data into train and test sets based on time
        # Use the first 80% of the data for training and the last 20% for testing
        train_size = int(0.8 * len(sorted_df))
        train_df = sorted_df.iloc[:train_size]
        test_df = sorted_df.iloc[train_size:]
        
        # Check that the split is correct
        assert len(train_df) == train_size
        assert len(test_df) == len(sorted_df) - train_size
        
        # Check that the timestamps are correctly split
        assert train_df['Timestamp'].max() <= test_df['Timestamp'].min(), \
            "Train set contains timestamps that are later than the earliest timestamp in the test set"
    
    def test_time_based_cross_validation(self, sample_puzzle_df_with_timestamps):
        """Test time-based cross-validation."""
        # Sort the data by timestamp
        sorted_df = sample_puzzle_df_with_timestamps.sort_values('Timestamp')
        
        # Create a function for time-based cross-validation
        def time_based_cv(df, n_splits):
            """Generate time-based cross-validation splits."""
            # Calculate the size of each fold
            fold_size = len(df) // n_splits
            
            for i in range(n_splits - 1):
                # Use all data up to the current fold as training
                train_end = (i + 1) * fold_size
                train_indices = df.index[:train_end]
                
                # Use the next fold as validation
                val_start = train_end
                val_end = val_start + fold_size
                val_indices = df.index[val_start:val_end]
                
                yield train_indices, val_indices
        
        # Perform time-based cross-validation
        n_splits = 3
        for train_indices, val_indices in time_based_cv(sorted_df, n_splits):
            train_df = sorted_df.loc[train_indices]
            val_df = sorted_df.loc[val_indices]
            
            # Check that the validation set comes after the training set
            assert train_df['Timestamp'].max() <= val_df['Timestamp'].min(), \
                "Train set contains timestamps that are later than the earliest timestamp in the validation set"
    
    def test_time_based_model_evaluation(self, sample_puzzle_df_with_timestamps):
        """Test evaluating a model using time-based validation."""
        # Sort the data by timestamp
        sorted_df = sample_puzzle_df_with_timestamps.sort_values('Timestamp')
        
        # Split the data into train and test sets based on time
        train_size = int(0.8 * len(sorted_df))
        train_df = sorted_df.iloc[:train_size]
        test_df = sorted_df.iloc[train_size:]
        
        # Create dummy features for testing
        X_train = pd.DataFrame({
            'feature1': np.random.rand(len(train_df)),
            'feature2': np.random.rand(len(train_df))
        })
        y_train = train_df['Rating']
        
        X_test = pd.DataFrame({
            'feature1': np.random.rand(len(test_df)),
            'feature2': np.random.rand(len(test_df))
        })
        y_test = test_df['Rating']
        
        # Train a model
        model = LightGBMModel(name="test_model", model_params={"n_estimators": 10})
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        predictions = model.predict(X_test)
        
        # Calculate the mean squared error
        mse = np.mean((predictions - y_test) ** 2)
        
        # Check that the MSE is reasonable (this is a dummy test, so we just check that it's not NaN or infinite)
        assert np.isfinite(mse), f"MSE is not finite: {mse}"
    
    def test_combined_time_and_rating_based_validation(self, sample_puzzle_df_with_timestamps, sample_rating_ranges):
        """Test combining time-based and rating-based validation."""
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
        
        # Sort the data by timestamp
        sorted_df = sample_puzzle_df_with_timestamps.sort_values('Timestamp')
        
        # Split the data into train and test sets based on time
        train_size = int(0.8 * len(sorted_df))
        train_df = sorted_df.iloc[:train_size]
        test_df = sorted_df.iloc[train_size:]
        
        # Check the distribution of rating ranges in the train and test sets
        train_distribution = train_df['RatingRange'].value_counts(normalize=True)
        test_distribution = test_df['RatingRange'].value_counts(normalize=True)
        
        # Print the distributions for debugging
        print("Train distribution:", train_distribution)
        print("Test distribution:", test_distribution)
        
        # Create dummy features for testing
        X_train = pd.DataFrame({
            'feature1': np.random.rand(len(train_df)),
            'feature2': np.random.rand(len(train_df))
        })
        y_train = train_df['Rating']
        
        X_test = pd.DataFrame({
            'feature1': np.random.rand(len(test_df)),
            'feature2': np.random.rand(len(test_df))
        })
        y_test = test_df['Rating']
        
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
        
        # Check that the metrics are reasonable (this is a dummy test, so we just check that they're not NaN or infinite)
        for label, mse in range_metrics.items():
            assert np.isfinite(mse), f"MSE for range {label} is not finite: {mse}"