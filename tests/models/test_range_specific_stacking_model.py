import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from chess_puzzle_rating.models.stacking import (
    BaseModel, LightGBMModel, XGBoostModel, RangeSpecificStackingModel
)

class TestRangeSpecificStackingModel:
    """Tests for the RangeSpecificStackingModel class."""
    
    def test_init(self):
        """Test initializing a RangeSpecificStackingModel."""
        # Define rating ranges
        rating_ranges = [
            [0, 1500, "Beginner"],
            [1501, 2000, "Intermediate"],
            [2001, 3000, "Advanced"]
        ]
        
        # Create a range-specific stacking model
        model = RangeSpecificStackingModel(rating_ranges=rating_ranges)
        
        # Check that the model has the right attributes
        assert model.rating_ranges == rating_ranges
        assert model.range_overlap == 100  # Default value
        assert model.ensemble_method == "weighted_average"  # Default value
        assert model.n_splits == 5  # Default value
        assert model.random_state == 42  # Default value
        assert model.optimize_meta is True  # Default value
        assert model.meta_learner_type == "lightgbm"  # Default value
        assert model.use_features_in_meta is True  # Default value
        
        # Test with custom parameters
        model = RangeSpecificStackingModel(
            rating_ranges=rating_ranges,
            range_overlap=50,
            ensemble_method="simple_average",
            n_splits=3,
            random_state=123,
            optimize_meta=False,
            meta_learner_type="xgboost",
            use_features_in_meta=False
        )
        
        # Check that the model has the right attributes
        assert model.rating_ranges == rating_ranges
        assert model.range_overlap == 50
        assert model.ensemble_method == "simple_average"
        assert model.n_splits == 3
        assert model.random_state == 123
        assert model.optimize_meta is False
        assert model.meta_learner_type == "xgboost"
        assert model.use_features_in_meta is False
    
    def test_fit_predict(self):
        """Test fitting and predicting with a RangeSpecificStackingModel."""
        # Define rating ranges
        rating_ranges = [
            [0, 1500, "Beginner"],
            [1501, 2000, "Intermediate"],
            [2001, 3000, "Advanced"]
        ]
        
        # Create a range-specific stacking model with a small number of splits for testing
        model = RangeSpecificStackingModel(
            rating_ranges=rating_ranges,
            n_splits=2,
            optimize_meta=False  # Disable optimization for faster testing
        )
        
        # Create dummy data with ratings in different ranges
        np.random.seed(42)  # For reproducibility
        n_samples = 300  # 100 samples per range
        
        # Create features
        X = pd.DataFrame({
            "feature1": np.random.rand(n_samples),
            "feature2": np.random.rand(n_samples)
        })
        
        # Create ratings in different ranges
        beginner_ratings = np.random.randint(1000, 1500, 100)
        intermediate_ratings = np.random.randint(1501, 2000, 100)
        advanced_ratings = np.random.randint(2001, 2500, 100)
        
        y = pd.Series(np.concatenate([beginner_ratings, intermediate_ratings, advanced_ratings]))
        
        # Fit the model
        model.fit(X, y)
        
        # Check that the model has been fitted
        assert len(model.range_models) == 3  # One model per range
        
        # Make predictions
        predictions = model.predict(X)
        
        # Check that predictions have the right shape
        assert len(predictions) == len(X)
        
        # Check that predictions are reasonable (within the range of y)
        assert predictions.min() >= 1000
        assert predictions.max() <= 2500
    
    def test_fit_with_validation_data(self):
        """Test fitting with validation data."""
        # Define rating ranges
        rating_ranges = [
            [0, 1500, "Beginner"],
            [1501, 2000, "Intermediate"],
            [2001, 3000, "Advanced"]
        ]
        
        # Create a range-specific stacking model with a small number of splits for testing
        model = RangeSpecificStackingModel(
            rating_ranges=rating_ranges,
            n_splits=2,
            optimize_meta=False  # Disable optimization for faster testing
        )
        
        # Create dummy data with ratings in different ranges
        np.random.seed(42)  # For reproducibility
        n_train_samples = 300  # 100 samples per range
        n_val_samples = 60  # 20 samples per range
        
        # Create training features
        X_train = pd.DataFrame({
            "feature1": np.random.rand(n_train_samples),
            "feature2": np.random.rand(n_train_samples)
        })
        
        # Create training ratings in different ranges
        beginner_ratings_train = np.random.randint(1000, 1500, 100)
        intermediate_ratings_train = np.random.randint(1501, 2000, 100)
        advanced_ratings_train = np.random.randint(2001, 2500, 100)
        
        y_train = pd.Series(np.concatenate([
            beginner_ratings_train,
            intermediate_ratings_train,
            advanced_ratings_train
        ]))
        
        # Create validation features
        X_val = pd.DataFrame({
            "feature1": np.random.rand(n_val_samples),
            "feature2": np.random.rand(n_val_samples)
        })
        
        # Create validation ratings in different ranges
        beginner_ratings_val = np.random.randint(1000, 1500, 20)
        intermediate_ratings_val = np.random.randint(1501, 2000, 20)
        advanced_ratings_val = np.random.randint(2001, 2500, 20)
        
        y_val = pd.Series(np.concatenate([
            beginner_ratings_val,
            intermediate_ratings_val,
            advanced_ratings_val
        ]))
        
        # Fit the model with validation data
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        # Check that the model has been fitted
        assert len(model.range_models) == 3  # One model per range
        
        # Make predictions
        predictions = model.predict(X_train)
        
        # Check that predictions have the right shape
        assert len(predictions) == len(X_train)
    
    def test_save_load(self, tmp_path):
        """Test saving and loading a model."""
        # Define rating ranges
        rating_ranges = [
            [0, 1500, "Beginner"],
            [1501, 2000, "Intermediate"],
            [2001, 3000, "Advanced"]
        ]
        
        # Create a range-specific stacking model with a small number of splits for testing
        model = RangeSpecificStackingModel(
            rating_ranges=rating_ranges,
            n_splits=2,
            optimize_meta=False  # Disable optimization for faster testing
        )
        
        # Create dummy data with ratings in different ranges
        np.random.seed(42)  # For reproducibility
        n_samples = 300  # 100 samples per range
        
        # Create features
        X = pd.DataFrame({
            "feature1": np.random.rand(n_samples),
            "feature2": np.random.rand(n_samples)
        })
        
        # Create ratings in different ranges
        beginner_ratings = np.random.randint(1000, 1500, 100)
        intermediate_ratings = np.random.randint(1501, 2000, 100)
        advanced_ratings = np.random.randint(2001, 2500, 100)
        
        y = pd.Series(np.concatenate([beginner_ratings, intermediate_ratings, advanced_ratings]))
        
        # Fit the model
        model.fit(X, y)
        
        # Create a temporary directory path
        model_dir = os.path.join(tmp_path, "range_specific_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model.save(model_dir)
        
        # Check that the directory exists
        assert os.path.exists(model_dir)
        
        # Check that the range models have been saved
        assert os.path.exists(os.path.join(model_dir, "range_models"))
        assert os.path.exists(os.path.join(model_dir, "range_models", "Beginner"))
        assert os.path.exists(os.path.join(model_dir, "range_models", "Intermediate"))
        assert os.path.exists(os.path.join(model_dir, "range_models", "Advanced"))
        
        # Create a new model
        new_model = RangeSpecificStackingModel(rating_ranges=rating_ranges)
        
        # Load the model
        new_model.load(model_dir)
        
        # Check that the model has been loaded
        assert len(new_model.range_models) == 3  # One model per range
        
        # Make predictions with the loaded model
        new_predictions = new_model.predict(X)
        
        # Check that predictions have the right shape
        assert len(new_predictions) == len(X)
        
        # Check that predictions are the same as the original model
        original_predictions = model.predict(X)
        np.testing.assert_array_almost_equal(new_predictions, original_predictions)