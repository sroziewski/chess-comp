import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from chess_puzzle_rating.models.stacking import (
    LightGBMModel, XGBoostModel, NeuralNetworkModel, StackingModel, RangeSpecificStackingModel
)

class TestModelTrainingWorkflow:
    """Tests for the model training workflow."""
    
    def test_single_model_workflow(self):
        """Test training and evaluating a single model."""
        # Create dummy data
        np.random.seed(42)  # For reproducibility
        n_samples = 100
        n_features = 10
        
        X = pd.DataFrame(
            np.random.rand(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(1000, 2000, n_samples))
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a LightGBM model
        model = LightGBMModel(name="test_lightgbm", model_params={"n_estimators": 10})
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        predictions = model.predict(X_test)
        
        # Calculate the mean squared error
        mse = np.mean((predictions - y_test) ** 2)
        
        # Check that the MSE is reasonable (this is a dummy test, so we just check that it's not NaN or infinite)
        assert np.isfinite(mse), f"MSE is not finite: {mse}"
        
        # Check that the predictions have the right shape
        assert len(predictions) == len(X_test)
        
        # Check that the predictions are reasonable (within the range of y)
        assert predictions.min() >= 1000, f"Prediction below minimum: {predictions.min()}"
        assert predictions.max() <= 2000, f"Prediction above maximum: {predictions.max()}"
    
    def test_stacking_model_workflow(self):
        """Test training and evaluating a stacking model."""
        # Create dummy data
        np.random.seed(42)  # For reproducibility
        n_samples = 100
        n_features = 10
        
        X = pd.DataFrame(
            np.random.rand(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(1000, 2000, n_samples))
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create base models
        base_models = [
            LightGBMModel(name="lightgbm1", model_params={"n_estimators": 10}),
            XGBoostModel(name="xgboost1", model_params={"n_estimators": 10})
        ]
        
        # Create a stacking model with a small number of splits for testing
        model = StackingModel(base_models=base_models, n_splits=2)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        predictions = model.predict(X_test)
        
        # Calculate the mean squared error
        mse = np.mean((predictions - y_test) ** 2)
        
        # Check that the MSE is reasonable (this is a dummy test, so we just check that it's not NaN or infinite)
        assert np.isfinite(mse), f"MSE is not finite: {mse}"
        
        # Check that the predictions have the right shape
        assert len(predictions) == len(X_test)
        
        # Check that the predictions are reasonable (within the range of y)
        assert predictions.min() >= 1000, f"Prediction below minimum: {predictions.min()}"
        assert predictions.max() <= 2000, f"Prediction above maximum: {predictions.max()}"
    
    def test_range_specific_model_workflow(self):
        """Test training and evaluating a range-specific stacking model."""
        # Create dummy data
        np.random.seed(42)  # For reproducibility
        n_samples = 300  # 100 samples per range
        n_features = 10
        
        X = pd.DataFrame(
            np.random.rand(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create ratings in different ranges
        beginner_ratings = np.random.randint(1000, 1500, 100)
        intermediate_ratings = np.random.randint(1501, 2000, 100)
        advanced_ratings = np.random.randint(2001, 2500, 100)
        
        y = pd.Series(np.concatenate([beginner_ratings, intermediate_ratings, advanced_ratings]))
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=[999, 1500, 2000, 2501])
        )
        
        # Define rating ranges
        rating_ranges = [
            [1000, 1500, "Beginner"],
            [1501, 2000, "Intermediate"],
            [2001, 2500, "Advanced"]
        ]
        
        # Create a range-specific stacking model with a small number of splits for testing
        model = RangeSpecificStackingModel(
            rating_ranges=rating_ranges,
            n_splits=2,
            optimize_meta=False  # Disable optimization for faster testing
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        predictions = model.predict(X_test)
        
        # Calculate the mean squared error
        mse = np.mean((predictions - y_test) ** 2)
        
        # Check that the MSE is reasonable (this is a dummy test, so we just check that it's not NaN or infinite)
        assert np.isfinite(mse), f"MSE is not finite: {mse}"
        
        # Check that the predictions have the right shape
        assert len(predictions) == len(X_test)
        
        # Check that the predictions are reasonable (within the range of y)
        assert predictions.min() >= 1000, f"Prediction below minimum: {predictions.min()}"
        assert predictions.max() <= 2500, f"Prediction above maximum: {predictions.max()}"
    
    def test_cross_validation_workflow(self):
        """Test a cross-validation workflow."""
        # Create dummy data
        np.random.seed(42)  # For reproducibility
        n_samples = 100
        n_features = 10
        
        X = pd.DataFrame(
            np.random.rand(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(1000, 2000, n_samples))
        
        # Perform cross-validation
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        mse_scores = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train a LightGBM model
            model = LightGBMModel(name="test_lightgbm", model_params={"n_estimators": 10})
            model.fit(X_train, y_train)
            
            # Make predictions on the test set
            predictions = model.predict(X_test)
            
            # Calculate the mean squared error
            mse = np.mean((predictions - y_test) ** 2)
            mse_scores.append(mse)
        
        # Calculate the mean and standard deviation of the MSE scores
        mean_mse = np.mean(mse_scores)
        std_mse = np.std(mse_scores)
        
        # Check that the mean MSE is reasonable (this is a dummy test, so we just check that it's not NaN or infinite)
        assert np.isfinite(mean_mse), f"Mean MSE is not finite: {mean_mse}"
        
        # Check that the standard deviation of MSE is reasonable
        assert np.isfinite(std_mse), f"Standard deviation of MSE is not finite: {std_mse}"