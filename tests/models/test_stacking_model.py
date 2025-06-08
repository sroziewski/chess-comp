import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from chess_puzzle_rating.models.stacking import (
    BaseModel, LightGBMModel, XGBoostModel, NeuralNetworkModel, StackingModel
)

class TestStackingModel:
    """Tests for the StackingModel class."""
    
    def test_init(self):
        """Test initializing a StackingModel."""
        # Create base models
        base_models = [
            LightGBMModel(name="lightgbm1"),
            XGBoostModel(name="xgboost1")
        ]
        
        # Create a stacking model
        model = StackingModel(base_models=base_models)
        
        # Check that the model has the right attributes
        assert model.base_models == base_models
        assert model.meta_learner is None  # Default is None, created during fit
        assert model.n_splits == 5  # Default value
        assert model.random_state == 42  # Default value
        assert model.use_features_in_meta is True  # Default value
        
        # Test with custom parameters
        meta_learner = LightGBMModel(name="meta_learner")
        model = StackingModel(
            base_models=base_models,
            meta_learner=meta_learner,
            n_splits=3,
            random_state=123,
            use_features_in_meta=False
        )
        
        # Check that the model has the right attributes
        assert model.base_models == base_models
        assert model.meta_learner == meta_learner
        assert model.n_splits == 3
        assert model.random_state == 123
        assert model.use_features_in_meta is False
    
    def test_fit_predict(self):
        """Test fitting and predicting with a StackingModel."""
        # Create base models
        base_models = [
            LightGBMModel(name="lightgbm1", model_params={"n_estimators": 10}),
            XGBoostModel(name="xgboost1", model_params={"n_estimators": 10})
        ]
        
        # Create a stacking model with a small number of splits for testing
        model = StackingModel(base_models=base_models, n_splits=2)
        
        # Create dummy data
        X = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100)
        })
        y = pd.Series(np.random.rand(100) * 100)
        
        # Fit the model
        model.fit(X, y)
        
        # Check that the model has been fitted
        assert model.meta_learner is not None
        assert all(hasattr(base_model, 'model') for base_model in model.base_models)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Check that predictions have the right shape
        assert len(predictions) == len(X)
        
        # Check that predictions are reasonable (within the range of y)
        assert predictions.min() >= 0
        assert predictions.max() <= 100
    
    def test_fit_with_validation_data(self):
        """Test fitting with validation data."""
        # Create base models
        base_models = [
            LightGBMModel(name="lightgbm1", model_params={"n_estimators": 10}),
            XGBoostModel(name="xgboost1", model_params={"n_estimators": 10})
        ]
        
        # Create a stacking model with a small number of splits for testing
        model = StackingModel(base_models=base_models, n_splits=2)
        
        # Create dummy data
        X_train = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100)
        })
        y_train = pd.Series(np.random.rand(100) * 100)
        
        X_val = pd.DataFrame({
            "feature1": np.random.rand(20),
            "feature2": np.random.rand(20)
        })
        y_val = pd.Series(np.random.rand(20) * 100)
        
        # Fit the model with validation data
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        # Check that the model has been fitted
        assert model.meta_learner is not None
        assert all(hasattr(base_model, 'model') for base_model in model.base_models)
        
        # Make predictions
        predictions = model.predict(X_train)
        
        # Check that predictions have the right shape
        assert len(predictions) == len(X_train)
    
    def test_save_load(self, tmp_path):
        """Test saving and loading a model."""
        # Create base models
        base_models = [
            LightGBMModel(name="lightgbm1", model_params={"n_estimators": 10}),
            XGBoostModel(name="xgboost1", model_params={"n_estimators": 10})
        ]
        
        # Create a stacking model with a small number of splits for testing
        model = StackingModel(base_models=base_models, n_splits=2)
        
        # Create dummy data
        X = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100)
        })
        y = pd.Series(np.random.rand(100) * 100)
        
        # Fit the model
        model.fit(X, y)
        
        # Create a temporary directory path
        model_dir = os.path.join(tmp_path, "stacking_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model.save(model_dir)
        
        # Check that the directory exists
        assert os.path.exists(model_dir)
        
        # Check that the base models and meta-learner have been saved
        assert os.path.exists(os.path.join(model_dir, "meta_learner.pkl"))
        assert os.path.exists(os.path.join(model_dir, "base_models"))
        assert os.path.exists(os.path.join(model_dir, "base_models", "lightgbm1.pkl"))
        assert os.path.exists(os.path.join(model_dir, "base_models", "xgboost1.pkl"))
        
        # Create a new model
        new_model = StackingModel(base_models=[
            LightGBMModel(name="lightgbm1"),
            XGBoostModel(name="xgboost1")
        ])
        
        # Load the model
        new_model.load(model_dir)
        
        # Check that the model has been loaded
        assert new_model.meta_learner is not None
        assert all(hasattr(base_model, 'model') for base_model in new_model.base_models)
        
        # Make predictions with the loaded model
        new_predictions = new_model.predict(X)
        
        # Check that predictions have the right shape
        assert len(new_predictions) == len(X)
        
        # Check that predictions are the same as the original model
        original_predictions = model.predict(X)
        np.testing.assert_array_almost_equal(new_predictions, original_predictions)