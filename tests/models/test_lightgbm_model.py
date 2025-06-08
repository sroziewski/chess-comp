import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from chess_puzzle_rating.models.stacking import LightGBMModel

class TestLightGBMModel:
    """Tests for the LightGBMModel class."""
    
    def test_init(self):
        """Test initializing a LightGBMModel."""
        model = LightGBMModel(name="test_lightgbm")
        assert model.name == "test_lightgbm"
        assert model.model_params is not None
        assert "objective" in model.model_params
        assert model.model_params["objective"] == "regression"
        
        # Test with custom parameters
        model_params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "n_estimators": 100
        }
        model = LightGBMModel(name="test_lightgbm", model_params=model_params)
        assert model.name == "test_lightgbm"
        assert model.model_params == model_params
    
    def test_fit_predict(self):
        """Test fitting and predicting with a LightGBMModel."""
        model = LightGBMModel(name="test_lightgbm")
        
        # Create dummy data
        X = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100)
        })
        y = pd.Series(np.random.rand(100) * 100)
        
        # Fit the model
        model.fit(X, y)
        
        # Check that the model has been fitted
        assert model.model is not None
        
        # Make predictions
        predictions = model.predict(X)
        
        # Check that predictions have the right shape
        assert len(predictions) == len(X)
        
        # Check that predictions are reasonable (within the range of y)
        assert predictions.min() >= 0
        assert predictions.max() <= 100
    
    def test_fit_with_eval_set(self):
        """Test fitting with an evaluation set."""
        model = LightGBMModel(name="test_lightgbm")
        
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
        
        # Fit the model with an evaluation set
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        # Check that the model has been fitted
        assert model.model is not None
        
        # Make predictions
        predictions = model.predict(X_train)
        
        # Check that predictions have the right shape
        assert len(predictions) == len(X_train)
    
    def test_save_load(self, tmp_path):
        """Test saving and loading a model."""
        model = LightGBMModel(name="test_lightgbm")
        
        # Create dummy data
        X = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100)
        })
        y = pd.Series(np.random.rand(100) * 100)
        
        # Fit the model
        model.fit(X, y)
        
        # Create a temporary file path
        model_path = os.path.join(tmp_path, "test_lightgbm.pkl")
        
        # Save the model
        model.save(model_path)
        
        # Check that the file exists
        assert os.path.exists(model_path)
        
        # Create a new model
        new_model = LightGBMModel(name="test_lightgbm")
        
        # Load the model
        new_model.load(model_path)
        
        # Check that the model has been loaded
        assert new_model.model is not None
        
        # Make predictions with the loaded model
        new_predictions = new_model.predict(X)
        
        # Check that predictions have the right shape
        assert len(new_predictions) == len(X)
        
        # Check that predictions are the same as the original model
        original_predictions = model.predict(X)
        np.testing.assert_array_almost_equal(new_predictions, original_predictions)
    
    def test_get_feature_importances(self):
        """Test getting feature importances."""
        model = LightGBMModel(name="test_lightgbm")
        
        # Create dummy data
        X = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100)
        })
        y = pd.Series(np.random.rand(100) * 100)
        
        # Fit the model
        model.fit(X, y)
        
        # Get feature importances
        importances = model.get_feature_importances()
        
        # Check that importances is a dictionary
        assert isinstance(importances, dict)
        
        # Check that importances has the right keys
        assert set(importances.keys()) == set(X.columns)
        
        # Check that importances values are non-negative
        for importance in importances.values():
            assert importance >= 0