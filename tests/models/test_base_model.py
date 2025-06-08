import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from chess_puzzle_rating.models.stacking import BaseModel

class TestBaseModel:
    """Tests for the BaseModel class."""
    
    def test_init(self):
        """Test initializing a BaseModel."""
        model = BaseModel(name="test_model")
        assert model.name == "test_model"
        assert model.model_params == {}
        
        # Test with custom parameters
        model_params = {"param1": 1, "param2": "value"}
        model = BaseModel(name="test_model", model_params=model_params)
        assert model.name == "test_model"
        assert model.model_params == model_params
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        model = BaseModel(name="test_model")
        
        # Create dummy data
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y = pd.Series([10, 20, 30])
        
        # Test fit method
        with pytest.raises(NotImplementedError):
            model.fit(X, y)
        
        # Test predict method
        with pytest.raises(NotImplementedError):
            model.predict(X)
        
        # Test get_feature_importances method
        with pytest.raises(NotImplementedError):
            model.get_feature_importances()
    
    def test_save_load(self, tmp_path):
        """Test saving and loading a model."""
        model = BaseModel(name="test_model")
        
        # Create a temporary file path
        model_path = os.path.join(tmp_path, "test_model.pkl")
        
        # Test save method
        with pytest.raises(NotImplementedError):
            model.save(model_path)
        
        # Test load method
        with pytest.raises(NotImplementedError):
            model.load(model_path)