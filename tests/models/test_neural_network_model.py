import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import torch
from chess_puzzle_rating.models.stacking import NeuralNetworkModel, EnhancedNeuralNetworkRegressor, ChessRatingLoss

class TestNeuralNetworkModel:
    """Tests for the NeuralNetworkModel class."""

    def test_init(self):
        """Test initializing a NeuralNetworkModel."""
        model = NeuralNetworkModel(name="test_nn")
        assert model.name == "test_nn"
        assert model.model_params is not None
        assert "hidden_dims" in model.model_params
        assert isinstance(model.model_params["hidden_dims"], list)

        # Test with custom parameters
        model_params = {
            "hidden_dims": [64, 32],
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
        model = NeuralNetworkModel(name="test_nn", model_params=model_params)
        assert model.name == "test_nn"
        assert model.model_params == model_params

    def test_fit_predict(self):
        """Test fitting and predicting with a NeuralNetworkModel."""
        # Use a small model and few epochs for testing
        model_params = {
            "hidden_dims": [16, 8],
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 2
        }
        model = NeuralNetworkModel(name="test_nn", model_params=model_params)

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
        assert isinstance(model.model, EnhancedNeuralNetworkRegressor)

        # Make predictions
        predictions = model.predict(X)

        # Check that predictions have the right shape
        assert len(predictions) == len(X)

        # Check that predictions are reasonable (within the range of y)
        assert predictions.min() >= 0
        assert predictions.max() <= 100

    def test_fit_with_eval_set(self):
        """Test fitting with an evaluation set."""
        # Use a small model and few epochs for testing
        model_params = {
            "hidden_dims": [16, 8],
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 2
        }
        model = NeuralNetworkModel(name="test_nn", model_params=model_params)

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
        assert isinstance(model.model, EnhancedNeuralNetworkRegressor)

        # Make predictions
        predictions = model.predict(X_train)

        # Check that predictions have the right shape
        assert len(predictions) == len(X_train)

    def test_save_load(self, tmp_path):
        """Test saving and loading a model."""
        # Use a small model and few epochs for testing
        model_params = {
            "hidden_dims": [16, 8],
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 2
        }
        model = NeuralNetworkModel(name="test_nn", model_params=model_params)

        # Create dummy data
        X = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100)
        })
        y = pd.Series(np.random.rand(100) * 100)

        # Fit the model
        model.fit(X, y)

        # Create a temporary file path
        model_path = os.path.join(tmp_path, "test_nn.pt")

        # Save the model
        model.save(model_path)

        # Check that the file exists
        assert os.path.exists(model_path)

        # Create a new model
        new_model = NeuralNetworkModel(name="test_nn", model_params=model_params)

        # Load the model
        new_model.load(model_path)

        # Check that the model has been loaded
        assert new_model.model is not None
        assert isinstance(new_model.model, EnhancedNeuralNetworkRegressor)

        # Make predictions with the loaded model
        new_predictions = new_model.predict(X)

        # Check that predictions have the right shape
        assert len(new_predictions) == len(X)

        # Check that predictions are the same as the original model
        # Note: Due to floating point precision issues, we use a tolerance
        original_predictions = model.predict(X)
        np.testing.assert_allclose(new_predictions, original_predictions, rtol=1e-5, atol=1e-5)

    def test_get_feature_importances(self):
        """Test getting feature importances."""
        # Neural networks don't have built-in feature importances, so this should raise NotImplementedError
        model = NeuralNetworkModel(name="test_nn")

        # Create dummy data
        X = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100)
        })
        y = pd.Series(np.random.rand(100) * 100)

        # Fit the model
        model_params = {
            "hidden_dims": [16, 8],
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 2
        }
        model = NeuralNetworkModel(name="test_nn", model_params=model_params)
        model.fit(X, y)

        # Get feature importances
        with pytest.raises(NotImplementedError):
            model.get_feature_importances()


class TestEnhancedNeuralNetworkRegressor:
    """Tests for the EnhancedNeuralNetworkRegressor class."""

    def test_init(self):
        """Test initializing an EnhancedNeuralNetworkRegressor."""
        model = EnhancedNeuralNetworkRegressor(input_dim=10)
        assert isinstance(model, torch.nn.Module)

        # Test with custom hidden dimensions
        model = EnhancedNeuralNetworkRegressor(input_dim=10, hidden_dims=[64, 32])
        assert isinstance(model, torch.nn.Module)

        # Test with custom dropout rate
        model = EnhancedNeuralNetworkRegressor(input_dim=10, dropout_rate=0.5)
        assert model.dropout_rate == 0.5

        # Test with residual connections disabled
        model = EnhancedNeuralNetworkRegressor(input_dim=10, use_residual=False)
        assert model.use_residual == False

        # Test with different activation functions
        model = EnhancedNeuralNetworkRegressor(input_dim=10, activation='leaky_relu')
        assert isinstance(model.activation, torch.nn.LeakyReLU)

        model = EnhancedNeuralNetworkRegressor(input_dim=10, activation='elu')
        assert isinstance(model.activation, torch.nn.ELU)

        model = EnhancedNeuralNetworkRegressor(input_dim=10, activation='gelu')
        assert isinstance(model.activation, torch.nn.GELU)

    def test_forward(self):
        """Test the forward pass of an EnhancedNeuralNetworkRegressor."""
        model = EnhancedNeuralNetworkRegressor(input_dim=2, hidden_dims=[16, 8])

        # Create a dummy input tensor
        x = torch.rand(10, 2)

        # Perform a forward pass
        output = model(x)

        # Check that the output has the right shape
        assert output.shape == (10,)

        # Check that the output is a tensor
        assert isinstance(output, torch.Tensor)

    def test_residual_connections(self):
        """Test that residual connections work correctly."""
        # Create a model with residual connections
        model = EnhancedNeuralNetworkRegressor(
            input_dim=32, 
            hidden_dims=[32, 32, 32],  # Same dimensions to enable residual connections
            use_residual=True
        )

        # Check that residual connections are enabled for the appropriate layers
        assert model.layers[0]['residual'] == False  # First layer should not have residual
        assert model.layers[1]['residual'] == True   # Second layer should have residual
        assert model.layers[2]['residual'] == True   # Third layer should have residual

        # Create a model without residual connections
        model_no_residual = EnhancedNeuralNetworkRegressor(
            input_dim=32, 
            hidden_dims=[32, 32, 32],
            use_residual=False
        )

        # Check that residual connections are disabled for all layers
        assert model_no_residual.layers[0]['residual'] == False
        assert model_no_residual.layers[1]['residual'] == False
        assert model_no_residual.layers[2]['residual'] == False


class TestChessRatingLoss:
    """Tests for the ChessRatingLoss class."""

    def test_init(self):
        """Test initializing a ChessRatingLoss."""
        loss = ChessRatingLoss()
        assert isinstance(loss, torch.nn.Module)

        # Test with custom rating ranges and weights
        custom_ranges = [(0, 1000), (1000, 2000), (2000, 3000)]
        custom_weights = [1.0, 2.0, 3.0]
        loss = ChessRatingLoss(rating_ranges=custom_ranges, weights=custom_weights)
        assert loss.rating_ranges == custom_ranges
        assert loss.weights == custom_weights

    def test_forward(self):
        """Test the forward pass of a ChessRatingLoss."""
        loss = ChessRatingLoss()

        # Create dummy predictions and targets
        pred = torch.tensor([1100.0, 1500.0, 2000.0, 2500.0])
        target = torch.tensor([1000.0, 1600.0, 1900.0, 2600.0])

        # Calculate loss
        output = loss(pred, target)

        # Check that the output is a scalar
        assert output.dim() == 0

        # Check that the output is non-negative
        assert output.item() >= 0
