"""
Model stacking implementation for chess puzzle rating prediction.

This module provides classes and functions for:
1. Creating a diverse set of base models (LightGBM, XGBoost, Neural Networks)
2. Implementing proper cross-validation for stacking
3. Adding meta-learner optimization
4. Creating rating range-specific models and ensembles
"""

import os
# Set the boost_compute directory to /raid/sroziewski/.boost_compute
os.environ['BOOST_COMPUTE_DEFAULT_TEMP_PATH'] = '/raid/sroziewski/.boost_compute'

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional, Any
import time
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna
from tqdm.auto import tqdm

from chess_puzzle_rating.utils.config import get_config
from chess_puzzle_rating.utils.progress import (
    setup_logging, get_logger, log_time, ProgressTracker, 
    track_progress, record_metric
)

# Get configuration
config = get_config()

# Set up logger
logger = get_logger()

class BaseModel:
    """Base class for all models used in stacking."""

    def __init__(self, name: str, model_params: Dict[str, Any] = None):
        """
        Initialize the base model.

        Args:
            name: Name of the model
            model_params: Parameters for the model
        """
        self.name = name
        self.model_params = model_params or {}
        self.model = None
        self.feature_importances_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: List[Tuple[pd.DataFrame, pd.Series]] = None) -> 'BaseModel':
        """
        Fit the model to the data.

        Args:
            X: Training features
            y: Training target
            eval_set: Evaluation set for early stopping

        Returns:
            self: The fitted model
        """
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model.

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """
        raise NotImplementedError("Subclasses must implement predict method")

    def save(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model to
        """
        raise NotImplementedError("Subclasses must implement save method")

    def load(self, path: str) -> 'BaseModel':
        """
        Load the model from disk.

        Args:
            path: Path to load the model from

        Returns:
            self: The loaded model
        """
        raise NotImplementedError("Subclasses must implement load method")

    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the model.

        Returns:
            Feature importances
        """
        return self.feature_importances_


class LightGBMModel(BaseModel):
    """LightGBM model implementation."""

    def __init__(self, name: str = "lightgbm", model_params: Dict[str, Any] = None):
        """
        Initialize the LightGBM model.

        Args:
            name: Name of the model
            model_params: Parameters for the model
        """
        super().__init__(name, model_params)
        self.early_stopping_rounds = self.model_params.pop('early_stopping_rounds', 50)
        self.model = lgb.LGBMRegressor(**self.model_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: List[Tuple[pd.DataFrame, pd.Series]] = None) -> 'LightGBMModel':
        """
        Fit the LightGBM model to the data.

        Args:
            X: Training features
            y: Training target
            eval_set: Evaluation set for early stopping

        Returns:
            self: The fitted model
        """
        callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]

        self.model.fit(
            X, y,
            eval_set=eval_set,
            eval_metric='rmse',
            callbacks=callbacks
        )

        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the LightGBM model.

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """
        return self.model.predict(X, num_iteration=self.model.best_iteration_)

    def save(self, path: str) -> None:
        """
        Save the LightGBM model to disk.

        Args:
            path: Path to save the model to
        """
        self.model.booster_.save_model(path)

    def load(self, path: str) -> 'LightGBMModel':
        """
        Load the LightGBM model from disk.

        Args:
            path: Path to load the model from

        Returns:
            self: The loaded model
        """
        self.model = lgb.Booster(model_file=path)
        return self


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""

    def __init__(self, name: str = "xgboost", model_params: Dict[str, Any] = None):
        """
        Initialize the XGBoost model.

        Args:
            name: Name of the model
            model_params: Parameters for the model
        """
        super().__init__(name, model_params)
        self.early_stopping_rounds = self.model_params.pop('early_stopping_rounds', 50)
        # Remove eval_metric from model_params to avoid it being passed to fit() method
        self.eval_metric = self.model_params.pop('eval_metric', None)
        self.model = xgb.XGBRegressor(**self.model_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: List[Tuple[pd.DataFrame, pd.Series]] = None) -> 'XGBoostModel':
        """
        Fit the XGBoost model to the data.

        Args:
            X: Training features
            y: Training target
            eval_set: Evaluation set for early stopping

        Returns:
            self: The fitted model
        """
        self.model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False
        )

        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the XGBoost model.

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """
        return self.model.predict(X, ntree_limit=self.model.best_ntree_limit)

    def save(self, path: str) -> None:
        """
        Save the XGBoost model to disk.

        Args:
            path: Path to save the model to
        """
        self.model.save_model(path)

    def load(self, path: str) -> 'XGBoostModel':
        """
        Load the XGBoost model from disk.

        Args:
            path: Path to load the model from

        Returns:
            self: The loaded model
        """
        self.model = xgb.Booster()
        self.model.load_model(path)
        return self


class EnhancedNeuralNetworkRegressor(nn.Module):
    """Enhanced neural network model for regression with residual connections and more options."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = None, 
                 dropout_rate: float = 0.3, 
                 use_residual: bool = True,
                 activation: str = 'relu'):
        """
        Initialize the enhanced neural network model.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_residual: Whether to use residual connections
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'gelu')
        """
        super(EnhancedNeuralNetworkRegressor, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        self.use_residual = use_residual
        self.dropout_rate = dropout_rate

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Create layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for i, dim in enumerate(hidden_dims):
            layer_block = nn.ModuleDict({
                'linear': nn.Linear(prev_dim, dim),
                'bn': nn.BatchNorm1d(dim),
                'dropout': nn.Dropout(dropout_rate)
            })

            # Add residual connection if dimensions match and not first layer
            if use_residual and i > 0 and prev_dim == dim:
                layer_block['residual'] = True
            else:
                layer_block['residual'] = False

            self.layers.append(layer_block)
            prev_dim = dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Normalize input
        x = self.input_norm(x)

        # Process through hidden layers
        for i, layer_block in enumerate(self.layers):
            identity = x

            x = layer_block['linear'](x)
            x = self.activation(x)
            x = layer_block['bn'](x)

            # Apply residual connection if specified
            if layer_block['residual']:
                x = x + identity

            x = layer_block['dropout'](x)

        # Output layer
        x = self.output_layer(x)

        return x.squeeze()


class ChessRatingLoss(nn.Module):
    """
    Custom loss function for chess rating prediction that penalizes more for errors
    in certain rating ranges where precision is more important.
    """
    def __init__(self, rating_ranges=None, weights=None):
        """
        Initialize the chess rating loss.

        Args:
            rating_ranges: List of rating range tuples [(min1, max1), (min2, max2), ...]
            weights: List of weights for each range (higher weight = higher penalty)
        """
        super(ChessRatingLoss, self).__init__()

        # Default ranges and weights if not provided
        if rating_ranges is None:
            # Lower ratings (more beginners) might need more precision
            rating_ranges = [(0, 1200), (1200, 1800), (1800, 2400), (2400, 3500)]

        if weights is None:
            # Higher weight for middle ranges where most players are
            weights = [1.0, 1.5, 1.2, 1.0]

        self.rating_ranges = rating_ranges
        self.weights = weights

    def forward(self, pred, target):
        """
        Calculate the weighted MSE loss.

        Args:
            pred: Predicted ratings
            target: Ground truth ratings

        Returns:
            Weighted loss
        """
        mse_loss = (pred - target) ** 2

        # Apply weights based on target rating ranges
        weighted_loss = torch.zeros_like(mse_loss)

        for i, (min_val, max_val) in enumerate(self.rating_ranges):
            mask = (target >= min_val) & (target < max_val)
            weighted_loss[mask] = mse_loss[mask] * self.weights[i]

        return weighted_loss.mean()


class NeuralNetworkModel(BaseModel):
    """Neural network model implementation."""

    def __init__(self, name: str = "neural_network", model_params: Dict[str, Any] = None):
        """
        Initialize the neural network model.

        Args:
            name: Name of the model
            model_params: Parameters for the model
        """
        super().__init__(name, model_params)
        self.input_dim = None
        self.hidden_dims = self.model_params.get('hidden_dims', [256, 128, 64])
        self.learning_rate = self.model_params.get('learning_rate', 0.001)
        self.batch_size = self.model_params.get('batch_size', 256)
        self.epochs = self.model_params.get('epochs', 100)
        self.patience = self.model_params.get('patience', 10)
        self.dropout_rate = self.model_params.get('dropout_rate', 0.3)
        self.use_residual = self.model_params.get('use_residual', True)
        self.activation = self.model_params.get('activation', 'relu')
        self.use_custom_loss = self.model_params.get('use_custom_loss', True)
        self.rating_ranges = self.model_params.get('rating_ranges', None)
        self.loss_weights = self.model_params.get('loss_weights', None)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_importances_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: List[Tuple[pd.DataFrame, pd.Series]] = None) -> 'NeuralNetworkModel':
        """
        Fit the neural network model to the data.

        Args:
            X: Training features
            y: Training target
            eval_set: Evaluation set for early stopping

        Returns:
            self: The fitted model
        """
        self.input_dim = X.shape[1]
        self.model = EnhancedNeuralNetworkRegressor(
            input_dim=self.input_dim, 
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_residual=self.use_residual,
            activation=self.activation
        ).to(self.device)

        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)

        # Create data loaders
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Create validation data loader if eval_set is provided
        val_loader = None
        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Define loss function and optimizer
        if self.use_custom_loss:
            criterion = ChessRatingLoss(rating_ranges=self.rating_ranges, weights=self.loss_weights)
        else:
            criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)

                val_loss /= len(val_loader.dataset)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                # If no validation set, save the model at the last epoch
                best_model_state = self.model.state_dict().copy()

        # Load the best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Calculate feature importances using permutation importance
        # This is a simple implementation and can be improved
        self.feature_importances_ = np.zeros(self.input_dim)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the neural network model.

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions

    def save(self, path: str) -> None:
        """
        Save the neural network model to disk.

        Args:
            path: Path to save the model to
        """
        model_state = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'use_residual': self.use_residual,
            'activation': self.activation,
            'state_dict': self.model.state_dict()
        }
        torch.save(model_state, path)

    def load(self, path: str) -> 'NeuralNetworkModel':
        """
        Load the neural network model from disk.

        Args:
            path: Path to load the model from

        Returns:
            self: The loaded model
        """
        model_state = torch.load(path, map_location=self.device)
        self.input_dim = model_state['input_dim']
        self.hidden_dims = model_state['hidden_dims']

        # Get additional parameters if available, otherwise use defaults
        self.dropout_rate = model_state.get('dropout_rate', self.dropout_rate)
        self.use_residual = model_state.get('use_residual', self.use_residual)
        self.activation = model_state.get('activation', self.activation)

        self.model = EnhancedNeuralNetworkRegressor(
            input_dim=self.input_dim, 
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_residual=self.use_residual,
            activation=self.activation
        ).to(self.device)

        self.model.load_state_dict(model_state['state_dict'])
        self.model.eval()
        return self


class StackingModel:
    """
    Stacking model implementation that combines multiple base models.

    This class implements a stacking approach where multiple base models are trained
    on the same data, and their predictions are used as features for a meta-learner.
    """

    def __init__(
        self,
        base_models: List[BaseModel],
        meta_learner: BaseModel = None,
        n_splits: int = 5,
        random_state: int = 42,
        use_features_in_meta: bool = True
    ):
        """
        Initialize the stacking model.

        Args:
            base_models: List of base models to use in stacking
            meta_learner: Meta-learner model (if None, a LightGBM model will be used)
            n_splits: Number of cross-validation splits
            random_state: Random state for reproducibility
            use_features_in_meta: Whether to use original features in meta-learner
        """
        self.base_models = base_models
        self.meta_learner = meta_learner or LightGBMModel(
            name="meta_learner",
            model_params={
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'min_child_samples': 10,  # Reduced from default 20
                'min_gain_to_split': 0.0,  # Allow splits with no gain
                'early_stopping_rounds': 50,
                'random_state': random_state
            }
        )
        self.n_splits = n_splits
        self.random_state = random_state
        self.use_features_in_meta = use_features_in_meta
        self.base_model_paths = []
        self.meta_learner_path = None
        self.feature_names = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        model_dir: str = 'models'
    ) -> 'StackingModel':
        """
        Fit the stacking model to the data.

        Args:
            X: Training features
            y: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            model_dir: Directory to save models

        Returns:
            self: The fitted model
        """
        logger.info(f"Fitting stacking model with {len(self.base_models)} base models")
        logger.info(f"Using {self.n_splits}-fold cross-validation")

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Create cross-validation folds
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        # Initialize arrays to store out-of-fold predictions and test predictions
        oof_predictions = np.zeros((len(self.base_models), X.shape[0]))

        # If validation set is provided, initialize array for validation predictions
        val_predictions = None
        if X_val is not None and y_val is not None:
            val_predictions = np.zeros((len(self.base_models), X_val.shape[0]))

        # Train each base model using cross-validation
        for i, base_model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {base_model.name}")

            # Initialize array for out-of-fold predictions for this model
            model_oof_preds = np.zeros(X.shape[0])

            # Initialize array for validation predictions for this model
            model_val_preds = None
            if val_predictions is not None:
                model_val_preds = np.zeros(X_val.shape[0])

            # Train the model on each fold
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                logger.info(f"Training fold {fold+1}/{self.n_splits}")

                # Split data for this fold
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                # Fit the model
                base_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)])

                # Make predictions on validation fold
                model_oof_preds[val_idx] = base_model.predict(X_val_fold)

                # Save the model
                model_path = os.path.join(model_dir, f"{base_model.name}_fold_{fold}.model")
                base_model.save(model_path)
                self.base_model_paths.append((base_model.name, model_path))

                # If validation set is provided, make predictions on it
                if model_val_preds is not None:
                    # For validation, we train on the entire training set
                    if fold == self.n_splits - 1:  # Only on the last fold
                        base_model.fit(X, y, eval_set=[(X_val, y_val)])
                        model_val_preds = base_model.predict(X_val)

            # Store out-of-fold predictions for this model
            oof_predictions[i] = model_oof_preds

            # Store validation predictions for this model
            if val_predictions is not None:
                val_predictions[i] = model_val_preds

            # Calculate RMSE for this model
            model_rmse = np.sqrt(mean_squared_error(y, model_oof_preds))
            logger.info(f"Base model {base_model.name} OOF RMSE: {model_rmse:.4f}")

        # Prepare meta-features for training the meta-learner
        meta_features = pd.DataFrame(oof_predictions.T, index=X.index)
        meta_features.columns = [f"{model.name}_pred" for model in self.base_models]

        # If using original features in meta-learner, add them
        if self.use_features_in_meta:
            meta_features = pd.concat([meta_features, X.reset_index(drop=True)], axis=1)

        # Prepare validation meta-features if validation set is provided
        meta_features_val = None
        if val_predictions is not None:
            meta_features_val = pd.DataFrame(val_predictions.T, index=X_val.index)
            meta_features_val.columns = [f"{model.name}_pred" for model in self.base_models]

            if self.use_features_in_meta:
                meta_features_val = pd.concat([meta_features_val, X_val.reset_index(drop=True)], axis=1)

        # Train the meta-learner
        logger.info("Training meta-learner")
        eval_set = [(meta_features_val, y_val)] if meta_features_val is not None else None
        self.meta_learner.fit(meta_features, y, eval_set=eval_set)

        # Save the meta-learner
        meta_learner_path = os.path.join(model_dir, f"{self.meta_learner.name}.model")
        self.meta_learner.save(meta_learner_path)
        self.meta_learner_path = meta_learner_path

        # Calculate final RMSE
        final_preds = self.meta_learner.predict(meta_features)
        final_rmse = np.sqrt(mean_squared_error(y, final_preds))
        logger.info(f"Final stacking model RMSE: {final_rmse:.4f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the stacking model.

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """
        # Make predictions with each base model
        base_predictions = np.zeros((len(self.base_models), X.shape[0]))

        for i, base_model in enumerate(self.base_models):
            base_predictions[i] = base_model.predict(X)

        # Prepare meta-features for prediction
        meta_features = pd.DataFrame(base_predictions.T, index=X.index)
        meta_features.columns = [f"{model.name}_pred" for model in self.base_models]

        # If using original features in meta-learner, add them
        if self.use_features_in_meta:
            meta_features = pd.concat([meta_features, X.reset_index(drop=True)], axis=1)

        # Make predictions with meta-learner
        return self.meta_learner.predict(meta_features)

    def save(self, path: str) -> None:
        """
        Save the stacking model configuration to disk.

        Args:
            path: Path to save the model configuration to
        """
        model_config = {
            'base_model_paths': self.base_model_paths,
            'meta_learner_path': self.meta_learner_path,
            'n_splits': self.n_splits,
            'random_state': self.random_state,
            'use_features_in_meta': self.use_features_in_meta,
            'feature_names': self.feature_names
        }

        import pickle
        with open(path, 'wb') as f:
            pickle.dump(model_config, f)

    def load(self, path: str) -> 'StackingModel':
        """
        Load the stacking model configuration from disk.

        Args:
            path: Path to load the model configuration from

        Returns:
            self: The loaded model
        """
        import pickle
        with open(path, 'rb') as f:
            model_config = pickle.load(f)

        self.base_model_paths = model_config['base_model_paths']
        self.meta_learner_path = model_config['meta_learner_path']
        self.n_splits = model_config['n_splits']
        self.random_state = model_config['random_state']
        self.use_features_in_meta = model_config['use_features_in_meta']
        self.feature_names = model_config['feature_names']

        # Load base models
        self.base_models = []
        for model_name, model_path in self.base_model_paths:
            if 'lightgbm' in model_name.lower():
                model = LightGBMModel(name=model_name)
            elif 'xgboost' in model_name.lower():
                model = XGBoostModel(name=model_name)
            elif 'neural' in model_name.lower():
                model = NeuralNetworkModel(name=model_name)
            else:
                raise ValueError(f"Unknown model type: {model_name}")

            model.load(model_path)
            self.base_models.append(model)

        # Load meta-learner
        if 'lightgbm' in self.meta_learner_path.lower():
            self.meta_learner = LightGBMModel(name="meta_learner")
        elif 'xgboost' in self.meta_learner_path.lower():
            self.meta_learner = XGBoostModel(name="meta_learner")
        elif 'neural' in self.meta_learner_path.lower():
            self.meta_learner = NeuralNetworkModel(name="meta_learner")
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_path}")

        self.meta_learner.load(self.meta_learner_path)

        return self


def optimize_meta_learner(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    meta_learner_type: str = 'lightgbm',
    n_trials: int = 100,
    timeout: int = 3600,
    random_state: int = 42
) -> Tuple[Dict[str, Any], float]:
    """
    Optimize hyperparameters for the meta-learner using Optuna.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        meta_learner_type: Type of meta-learner ('lightgbm', 'xgboost', or 'neural_network')
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        random_state: Random state for reproducibility

    Returns:
        Tuple of (best_params, best_score)
    """
    logger.info(f"Optimizing {meta_learner_type} meta-learner with Optuna")

    def objective(trial):
        if meta_learner_type == 'lightgbm':
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': random_state
            }
            model = LightGBMModel(name="meta_learner_optuna", model_params=params)

        elif meta_learner_type == 'xgboost':
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
                'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
                'random_state': random_state
            }
            model = XGBoostModel(name="meta_learner_optuna", model_params=params)

        elif meta_learner_type == 'neural_network':
            hidden_dims = [
                trial.suggest_int('hidden_dim_1', 64, 512),
                trial.suggest_int('hidden_dim_2', 32, 256),
                trial.suggest_int('hidden_dim_3', 16, 128)
            ]
            params = {
                'hidden_dims': hidden_dims,
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
                'epochs': 100,  # Fixed for optimization
                'patience': 10,  # Fixed for optimization
            }
            model = NeuralNetworkModel(name="meta_learner_optuna", model_params=params)

        else:
            raise ValueError(f"Unknown meta-learner type: {meta_learner_type}")

        # Fit the model
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        # Make predictions on validation set
        val_preds = model.predict(X_val)

        # Calculate RMSE
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

        return val_rmse

    # Create Optuna study
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(f"Best {meta_learner_type} meta-learner parameters: {study.best_params}")
    logger.info(f"Best {meta_learner_type} meta-learner RMSE: {study.best_value:.4f}")

    return study.best_params, study.best_value


class RangeSpecificStackingModel:
    """
    Rating range-specific stacking model implementation.

    This class trains separate stacking models for different rating ranges and
    combines their predictions using an ensemble approach.
    """

    def __init__(
        self,
        rating_ranges: List[List[int]],
        range_overlap: int = 100,
        ensemble_method: str = 'weighted_average',
        n_splits: int = 5,
        random_state: int = 42,
        optimize_meta: bool = True,
        meta_learner_type: str = 'lightgbm',
        use_features_in_meta: bool = True
    ):
        """
        Initialize the range-specific stacking model.

        Args:
            rating_ranges: List of rating ranges, each as [min_rating, max_rating]
            range_overlap: Overlap between ranges for smooth transitions
            ensemble_method: Method for combining predictions ('weighted_average' or 'stacking')
            n_splits: Number of cross-validation splits
            random_state: Random state for reproducibility
            optimize_meta: Whether to optimize meta-learner hyperparameters
            meta_learner_type: Type of meta-learner ('lightgbm', 'xgboost', or 'neural_network')
            use_features_in_meta: Whether to use original features in meta-learner
        """
        self.rating_ranges = rating_ranges
        self.range_overlap = range_overlap
        self.ensemble_method = ensemble_method
        self.n_splits = n_splits
        self.random_state = random_state
        self.optimize_meta = optimize_meta
        self.meta_learner_type = meta_learner_type
        self.use_features_in_meta = use_features_in_meta
        self.range_models = {}
        self.ensemble_meta_learner = None
        self.feature_names = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        model_dir: str = 'models'
    ) -> 'RangeSpecificStackingModel':
        """
        Fit range-specific models to the data.

        Args:
            X: Training features
            y: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            model_dir: Directory to save models

        Returns:
            self: The fitted model
        """
        logger.info(f"Fitting range-specific stacking models for {len(self.rating_ranges)} rating ranges")

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Train a model for each rating range
        for i, (min_rating, max_rating) in enumerate(self.rating_ranges):
            range_name = f"range_{min_rating}_{max_rating}"
            range_dir = os.path.join(model_dir, range_name)
            os.makedirs(range_dir, exist_ok=True)

            logger.info(f"Training model for rating range {min_rating}-{max_rating}")

            # Select data for this range (including overlap)
            min_with_overlap = min_rating - self.range_overlap
            max_with_overlap = max_rating + self.range_overlap

            # For training data
            range_mask = (y >= min_with_overlap) & (y <= max_with_overlap)
            X_range = X[range_mask]
            y_range = y[range_mask]

            # For validation data if provided
            X_val_range = None
            y_val_range = None
            if X_val is not None and y_val is not None:
                val_range_mask = (y_val >= min_with_overlap) & (y_val <= max_with_overlap)
                X_val_range = X_val[val_range_mask]
                y_val_range = y_val[val_range_mask]

            # Skip if not enough data
            if len(y_range) < 100:
                logger.warning(f"Not enough data for range {min_rating}-{max_rating}, skipping")
                continue

            logger.info(f"Training with {len(y_range)} samples for range {min_rating}-{max_rating}")

            # Create and train a stacking model for this range
            range_model = create_stacking_model(
                X_train=X_range,
                y_train=y_range,
                X_val=X_val_range,
                y_val=y_val_range,
                n_splits=self.n_splits,
                random_state=self.random_state,
                optimize_meta=self.optimize_meta,
                meta_learner_type=self.meta_learner_type,
                use_features_in_meta=self.use_features_in_meta,
                model_dir=range_dir
            )

            # Save the range model configuration
            range_config_path = os.path.join(range_dir, f"{range_name}_config.pkl")
            range_model.save(range_config_path)

            # Store the model
            self.range_models[range_name] = {
                'model': range_model,
                'min_rating': min_rating,
                'max_rating': max_rating,
                'config_path': range_config_path
            }

            logger.info(f"Model for range {min_rating}-{max_rating} trained and saved")

        # If using stacking as ensemble method, train a meta-learner
        if self.ensemble_method == 'stacking' and X_val is not None and y_val is not None:
            logger.info("Training ensemble meta-learner for stacking ensemble method")

            # Get predictions from each range model
            range_preds = np.zeros((len(self.range_models), X_val.shape[0]))
            range_names = []

            for i, (range_name, range_info) in enumerate(self.range_models.items()):
                range_preds[i] = range_info['model'].predict(X_val)
                range_names.append(range_name)

            # Create meta-features for ensemble meta-learner
            meta_features = pd.DataFrame(range_preds.T, index=X_val.index)
            meta_features.columns = [f"{name}_pred" for name in range_names]

            # Train ensemble meta-learner
            self.ensemble_meta_learner = LightGBMModel(
                name="ensemble_meta_learner",
                model_params={
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'learning_rate': 0.05,
                    'n_estimators': 1000,
                    'min_child_samples': 10,  # Reduced from default 20
                    'min_gain_to_split': 0.0,  # Allow splits with no gain
                    'early_stopping_rounds': 50,
                    'random_state': self.random_state
                }
            )

            self.ensemble_meta_learner.fit(meta_features, y_val)

            # Save ensemble meta-learner
            ensemble_meta_path = os.path.join(model_dir, "ensemble_meta_learner.model")
            self.ensemble_meta_learner.save(ensemble_meta_path)

            logger.info("Ensemble meta-learner trained and saved")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the range-specific models.

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """
        if not self.range_models:
            raise ValueError("No range models available. Please train the model first.")

        # Get predictions from each range model
        range_preds = {}
        for range_name, range_info in self.range_models.items():
            range_preds[range_name] = range_info['model'].predict(X)

        # If using stacking ensemble method and ensemble meta-learner is available
        if self.ensemble_method == 'stacking' and self.ensemble_meta_learner is not None:
            # Create meta-features for ensemble meta-learner
            meta_features = pd.DataFrame(
                {f"{name}_pred": preds for name, preds in range_preds.items()},
                index=X.index
            )

            # Make predictions with ensemble meta-learner
            return self.ensemble_meta_learner.predict(meta_features)

        # Otherwise, use weighted average ensemble method
        else:
            # Initialize weights and predictions arrays
            weights = np.zeros((len(self.range_models), X.shape[0]))
            predictions = np.zeros((len(self.range_models), X.shape[0]))

            # For each range model
            for i, (range_name, range_info) in enumerate(self.range_models.items()):
                min_rating = range_info['min_rating']
                max_rating = range_info['max_rating']

                # Get predictions from this model
                predictions[i] = range_preds[range_name]

                # Calculate weights based on predicted values
                # Higher weight when prediction is within the model's range
                # Linear decay outside the range
                for j, pred in enumerate(predictions[i]):
                    if pred < min_rating:
                        # Linear decay below min_rating
                        distance = min_rating - pred
                        if distance <= self.range_overlap:
                            weights[i, j] = 1.0 - (distance / self.range_overlap)
                        else:
                            weights[i, j] = 0.0
                    elif pred > max_rating:
                        # Linear decay above max_rating
                        distance = pred - max_rating
                        if distance <= self.range_overlap:
                            weights[i, j] = 1.0 - (distance / self.range_overlap)
                        else:
                            weights[i, j] = 0.0
                    else:
                        # Full weight within range
                        weights[i, j] = 1.0

            # Normalize weights
            weight_sums = np.sum(weights, axis=0)
            # Avoid division by zero
            weight_sums[weight_sums == 0] = 1.0
            normalized_weights = weights / weight_sums

            # Weighted average of predictions
            final_predictions = np.sum(predictions * normalized_weights, axis=0)

            return final_predictions

    def save(self, path: str) -> None:
        """
        Save the range-specific stacking model configuration to disk.

        Args:
            path: Path to save the model configuration to
        """
        model_config = {
            'rating_ranges': self.rating_ranges,
            'range_overlap': self.range_overlap,
            'ensemble_method': self.ensemble_method,
            'n_splits': self.n_splits,
            'random_state': self.random_state,
            'optimize_meta': self.optimize_meta,
            'meta_learner_type': self.meta_learner_type,
            'use_features_in_meta': self.use_features_in_meta,
            'feature_names': self.feature_names,
            'range_models': {
                name: {
                    'min_rating': info['min_rating'],
                    'max_rating': info['max_rating'],
                    'config_path': info['config_path']
                } for name, info in self.range_models.items()
            }
        }

        # If ensemble meta-learner exists, save its path
        if self.ensemble_meta_learner is not None:
            model_dir = os.path.dirname(path)
            ensemble_meta_path = os.path.join(model_dir, "ensemble_meta_learner.model")
            model_config['ensemble_meta_path'] = ensemble_meta_path

        import pickle
        with open(path, 'wb') as f:
            pickle.dump(model_config, f)

    def load(self, path: str) -> 'RangeSpecificStackingModel':
        """
        Load the range-specific stacking model configuration from disk.

        Args:
            path: Path to load the model configuration from

        Returns:
            self: The loaded model
        """
        import pickle
        with open(path, 'rb') as f:
            model_config = pickle.load(f)

        self.rating_ranges = model_config['rating_ranges']
        self.range_overlap = model_config['range_overlap']
        self.ensemble_method = model_config['ensemble_method']
        self.n_splits = model_config['n_splits']
        self.random_state = model_config['random_state']
        self.optimize_meta = model_config['optimize_meta']
        self.meta_learner_type = model_config['meta_learner_type']
        self.use_features_in_meta = model_config['use_features_in_meta']
        self.feature_names = model_config['feature_names']

        # Load range models
        self.range_models = {}
        for name, info in model_config['range_models'].items():
            # Load the stacking model for this range
            range_model = StackingModel(
                base_models=[],  # Will be loaded from config
                n_splits=self.n_splits,
                random_state=self.random_state,
                use_features_in_meta=self.use_features_in_meta
            )
            range_model.load(info['config_path'])

            # Store the model
            self.range_models[name] = {
                'model': range_model,
                'min_rating': info['min_rating'],
                'max_rating': info['max_rating'],
                'config_path': info['config_path']
            }

        # Load ensemble meta-learner if it exists
        if 'ensemble_meta_path' in model_config:
            self.ensemble_meta_learner = LightGBMModel(name="ensemble_meta_learner")
            self.ensemble_meta_learner.load(model_config['ensemble_meta_path'])
        else:
            self.ensemble_meta_learner = None

        return self


def create_stacking_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    n_splits: int = 5,
    random_state: int = 42,
    optimize_meta: bool = True,
    meta_learner_type: str = 'lightgbm',
    use_features_in_meta: bool = True,
    model_dir: str = 'models'
) -> StackingModel:
    """
    Create and train a stacking model with optimized hyperparameters.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        n_splits: Number of cross-validation splits
        random_state: Random state for reproducibility
        optimize_meta: Whether to optimize meta-learner hyperparameters
        meta_learner_type: Type of meta-learner ('lightgbm', 'xgboost', or 'neural_network')
        use_features_in_meta: Whether to use original features in meta-learner
        model_dir: Directory to save models

    Returns:
        Trained stacking model
    """
    logger.info("Creating stacking model")

    # Create base models
    base_models = [
        LightGBMModel(
            name="lightgbm_base",
            model_params={
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'num_leaves': 31,
                'max_depth': -1,
                'min_child_samples': 10,  # Reduced from default 20
                'min_gain_to_split': 0.0,  # Allow splits with no gain
                'early_stopping_rounds': 50,
                'random_state': random_state
            }
        ),
        XGBoostModel(
            name="xgboost_base",
            model_params={
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'max_depth': 6,
                'early_stopping_rounds': 50,
                'random_state': random_state
            }
        ),
        NeuralNetworkModel(
            name="neural_network_base",
            model_params={
                'hidden_dims': [256, 128, 64],
                'learning_rate': 0.001,
                'batch_size': 256,
                'epochs': 100,
                'patience': 10
            }
        )
    ]

    # Create meta-learner
    meta_learner = None

    if optimize_meta and X_val is not None and y_val is not None:
        # First, train base models on the entire training set
        logger.info("Training base models for meta-learner optimization")
        base_preds_train = np.zeros((len(base_models), X_train.shape[0]))
        base_preds_val = np.zeros((len(base_models), X_val.shape[0]))

        for i, model in enumerate(base_models):
            logger.info(f"Training base model {i+1}/{len(base_models)}: {model.name}")
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            base_preds_train[i] = model.predict(X_train)
            base_preds_val[i] = model.predict(X_val)

        # Prepare meta-features
        meta_features_train = pd.DataFrame(base_preds_train.T, index=X_train.index)
        meta_features_train.columns = [f"{model.name}_pred" for model in base_models]

        meta_features_val = pd.DataFrame(base_preds_val.T, index=X_val.index)
        meta_features_val.columns = [f"{model.name}_pred" for model in base_models]

        if use_features_in_meta:
            meta_features_train = pd.concat([meta_features_train, X_train.reset_index(drop=True)], axis=1)
            meta_features_val = pd.concat([meta_features_val, X_val.reset_index(drop=True)], axis=1)

        # Optimize meta-learner
        best_params, _ = optimize_meta_learner(
            meta_features_train, y_train,
            meta_features_val, y_val,
            meta_learner_type=meta_learner_type,
            n_trials=100,
            timeout=3600,
            random_state=random_state
        )

        # Create meta-learner with optimized parameters
        if meta_learner_type == 'lightgbm':
            meta_learner = LightGBMModel(name="meta_learner_optimized", model_params=best_params)
        elif meta_learner_type == 'xgboost':
            meta_learner = XGBoostModel(name="meta_learner_optimized", model_params=best_params)
        elif meta_learner_type == 'neural_network':
            meta_learner = NeuralNetworkModel(name="meta_learner_optimized", model_params=best_params)

    # Create stacking model
    stacking_model = StackingModel(
        base_models=base_models,
        meta_learner=meta_learner,
        n_splits=n_splits,
        random_state=random_state,
        use_features_in_meta=use_features_in_meta
    )

    # Train stacking model
    stacking_model.fit(X_train, y_train, X_val, y_val, model_dir=model_dir)

    return stacking_model
