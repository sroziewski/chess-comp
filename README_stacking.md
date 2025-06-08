# Model Stacking Implementation for Chess Puzzle Rating Prediction

This document provides an overview of the model stacking implementation for the Chess Puzzle Rating Prediction project.

## Overview

Model stacking is an ensemble learning technique that combines multiple base models to improve prediction accuracy. The implementation includes:

1. A diverse set of base models (LightGBM, XGBoost, Neural Networks)
2. Proper cross-validation for stacking
3. Meta-learner optimization

## Implementation Details

### Base Models

The implementation includes three base models:

1. **LightGBM**: A gradient boosting framework that uses tree-based learning algorithms.
2. **XGBoost**: Another gradient boosting framework known for its performance and speed.
3. **Neural Network**: A deep learning model with configurable architecture.

Each base model is trained using k-fold cross-validation, and their predictions are used as features for the meta-learner.

### Cross-Validation for Stacking

The stacking approach uses k-fold cross-validation to generate out-of-fold predictions for the meta-learner. This ensures that the meta-learner is trained on predictions that the base models didn't see during their training, preventing leakage.

### Meta-Learner Optimization

The meta-learner's hyperparameters can be optimized using Optuna, a hyperparameter optimization framework. This ensures that the meta-learner is well-tuned to combine the base models' predictions effectively.

## How to Use

### Configuration

The stacking implementation can be configured through the `config.yaml` file. The relevant section is:

```yaml
# Stacking model configuration
stacking:
  # Number of cross-validation splits for stacking
  n_splits: 5
  # Whether to optimize meta-learner hyperparameters
  optimize_meta_learner: true
  # Type of meta-learner to use ('lightgbm', 'xgboost', or 'neural_network')
  meta_learner_type: 'lightgbm'
  # Whether to use original features in meta-learner
  use_features_in_meta: true
  # Directory to save stacking models
  model_save_dir: "trained_models_stacking"
  # Base model parameters
  base_models:
    # LightGBM base model parameters
    lightgbm:
      n_estimators: 1000
      learning_rate: 0.05
      num_leaves: 31
      max_depth: -1
      early_stopping_rounds: 50
    # XGBoost base model parameters
    xgboost:
      n_estimators: 1000
      learning_rate: 0.05
      max_depth: 6
      early_stopping_rounds: 50
    # Neural Network base model parameters
    neural_network:
      hidden_dims: [256, 128, 64]
      learning_rate: 0.001
      batch_size: 256
      epochs: 100
      patience: 10
```

You can customize:
- The number of cross-validation splits
- Whether to optimize the meta-learner
- The type of meta-learner to use
- Whether to use original features in the meta-learner
- The directory to save stacking models
- Parameters for each base model

### Training

To train a stacking model, run:

```bash
python train_stacking.py
```

This script will:
1. Load and preprocess the data
2. Perform feature engineering
3. Train the base models using cross-validation
4. Train the meta-learner on the base models' predictions
5. Evaluate the model on a validation set
6. Make predictions on the test set and generate a submission file

### Using the Stacking Model

The stacking model can be used for prediction as follows:

```python
from chess_puzzle_rating.models.stacking import StackingModel

# Load the stacking model
stacking_model = StackingModel(base_models=[], meta_learner=None)
stacking_model.load("path/to/stacking_model_config.pkl")

# Make predictions
predictions = stacking_model.predict(X_test)
```

## Benefits of Stacking

Model stacking offers several benefits:

1. **Improved Accuracy**: By combining multiple models, stacking can capture different patterns in the data and reduce overfitting.
2. **Robustness**: Stacking is less sensitive to the choice of a single model and can handle a wider range of data patterns.
3. **Flexibility**: The implementation allows for easy customization of base models and meta-learner.

## Implementation Files

- `chess_puzzle_rating/models/stacking.py`: Contains the implementation of the stacking approach, including base models, meta-learner optimization, and the stacking model.
- `train_stacking.py`: A script that demonstrates how to use the stacking implementation for training and prediction.
- `config.yaml`: Configuration file with stacking-specific parameters.

## Future Improvements

Potential improvements to the stacking implementation include:

1. Adding more diverse base models
2. Implementing feature selection for the meta-learner
3. Adding support for multi-level stacking
4. Implementing parallel training of base models