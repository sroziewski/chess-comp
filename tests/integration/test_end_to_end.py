import pytest
import pandas as pd
import numpy as np
import os
from chess_puzzle_rating.data.pipeline import run_data_pipeline
from chess_puzzle_rating.features.pipeline import complete_feature_engineering
from chess_puzzle_rating.models.stacking import create_stacking_model

class TestEndToEndPipeline:
    """Tests for the end-to-end pipeline."""
    
    @pytest.mark.skip(reason="This test requires the full dataset and may take a long time to run")
    def test_full_pipeline(self):
        """Test the full pipeline from data loading to model prediction."""
        # Run the data pipeline
        X_train, X_test, y_train, test_ids = run_data_pipeline()
        
        # Create and train a stacking model
        model = create_stacking_model(
            X_train=X_train,
            y_train=y_train,
            n_splits=2,  # Use a small number of splits for testing
            optimize_meta=False  # Disable optimization for faster testing
        )
        
        # Make predictions on the test set
        predictions = model.predict(X_test)
        
        # Check that predictions have the right shape
        assert len(predictions) == len(X_test)
        
        # Check that predictions are reasonable (within the expected range)
        min_rating = 500
        max_rating = 3000
        assert predictions.min() >= min_rating, f"Prediction below minimum: {predictions.min()}"
        assert predictions.max() <= max_rating, f"Prediction above maximum: {predictions.max()}"
    
    def test_mini_pipeline(self, sample_puzzle_df):
        """Test a mini pipeline with a small sample dataset."""
        # Split the sample data into train and test
        train_df = sample_puzzle_df.iloc[:3]
        test_df = sample_puzzle_df.iloc[3:]
        
        # Extract features
        train_features, _, _ = complete_feature_engineering(train_df)
        test_features, _, _ = complete_feature_engineering(test_df)
        
        # Extract target
        y_train = train_df['Rating']
        
        # Create and train a simple model (use LightGBM directly for simplicity)
        from chess_puzzle_rating.models.stacking import LightGBMModel
        model = LightGBMModel(name="test_model", model_params={"n_estimators": 10})
        model.fit(train_features, y_train)
        
        # Make predictions on the test set
        predictions = model.predict(test_features)
        
        # Check that predictions have the right shape
        assert len(predictions) == len(test_features)
        
        # Check that predictions are reasonable (within the expected range)
        min_rating = 500
        max_rating = 3000
        assert predictions.min() >= min_rating, f"Prediction below minimum: {predictions.min()}"
        assert predictions.max() <= max_rating, f"Prediction above maximum: {predictions.max()}"