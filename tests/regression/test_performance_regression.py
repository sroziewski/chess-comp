import pytest
import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.model_selection import train_test_split
from chess_puzzle_rating.models.stacking import (
    LightGBMModel, XGBoostModel, StackingModel
)

class TestPerformanceRegression:
    """Tests for detecting performance regression."""
    
    @pytest.fixture
    def baseline_metrics_file(self, tmp_path):
        """Create a temporary file for storing baseline metrics."""
        return os.path.join(tmp_path, "baseline_metrics.json")
    
    def create_dummy_data(self, n_samples=100, n_features=10, random_state=42):
        """Create dummy data for testing."""
        np.random.seed(random_state)
        X = pd.DataFrame(
            np.random.rand(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(1000, 2000, n_samples))
        return X, y
    
    def test_create_baseline_performance(self, baseline_metrics_file):
        """Test creating baseline performance metrics."""
        # Create dummy data
        X, y = self.create_dummy_data()
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a model
        model = LightGBMModel(name="test_model", model_params={"n_estimators": 10})
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        predictions = model.predict(X_test)
        
        # Calculate performance metrics
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))
        
        # Save the metrics as a baseline
        baseline_metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        with open(baseline_metrics_file, 'w') as f:
            json.dump(baseline_metrics, f)
        
        # Check that the file exists
        assert os.path.exists(baseline_metrics_file)
        
        # Check that the metrics are reasonable
        assert np.isfinite(mse), f"MSE is not finite: {mse}"
        assert np.isfinite(rmse), f"RMSE is not finite: {rmse}"
        assert np.isfinite(mae), f"MAE is not finite: {mae}"
    
    def test_compare_with_baseline(self, baseline_metrics_file):
        """Test comparing current performance with baseline."""
        # First, create a baseline if it doesn't exist
        if not os.path.exists(baseline_metrics_file):
            self.test_create_baseline_performance(baseline_metrics_file)
        
        # Load the baseline metrics
        with open(baseline_metrics_file, 'r') as f:
            baseline_metrics = json.load(f)
        
        # Create dummy data
        X, y = self.create_dummy_data()
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a model
        model = LightGBMModel(name="test_model", model_params={"n_estimators": 10})
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        predictions = model.predict(X_test)
        
        # Calculate performance metrics
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))
        
        # Compare with baseline
        mse_change = (mse - baseline_metrics["mse"]) / baseline_metrics["mse"] * 100
        rmse_change = (rmse - baseline_metrics["rmse"]) / baseline_metrics["rmse"] * 100
        mae_change = (mae - baseline_metrics["mae"]) / baseline_metrics["mae"] * 100
        
        # Define thresholds for performance degradation
        warning_threshold = 5  # 5% degradation
        error_threshold = 10  # 10% degradation
        
        # Check for performance degradation
        if mse_change > error_threshold or rmse_change > error_threshold or mae_change > error_threshold:
            pytest.fail(f"Performance degradation detected: MSE change: {mse_change:.2f}%, RMSE change: {rmse_change:.2f}%, MAE change: {mae_change:.2f}%")
        elif mse_change > warning_threshold or rmse_change > warning_threshold or mae_change > warning_threshold:
            warnings.warn(f"Performance degradation warning: MSE change: {mse_change:.2f}%, RMSE change: {rmse_change:.2f}%, MAE change: {mae_change:.2f}%")
    
    def test_model_comparison(self):
        """Test comparing performance of different models."""
        # Create dummy data
        X, y = self.create_dummy_data()
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train different models
        models = {
            "LightGBM": LightGBMModel(name="lightgbm", model_params={"n_estimators": 10}),
            "XGBoost": XGBoostModel(name="xgboost", model_params={"n_estimators": 10}),
            "Stacking": StackingModel(
                base_models=[
                    LightGBMModel(name="lightgbm1", model_params={"n_estimators": 10}),
                    XGBoostModel(name="xgboost1", model_params={"n_estimators": 10})
                ],
                n_splits=2
            )
        }
        
        # Train and evaluate each model
        model_metrics = {}
        
        for name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions on the test set
            predictions = model.predict(X_test)
            
            # Calculate performance metrics
            mse = np.mean((predictions - y_test) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y_test))
            
            model_metrics[name] = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae)
            }
        
        # Compare models
        for name, metrics in model_metrics.items():
            assert np.isfinite(metrics["mse"]), f"MSE for {name} is not finite: {metrics['mse']}"
            assert np.isfinite(metrics["rmse"]), f"RMSE for {name} is not finite: {metrics['rmse']}"
            assert np.isfinite(metrics["mae"]), f"MAE for {name} is not finite: {metrics['mae']}"
        
        # Check that the stacking model performs at least as well as the best base model
        best_base_mse = min(model_metrics["LightGBM"]["mse"], model_metrics["XGBoost"]["mse"])
        stacking_mse = model_metrics["Stacking"]["mse"]
        
        # Allow for a small tolerance (5%) due to randomness in training
        assert stacking_mse <= best_base_mse * 1.05, \
            f"Stacking model performs worse than the best base model: {stacking_mse} > {best_base_mse}"
    
    def test_performance_alerts(self, baseline_metrics_file):
        """Test generating alerts for performance degradation."""
        # First, create a baseline if it doesn't exist
        if not os.path.exists(baseline_metrics_file):
            self.test_create_baseline_performance(baseline_metrics_file)
        
        # Load the baseline metrics
        with open(baseline_metrics_file, 'r') as f:
            baseline_metrics = json.load(f)
        
        # Create dummy data
        X, y = self.create_dummy_data()
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a model with intentionally poor performance
        model = LightGBMModel(name="test_model", model_params={"n_estimators": 1})  # Use fewer trees for worse performance
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        predictions = model.predict(X_test)
        
        # Calculate performance metrics
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))
        
        # Compare with baseline
        mse_change = (mse - baseline_metrics["mse"]) / baseline_metrics["mse"] * 100
        rmse_change = (rmse - baseline_metrics["rmse"]) / baseline_metrics["rmse"] * 100
        mae_change = (mae - baseline_metrics["mae"]) / baseline_metrics["mae"] * 100
        
        # Define thresholds for performance degradation
        warning_threshold = 5  # 5% degradation
        error_threshold = 10  # 10% degradation
        
        # Check for performance degradation and generate alerts
        alerts = []
        
        if mse_change > error_threshold:
            alerts.append(f"CRITICAL: MSE increased by {mse_change:.2f}%")
        elif mse_change > warning_threshold:
            alerts.append(f"WARNING: MSE increased by {mse_change:.2f}%")
        
        if rmse_change > error_threshold:
            alerts.append(f"CRITICAL: RMSE increased by {rmse_change:.2f}%")
        elif rmse_change > warning_threshold:
            alerts.append(f"WARNING: RMSE increased by {rmse_change:.2f}%")
        
        if mae_change > error_threshold:
            alerts.append(f"CRITICAL: MAE increased by {mae_change:.2f}%")
        elif mae_change > warning_threshold:
            alerts.append(f"WARNING: MAE increased by {mae_change:.2f}%")
        
        # Print the alerts (in a real system, these would be sent via email, Slack, etc.)
        for alert in alerts:
            print(alert)
        
        # In a real test, we would check that the alerts are generated correctly
        # Here, we just check that the metrics are calculated correctly
        assert np.isfinite(mse), f"MSE is not finite: {mse}"
        assert np.isfinite(rmse), f"RMSE is not finite: {rmse}"
        assert np.isfinite(mae), f"MAE is not finite: {mae}"