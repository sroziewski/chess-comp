"""
Data pipeline module for chess puzzle rating prediction.

This module provides a unified pipeline for data loading, preprocessing,
validation, and feature engineering with checkpoints for intermediate results.
"""

import os
import pandas as pd
import numpy as np
import json
import hashlib
import time
import datetime
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from tqdm import tqdm
from pathlib import Path

from ..utils.config import get_config
from ..utils.progress import (
    setup_logging, get_logger, log_time, ProgressTracker, 
    track_progress, record_metric, create_performance_dashboard
)
from ..features.pipeline import complete_feature_engineering


class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


class ChessPuzzleDataPipeline:
    """
    A unified data pipeline for chess puzzle rating prediction.

    This class handles data loading, preprocessing, validation, and feature engineering
    with checkpoints for intermediate results.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data pipeline.

        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file. If None, use default configuration.
        """
        self.config = get_config(config_path)

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = self.config['pipeline']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Set up logging
        logging_config = self.config.get('logging', {})
        self.logger = setup_logging(
            log_dir=logging_config.get('log_dir', 'logs'),
            log_level=getattr(logging, logging_config.get('log_level', 'INFO')),
            log_to_console=logging_config.get('log_to_console', True),
            log_to_file=logging_config.get('log_to_file', True),
            log_file_name=logging_config.get('log_file_name')
        )

        # Get progress tracking configuration
        self.progress_config = self.config.get('progress_tracking', {})
        self.enable_progress_tracking = self.progress_config.get('enabled', True)
        self.log_interval = self.progress_config.get('log_interval', 10)
        self.store_metrics = self.progress_config.get('store_metrics', True)

    def _generate_checkpoint_id(self, data: pd.DataFrame, stage: str) -> str:
        """
        Generate a unique identifier for a checkpoint based on data and stage.

        Parameters
        ----------
        data : pandas.DataFrame
            The data to generate a checkpoint ID for
        stage : str
            The pipeline stage name

        Returns
        -------
        str
            A unique checkpoint ID
        """
        # Create a hash based on the data shape, column names, and first/last few values
        data_hash = hashlib.md5()
        data_hash.update(str(data.shape).encode())
        data_hash.update(str(list(data.columns)).encode())

        # Add sample of data to hash
        if len(data) > 0:
            data_hash.update(str(data.iloc[0:min(5, len(data))]).encode())
            if len(data) > 5:
                data_hash.update(str(data.iloc[-min(5, len(data)):]).encode())

        return f"{stage}_{data_hash.hexdigest()}"

    def _save_checkpoint(self, data: pd.DataFrame, stage: str) -> str:
        """
        Save a checkpoint of the data at a specific pipeline stage.

        Parameters
        ----------
        data : pandas.DataFrame
            The data to save
        stage : str
            The pipeline stage name

        Returns
        -------
        str
            The path to the saved checkpoint
        """
        checkpoint_id = self._generate_checkpoint_id(data, stage)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.parquet")

        # Save the data
        data.to_parquet(checkpoint_path, index=False)

        # Save metadata
        metadata = {
            'stage': stage,
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'timestamp': pd.Timestamp.now().isoformat()
        }

        metadata_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def save_dataframe_for_analysis(self, df: pd.DataFrame, name: str, include_timestamp: bool = True) -> Dict[str, str]:
        """
        Save a dataframe to multiple formats for analysis purposes.

        This method saves the dataframe to CSV and parquet formats with an optional timestamp
        in the filename, making it easy to identify and analyze later.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe to save
        name : str
            A descriptive name for the dataframe
        include_timestamp : bool, optional
            Whether to include a timestamp in the filename, by default True

        Returns
        -------
        Dict[str, str]
            A dictionary with the paths to the saved files
        """
        # Create analysis directory if it doesn't exist
        analysis_dir = Path("analysis")
        analysis_dir.mkdir(exist_ok=True)

        # Generate filename with optional timestamp
        timestamp = ""
        if include_timestamp:
            timestamp = f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        base_filename = f"{name}{timestamp}"

        # Save to multiple formats
        csv_path = analysis_dir / f"{base_filename}.csv"
        parquet_path = analysis_dir / f"{base_filename}.parquet"

        # Save the dataframe
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)

        # Save metadata
        metadata = {
            'name': name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'timestamp': pd.Timestamp.now().isoformat(),
            'saved_formats': ['csv', 'parquet'],
            'file_paths': {
                'csv': str(csv_path),
                'parquet': str(parquet_path)
            }
        }

        metadata_path = analysis_dir / f"{base_filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Dataframe saved for analysis: {name}")
        self.logger.info(f"  CSV: {csv_path}")
        self.logger.info(f"  Parquet: {parquet_path}")
        self.logger.info(f"  Metadata: {metadata_path}")

        return {
            'csv': str(csv_path),
            'parquet': str(parquet_path),
            'metadata': str(metadata_path)
        }

    def _load_checkpoint(self, checkpoint_id: str) -> Optional[pd.DataFrame]:
        """
        Load a checkpoint from disk.

        Parameters
        ----------
        checkpoint_id : str
            The checkpoint ID to load

        Returns
        -------
        pandas.DataFrame or None
            The loaded data, or None if checkpoint doesn't exist
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.parquet")
        if os.path.exists(checkpoint_path):
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            return pd.read_parquet(checkpoint_path)
        return None

    def _validate_data(self, data: pd.DataFrame, is_train: bool = True) -> None:
        """
        Validate the data for common issues.

        Parameters
        ----------
        data : pandas.DataFrame
            The data to validate
        is_train : bool, optional
            Whether this is training data (True) or test data (False)

        Raises
        ------
        DataValidationError
            If validation fails
        """
        validation_config = self.config['pipeline']['validation']

        # Check required columns
        required_cols = validation_config['required_columns']
        if is_train and 'Rating' not in required_cols:
            required_cols.append('Rating')

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

        # Check for too many missing values
        max_missing_pct = validation_config['max_missing_values_pct']
        missing_pct = data[required_cols].isnull().mean()
        cols_with_too_many_missing = missing_pct[missing_pct > max_missing_pct].index.tolist()
        if cols_with_too_many_missing:
            raise DataValidationError(
                f"Columns with too many missing values (>{max_missing_pct*100}%): "
                f"{cols_with_too_many_missing}"
            )

        # Check rating range for training data
        if is_train and 'Rating' in data.columns:
            rating_range = validation_config['rating_range']
            out_of_range = data[
                (data['Rating'] < rating_range[0]) | 
                (data['Rating'] > rating_range[1])
            ]
            if len(out_of_range) > 0:
                self.logger.warning(f"{len(out_of_range)} ratings out of expected range "
                      f"[{rating_range[0]}, {rating_range[1]}]")

        self.logger.info("Data validation passed.")

    @log_time
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data from configured paths.

        Returns
        -------
        Tuple[pandas.DataFrame, pandas.DataFrame]
            Training and test dataframes
        """
        train_path = self.config['data_paths']['train_file']
        test_path = self.config['data_paths']['test_file']

        self.logger.info(f"Loading training data from: {train_path}")
        start_time = time.time()
        train_df = pd.read_csv(train_path)
        elapsed_time = time.time() - start_time
        self.logger.info(f"Training data loaded in {elapsed_time:.2f} seconds. Shape: {train_df.shape}")
        record_metric("train_data_load_time", elapsed_time, "data_loading")

        self.logger.info(f"Loading test data from: {test_path}")
        start_time = time.time()
        test_df = pd.read_csv(test_path)
        elapsed_time = time.time() - start_time
        self.logger.info(f"Test data loaded in {elapsed_time:.2f} seconds. Shape: {test_df.shape}")
        record_metric("test_data_load_time", elapsed_time, "data_loading")

        # Add dataset identifier
        train_df['is_train'] = 1
        test_df['is_train'] = 0

        # Validate data
        self.logger.info("Validating training data...")
        self._validate_data(train_df, is_train=True)

        self.logger.info("Validating test data...")
        self._validate_data(test_df, is_train=False)

        # Save checkpoints
        self._save_checkpoint(train_df, 'load_train')
        self._save_checkpoint(test_df, 'load_test')

        # Record metrics
        record_metric("train_data_rows", len(train_df), "data_stats")
        record_metric("train_data_columns", len(train_df.columns), "data_stats")
        record_metric("test_data_rows", len(test_df), "data_stats")
        record_metric("test_data_columns", len(test_df.columns), "data_stats")

        return train_df, test_df

    @log_time
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess and combine training and test data.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training data
        test_df : pandas.DataFrame
            Test data

        Returns
        -------
        pandas.DataFrame
            Combined preprocessed data
        """
        self.logger.info("Preprocessing data...")

        # Combine datasets
        start_time = time.time()
        combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
        elapsed_time = time.time() - start_time
        self.logger.info(f"Datasets combined in {elapsed_time:.2f} seconds. Combined shape: {combined_df.shape}")

        # Handle missing values in key columns
        missing_values_stats = {}
        for col in ['FEN', 'Moves']:
            if col in combined_df.columns:
                missing = combined_df[col].isnull().sum()
                missing_values_stats[col] = missing
                if missing > 0:
                    self.logger.warning(f"{missing} missing values in {col}")
                    if col == 'FEN':
                        # Can't impute FEN strings, so we'll use a placeholder
                        combined_df[col] = combined_df[col].fillna('8/8/8/8/8/8/8/8 w - - 0 1')
                    elif col == 'Moves':
                        # Empty moves
                        combined_df[col] = combined_df[col].fillna('')

        # Handle missing values in optional columns
        if 'Themes' in combined_df.columns:
            missing = combined_df['Themes'].isnull().sum()
            missing_values_stats['Themes'] = missing
            combined_df['Themes'] = combined_df['Themes'].fillna('')

        if 'OpeningTags' in combined_df.columns:
            missing = combined_df['OpeningTags'].isnull().sum()
            missing_values_stats['OpeningTags'] = missing
            combined_df['OpeningTags'] = combined_df['OpeningTags'].fillna('')

        # Record missing values metrics
        for col, count in missing_values_stats.items():
            record_metric(f"missing_values_{col}", count, "data_quality")
            record_metric(f"missing_values_pct_{col}", count / len(combined_df) * 100, "data_quality")

        # Save checkpoint
        self._save_checkpoint(combined_df, 'preprocess')

        return combined_df

    @log_time
    def engineer_features(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to the combined data.

        Parameters
        ----------
        combined_df : pandas.DataFrame
            Combined preprocessed data

        Returns
        -------
        pandas.DataFrame
            Data with engineered features
        """
        self.logger.info("Engineering features...")

        # Generate a checkpoint ID for the features
        features_checkpoint_id = self._generate_checkpoint_id(combined_df, 'features')
        features_checkpoint_path = os.path.join(self.checkpoint_dir, f"{features_checkpoint_id}.parquet")

        # Check if features checkpoint exists
        if os.path.exists(features_checkpoint_path):
            self.logger.info(f"Loading existing features from checkpoint: {features_checkpoint_path}")
            features_df = pd.read_parquet(features_checkpoint_path)

            # Try to load predictions if they exist
            predictions_checkpoint_id = self._generate_checkpoint_id(combined_df, 'tag_predictions')
            predictions_checkpoint_path = os.path.join(self.checkpoint_dir, f"{predictions_checkpoint_id}.parquet")

            if os.path.exists(predictions_checkpoint_path):
                predictions = pd.read_parquet(predictions_checkpoint_path)
                self.logger.info(f"Loaded existing tag predictions from checkpoint. Shape: {predictions.shape}")

                # Record metrics about tag predictions
                if 'prediction_confidence' in predictions.columns:
                    avg_confidence = predictions['prediction_confidence'].mean()
                    record_metric("avg_tag_prediction_confidence", avg_confidence, "feature_stats")
                    self.logger.info(f"Average tag prediction confidence: {avg_confidence:.4f}")
            else:
                predictions = None
                self.logger.info("No existing tag predictions found in checkpoint.")

            self.logger.info(f"Loaded features from checkpoint. Shape: {features_df.shape}")
            record_metric("num_features", features_df.shape[1], "feature_stats")
            record_metric("features_loaded_from_checkpoint", 1, "performance")

            # Save the loaded features dataframe for analysis
            self.save_dataframe_for_analysis(features_df, 'features_from_checkpoint')

            # If predictions are available, save them for analysis too
            if predictions is not None:
                self.save_dataframe_for_analysis(predictions, 'tag_predictions_from_checkpoint')

            return features_df

        # If no checkpoint exists, perform feature engineering
        self.logger.info("No existing features checkpoint found. Performing feature engineering...")

        # Record start time for estimating total time
        start_time = time.time()

        # Use the existing feature engineering pipeline
        features_df, model, predictions = complete_feature_engineering(
            combined_df, 
            tag_column='OpeningTags' if 'OpeningTags' in combined_df.columns else None
        )

        # Calculate elapsed time and record metrics
        elapsed_time = time.time() - start_time
        self.logger.info(f"Feature engineering completed in {elapsed_time:.2f} seconds. Features shape: {features_df.shape}")
        record_metric("feature_engineering_time", elapsed_time, "performance")
        record_metric("num_features", features_df.shape[1], "feature_stats")

        # Save checkpoint for the features
        self._save_checkpoint(features_df, 'features')

        # Save checkpoint for the predictions (if available)
        if predictions is not None:
            self._save_checkpoint(predictions, 'tag_predictions')
            self.logger.info(f"Opening tag predictions saved. Shape: {predictions.shape}")

            # Record metrics about tag predictions
            if 'prediction_confidence' in predictions.columns:
                avg_confidence = predictions['prediction_confidence'].mean()
                record_metric("avg_tag_prediction_confidence", avg_confidence, "feature_stats")
                self.logger.info(f"Average tag prediction confidence: {avg_confidence:.4f}")

        # Save the features dataframe for analysis
        self.save_dataframe_for_analysis(features_df, 'features_after_engineering')

        # If predictions are available, save them for analysis too
        if predictions is not None:
            self.save_dataframe_for_analysis(predictions, 'tag_predictions')

        return features_df

    @log_time
    def prepare_train_test_split(self, features_df: pd.DataFrame, combined_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare the final training and test datasets.

        Parameters
        ----------
        features_df : pandas.DataFrame
            Data with engineered features
        combined_df : pandas.DataFrame
            Original combined data with is_train indicator

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            X_train, X_test, y_train, test_ids
        """
        self.logger.info("Preparing train/test split...")

        # Get the is_train indicator and puzzle IDs
        is_train = combined_df['is_train'].values
        puzzle_ids = combined_df['PuzzleId'].values

        # Split features into train and test
        start_time = time.time()
        train_features = features_df[is_train == 1].copy()
        test_features = features_df[is_train == 0].copy()

        # Get the target variable for training
        y_train = combined_df.loc[is_train == 1, 'Rating'].copy()

        # Get test IDs
        test_ids = combined_df.loc[is_train == 0, 'PuzzleId'].copy()

        # Fill any remaining NaN values
        train_features = train_features.fillna(0)
        test_features = test_features.fillna(0)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Train/test split completed in {elapsed_time:.2f} seconds.")
        self.logger.info(f"Train features shape: {train_features.shape}, Test features shape: {test_features.shape}")
        self.logger.info(f"Target variable shape: {y_train.shape}, Test IDs shape: {test_ids.shape}")

        # Record metrics
        record_metric("train_test_split_time", elapsed_time, "performance")
        record_metric("train_features_rows", train_features.shape[0], "data_stats")
        record_metric("train_features_columns", train_features.shape[1], "data_stats")
        record_metric("test_features_rows", test_features.shape[0], "data_stats")
        record_metric("test_features_columns", test_features.shape[1], "data_stats")

        # Calculate and record target variable statistics
        if len(y_train) > 0:
            record_metric("target_min", y_train.min(), "target_stats")
            record_metric("target_max", y_train.max(), "target_stats")
            record_metric("target_mean", y_train.mean(), "target_stats")
            record_metric("target_median", y_train.median(), "target_stats")
            record_metric("target_std", y_train.std(), "target_stats")

            self.logger.info(f"Target variable statistics: min={y_train.min()}, max={y_train.max()}, "
                           f"mean={y_train.mean():.2f}, median={y_train.median()}, std={y_train.std():.2f}")

        # Save checkpoints
        self._save_checkpoint(train_features, 'train_features')
        self._save_checkpoint(test_features, 'test_features')
        self._save_checkpoint(pd.DataFrame({'Rating': y_train}), 'train_target')
        self._save_checkpoint(pd.DataFrame({'PuzzleId': test_ids}), 'test_ids')

        return train_features, test_features, y_train, test_ids

    @log_time(name="complete_data_pipeline")
    def run_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the complete data pipeline.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            X_train, X_test, y_train, test_ids
        """
        self.logger.info("Running complete data pipeline...")

        # Record overall start time
        pipeline_start_time = time.time()

        # Step 1: Load data
        self.logger.info("Step 1/4: Loading data")
        train_df, test_df = self.load_data()

        # Step 2: Preprocess data
        self.logger.info("Step 2/4: Preprocessing data")
        combined_df = self.preprocess_data(train_df, test_df)

        # Step 3: Engineer features
        self.logger.info("Step 3/4: Engineering features")
        features_df = self.engineer_features(combined_df)

        # Step 4: Prepare train/test split
        self.logger.info("Step 4/4: Preparing train/test split")
        X_train, X_test, y_train, test_ids = self.prepare_train_test_split(features_df, combined_df)

        # Calculate total pipeline time
        pipeline_time = time.time() - pipeline_start_time
        self.logger.info(f"Data pipeline completed successfully in {pipeline_time:.2f} seconds.")

        # Record final metrics
        record_metric("total_pipeline_time", pipeline_time, "performance")

        # Generate performance dashboard if enabled
        dashboard_config = self.config.get('dashboards', {})
        if dashboard_config.get('enabled', True) and dashboard_config.get('auto_generate', True):
            try:
                output_dir = dashboard_config.get('output_dir', 'dashboards')
                dashboard_path = create_performance_dashboard(output_dir=output_dir)
                self.logger.info(f"Performance dashboard created at: {dashboard_path}")
            except Exception as e:
                self.logger.error(f"Error creating performance dashboard: {str(e)}")

        return X_train, X_test, y_train, test_ids


@log_time(name="run_data_pipeline")
def run_data_pipeline(config_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the complete data pipeline with the given configuration.

    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file. If None, use default configuration.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        X_train, X_test, y_train, test_ids
    """
    # Initialize logging
    config = get_config(config_path)
    logging_config = config.get('logging', {})
    logger = setup_logging(
        log_dir=logging_config.get('log_dir', 'logs'),
        log_level=getattr(logging, logging_config.get('log_level', 'INFO')),
        log_to_console=logging_config.get('log_to_console', True),
        log_to_file=logging_config.get('log_to_file', True),
        log_file_name=logging_config.get('log_file_name')
    )

    logger.info("Starting data pipeline run")

    # Create and run the pipeline
    pipeline = ChessPuzzleDataPipeline(config_path)
    result = pipeline.run_pipeline()

    logger.info("Data pipeline run completed")
    return result


if __name__ == "__main__":
    # Example usage with progress tracking
    logger = get_logger()
    logger.info("Starting example data pipeline run")

    try:
        start_time = time.time()
        X_train, X_test, y_train, test_ids = run_data_pipeline()
        total_time = time.time() - start_time

        logger.info(f"Data pipeline completed in {total_time:.2f} seconds")
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"test_ids shape: {test_ids.shape}")

        # Generate a final performance dashboard
        dashboard_config = get_config().get('dashboards', {})
        if dashboard_config.get('enabled', True):
            output_dir = dashboard_config.get('output_dir', 'dashboards')
            dashboard_name = f"final_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dashboard_path = create_performance_dashboard(output_dir=output_dir, dashboard_name=dashboard_name)
            logger.info(f"Final performance dashboard created at: {dashboard_path}")

    except Exception as e:
        logger.error(f"Error running data pipeline: {str(e)}", exc_info=True)
        raise
