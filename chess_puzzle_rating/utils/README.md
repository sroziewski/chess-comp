# Progress Tracking and Logging Utilities

This directory contains utility modules for the Chess Puzzle Rating Prediction project, including a comprehensive progress tracking and logging system.

## Progress Tracking Module

The `progress.py` module provides utilities for logging, progress tracking, and performance monitoring. It includes:

### Logging Utilities

- `setup_logging()`: Set up logging for the application
- `get_logger()`: Get the configured logger instance
- `log_time()`: Decorator to log the execution time of a function

### Progress Tracking Utilities

- `estimate_remaining_time()`: Estimate the remaining time for a process
- `ProgressTracker`: Class to track progress of a long-running process with time estimates
- `track_progress()`: Function to track progress of an iterable with time estimates

### Performance Monitoring Utilities

- `record_metric()`: Record a metric for the performance dashboard
- `get_all_metrics()`: Get all recorded metrics
- `save_metrics()`: Save all metrics to a JSON file
- `load_metrics()`: Load metrics from a JSON file
- `create_performance_dashboard()`: Create a performance dashboard from metrics

## Configuration

The progress tracking system is configured in the main `config.yaml` file with the following sections:

```yaml
# Logging configuration
logging:
  # Directory to store log files
  log_dir: "logs"
  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  log_level: "INFO"
  # Whether to log to console
  log_to_console: true
  # Whether to log to file
  log_to_file: true
  # Name of the log file (null = auto-generated based on timestamp)
  log_file_name: null

# Progress tracking configuration
progress_tracking:
  # Enable progress tracking
  enabled: true
  # Interval (in percentage) at which to log progress
  log_interval: 10
  # Whether to store metrics for dashboard
  store_metrics: true

# Performance dashboards configuration
dashboards:
  # Enable performance dashboards
  enabled: true
  # Directory to save dashboards
  output_dir: "dashboards"
  # Whether to automatically generate dashboards at the end of training
  auto_generate: true
```

## Usage Examples

### Basic Logging

```python
from chess_puzzle_rating.utils.progress import get_logger

logger = get_logger()
logger.info("Starting process...")
logger.warning("Warning message")
logger.error("Error message")
```

### Timing Function Execution

```python
from chess_puzzle_rating.utils.progress import log_time

@log_time
def my_function():
    # Function code here
    pass
```

### Tracking Progress

```python
from chess_puzzle_rating.utils.progress import track_progress

for item in track_progress(items, description="Processing items"):
    # Process item
    pass
```

### Recording Metrics

```python
from chess_puzzle_rating.utils.progress import record_metric

record_metric("accuracy", 0.95, "model_performance")
record_metric("training_time", 120.5, "performance")
```

### Creating Performance Dashboard

```python
from chess_puzzle_rating.utils.progress import create_performance_dashboard

dashboard_path = create_performance_dashboard(output_dir="dashboards")
print(f"Dashboard created at: {dashboard_path}")
```

## Dashboard Features

The performance dashboard includes:

1. **Execution Times**: Charts and tables showing execution times of different functions
2. **Progress Tracking**: Charts showing progress over time for long-running processes
3. **Custom Metrics**: Charts for any custom metrics recorded during execution

Dashboards are saved as HTML files with accompanying JSON data and can be viewed in any web browser.