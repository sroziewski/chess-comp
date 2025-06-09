"""
Progress tracking module for chess puzzle rating prediction.

This module provides utilities for logging, progress tracking, and performance monitoring.
It includes functions for setting up logging, tracking progress of long-running processes,
and creating performance dashboards.
"""

import os
import time
import logging
import datetime
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
import threading
from functools import wraps
import json
import numpy as np
from tqdm.auto import tqdm

# Try to import optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Global metrics storage
_metrics_store = {}
_metrics_lock = threading.Lock()


def setup_logging(log_dir: str = "logs", 
                 log_level: int = logging.INFO,
                 log_to_console: bool = True,
                 log_to_file: bool = True,
                 log_file_name: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for the application.

    Parameters
    ----------
    log_dir : str, optional
        Directory to store log files, by default "logs"
    log_level : int, optional
        Logging level, by default logging.INFO
    log_to_console : bool, optional
        Whether to log to console, by default True
    log_to_file : bool, optional
        Whether to log to file, by default True
    log_file_name : str, optional
        Name of the log file, by default None (auto-generated based on timestamp)

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('chess_puzzle_rating')
    logger.setLevel(log_level)

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if requested
    if log_to_file:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if log_file_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_name = f"chess_puzzle_rating_{timestamp}.log"

        file_handler = logging.FileHandler(os.path.join(log_dir, log_file_name))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """
    Get the configured logger instance.

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger('chess_puzzle_rating')

    # If logger has no handlers, set up a default console handler
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def log_time(func=None, *, name: Optional[str] = None, logger: Optional[logging.Logger] = None):
    """
    Decorator to log the execution time of a function.

    Parameters
    ----------
    func : callable, optional
        Function to decorate
    name : str, optional
        Name to use in the log message, by default None (uses function name)
    logger : logging.Logger, optional
        Logger to use, by default None (uses default logger)

    Returns
    -------
    callable
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger()

            func_name = name or func.__name__
            logger.info(f"Starting {func_name}...")
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed_time = end_time - start_time

                # Log the execution time
                logger.info(f"Completed {func_name} in {elapsed_time:.2f} seconds")

                # Store the metric
                with _metrics_lock:
                    if 'execution_times' not in _metrics_store:
                        _metrics_store['execution_times'] = {}

                    if func_name not in _metrics_store['execution_times']:
                        _metrics_store['execution_times'][func_name] = []

                    _metrics_store['execution_times'][func_name].append({
                        'timestamp': datetime.datetime.now().isoformat(),
                        'elapsed_time': elapsed_time
                    })

                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {str(e)}")
                raise

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def estimate_remaining_time(current_step: int, 
                           total_steps: int, 
                           elapsed_time: float) -> Tuple[float, str]:
    """
    Estimate the remaining time for a process.

    Parameters
    ----------
    current_step : int
        Current step number
    total_steps : int
        Total number of steps
    elapsed_time : float
        Elapsed time so far in seconds

    Returns
    -------
    Tuple[float, str]
        Estimated remaining time in seconds and formatted string
    """
    if current_step == 0:
        return float('inf'), "unknown"

    steps_per_second = current_step / elapsed_time
    remaining_steps = total_steps - current_step
    remaining_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else float('inf')

    # Format the remaining time
    if remaining_seconds == float('inf'):
        return float('inf'), "unknown"

    if remaining_seconds < 60:
        return remaining_seconds, f"{remaining_seconds:.1f} seconds"
    elif remaining_seconds < 3600:
        minutes = remaining_seconds / 60
        return remaining_seconds, f"{minutes:.1f} minutes"
    else:
        hours = remaining_seconds / 3600
        return remaining_seconds, f"{hours:.1f} hours"


class ProgressTracker:
    """
    Track progress of a long-running process with time estimates.
    """

    def __init__(self, 
                total: int, 
                description: str = "Progress", 
                logger: Optional[logging.Logger] = None,
                log_interval: int = 10,
                store_metrics: bool = True):
        """
        Initialize the progress tracker.

        Parameters
        ----------
        total : int
            Total number of steps
        description : str, optional
            Description of the process, by default "Progress"
        logger : logging.Logger, optional
            Logger to use, by default None (uses default logger)
        log_interval : int, optional
            Interval (in percentage) at which to log progress, by default 10
        store_metrics : bool, optional
            Whether to store metrics for dashboard, by default True
        """
        self.total = total
        self.description = description
        self.logger = logger or get_logger()
        self.log_interval = log_interval
        self.store_metrics = store_metrics

        self.start_time = time.time()
        self.last_log_percent = -1
        self.current = 0

        # Initialize tqdm progress bar
        self.progress_bar = tqdm(total=total, desc=description, unit="steps")

        # Log the start
        self.logger.info(f"Started {description} with {total} steps")

        # Initialize metrics
        self.metrics = {
            'name': description,
            'total_steps': total,
            'start_time': datetime.datetime.now().isoformat(),
            'updates': []
        }

    def update(self, steps: int = 1) -> None:
        """
        Update the progress.

        Parameters
        ----------
        steps : int, optional
            Number of steps completed, by default 1
        """
        self.current += steps
        self.progress_bar.update(steps)

        # Calculate progress percentage
        percent_complete = int((self.current / self.total) * 100)

        # Log at specified intervals
        if percent_complete >= self.last_log_percent + self.log_interval or self.current >= self.total:
            elapsed_time = time.time() - self.start_time
            _, remaining_time_str = estimate_remaining_time(self.current, self.total, elapsed_time)

            self.logger.info(
                f"{self.description}: {percent_complete}% complete ({self.current}/{self.total}), "
                f"elapsed: {elapsed_time:.1f}s, estimated remaining: {remaining_time_str}"
            )

            self.last_log_percent = percent_complete

            # Store metrics
            if self.store_metrics:
                update_data = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'current_step': self.current,
                    'percent_complete': percent_complete,
                    'elapsed_time': elapsed_time
                }
                self.metrics['updates'].append(update_data)

    def finish(self) -> Dict[str, Any]:
        """
        Finish tracking and return metrics.

        Returns
        -------
        Dict[str, Any]
            Metrics collected during tracking
        """
        self.progress_bar.close()

        end_time = time.time()
        total_time = end_time - self.start_time

        self.logger.info(f"Completed {self.description} in {total_time:.2f} seconds")

        # Store final metrics
        self.metrics['end_time'] = datetime.datetime.now().isoformat()
        self.metrics['total_time'] = total_time

        # Store in global metrics
        if self.store_metrics:
            with _metrics_lock:
                if 'progress_tracking' not in _metrics_store:
                    _metrics_store['progress_tracking'] = []

                _metrics_store['progress_tracking'].append(self.metrics)

        return self.metrics


def track_progress(iterable, description: str = "Progress", logger: Optional[logging.Logger] = None,
                  log_interval: int = 10, store_metrics: bool = True, total: Optional[int] = None):
    """
    Track progress of an iterable with time estimates.

    Parameters
    ----------
    iterable : iterable
        Iterable to track
    description : str, optional
        Description of the process, by default "Progress"
    logger : logging.Logger, optional
        Logger to use, by default None (uses default logger)
    log_interval : int, optional
        Interval (in percentage) at which to log progress, by default 10
    store_metrics : bool, optional
        Whether to store metrics for dashboard, by default True
    total : int, optional
        Total number of items in the iterable, by default None (will be determined automatically)

    Yields
    ------
    Any
        Items from the iterable
    """
    # Use provided total or get the length of the iterable if possible
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            # If we can't get the length, convert to list (may be inefficient for large iterables)
            iterable = list(iterable)
            total = len(iterable)

    tracker = ProgressTracker(
        total=total,
        description=description,
        logger=logger,
        log_interval=log_interval,
        store_metrics=store_metrics
    )

    for item in iterable:
        yield item
        tracker.update()

    tracker.finish()


def record_metric(name: str, value: Any, category: str = "custom") -> None:
    """
    Record a metric for the performance dashboard.

    Parameters
    ----------
    name : str
        Name of the metric
    value : Any
        Value of the metric
    category : str, optional
        Category of the metric, by default "custom"
    """
    with _metrics_lock:
        if category not in _metrics_store:
            _metrics_store[category] = {}

        if name not in _metrics_store[category]:
            _metrics_store[category][name] = []

        _metrics_store[category][name].append({
            'timestamp': datetime.datetime.now().isoformat(),
            'value': value
        })


def get_all_metrics() -> Dict[str, Any]:
    """
    Get all recorded metrics.

    Returns
    -------
    Dict[str, Any]
        All recorded metrics
    """
    with _metrics_lock:
        return _metrics_store.copy()


def save_metrics(file_path: str) -> None:
    """
    Save all metrics to a JSON file.

    Parameters
    ----------
    file_path : str
        Path to save the metrics
    """
    with _metrics_lock:
        metrics = _metrics_store.copy()

    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(file_path: str) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.

    Parameters
    ----------
    file_path : str
        Path to load the metrics from

    Returns
    -------
    Dict[str, Any]
        Loaded metrics
    """
    with open(file_path, 'r') as f:
        metrics = json.load(f)

    return metrics


def create_performance_dashboard(metrics: Optional[Dict[str, Any]] = None, 
                               output_dir: str = "dashboards",
                               dashboard_name: Optional[str] = None) -> str:
    """
    Create a performance dashboard from metrics.

    Parameters
    ----------
    metrics : Dict[str, Any], optional
        Metrics to use, by default None (uses global metrics)
    output_dir : str, optional
        Directory to save the dashboard, by default "dashboards"
    dashboard_name : str, optional
        Name of the dashboard, by default None (auto-generated based on timestamp)

    Returns
    -------
    str
        Path to the created dashboard
    """
    if not VISUALIZATION_AVAILABLE:
        raise ImportError("Visualization dependencies (matplotlib, seaborn) not available")

    if metrics is None:
        with _metrics_lock:
            metrics = _metrics_store.copy()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dashboard_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_name = f"performance_dashboard_{timestamp}"

    dashboard_path = os.path.join(output_dir, dashboard_name)
    os.makedirs(dashboard_path, exist_ok=True)

    # Create index.html
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>Chess Puzzle Rating Performance Dashboard</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; }",
        "        h1 { color: #333; }",
        "        .section { margin-bottom: 30px; }",
        "        .chart-container { margin-bottom: 20px; }",
        "        img { max-width: 100%; border: 1px solid #ddd; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>Chess Puzzle Rating Performance Dashboard</h1>",
        f"    <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
    ]

    # Process execution times
    if 'execution_times' in metrics:
        html_content.extend([
            "    <div class='section'>",
            "        <h2>Execution Times</h2>"
        ])

        # Create execution time chart
        plt.figure(figsize=(12, 6))

        for func_name, times in metrics['execution_times'].items():
            if times:
                # Extract timestamps and elapsed times
                timestamps = [datetime.datetime.fromisoformat(t['timestamp']) for t in times]
                elapsed_times = [t['elapsed_time'] for t in times]

                plt.plot(timestamps, elapsed_times, marker='o', label=func_name)

        plt.xlabel('Timestamp')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Function Execution Times')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the chart
        execution_times_chart = os.path.join(dashboard_path, 'execution_times.png')
        plt.savefig(execution_times_chart)
        plt.close()

        html_content.extend([
            "        <div class='chart-container'>",
            f"            <img src='{os.path.basename(execution_times_chart)}' alt='Execution Times Chart'>",
            "        </div>"
        ])

        # Create a table of average execution times
        html_content.extend([
            "        <h3>Average Execution Times</h3>",
            "        <table border='1' cellpadding='5'>",
            "            <tr><th>Function</th><th>Average Time (seconds)</th><th>Min Time (seconds)</th><th>Max Time (seconds)</th><th>Calls</th></tr>"
        ])

        for func_name, times in metrics['execution_times'].items():
            if times:
                elapsed_times = [t['elapsed_time'] for t in times]
                avg_time = sum(elapsed_times) / len(elapsed_times)
                min_time = min(elapsed_times)
                max_time = max(elapsed_times)
                calls = len(elapsed_times)

                html_content.append(
                    f"            <tr><td>{func_name}</td><td>{avg_time:.2f}</td><td>{min_time:.2f}</td><td>{max_time:.2f}</td><td>{calls}</td></tr>"
                )

        html_content.append("        </table>")
        html_content.append("    </div>")

    # Process progress tracking
    if 'progress_tracking' in metrics:
        html_content.extend([
            "    <div class='section'>",
            "        <h2>Progress Tracking</h2>"
        ])

        for i, progress in enumerate(metrics['progress_tracking']):
            if 'updates' in progress and progress['updates']:
                # Extract data
                timestamps = [datetime.datetime.fromisoformat(u['timestamp']) for u in progress['updates']]
                percent_complete = [u['percent_complete'] for u in progress['updates']]
                elapsed_times = [u['elapsed_time'] for u in progress['updates']]

                # Create progress chart
                plt.figure(figsize=(12, 6))

                plt.subplot(2, 1, 1)
                plt.plot(timestamps, percent_complete, marker='o', label='Percent Complete')
                plt.xlabel('Timestamp')
                plt.ylabel('Percent Complete')
                plt.title(f"Progress: {progress['name']}")
                plt.grid(True, linestyle='--', alpha=0.7)

                plt.subplot(2, 1, 2)
                plt.plot(timestamps, elapsed_times, marker='o', color='orange', label='Elapsed Time')
                plt.xlabel('Timestamp')
                plt.ylabel('Elapsed Time (seconds)')
                plt.grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()

                # Save the chart
                progress_chart = os.path.join(dashboard_path, f"progress_{i}.png")
                plt.savefig(progress_chart)
                plt.close()

                html_content.extend([
                    "        <div class='chart-container'>",
                    f"            <h3>{progress['name']}</h3>",
                    f"            <p>Total Time: {progress.get('total_time', 'N/A'):.2f} seconds</p>",
                    f"            <img src='{os.path.basename(progress_chart)}' alt='Progress Chart'>",
                    "        </div>"
                ])

        html_content.append("    </div>")

    # Process custom metrics
    custom_categories = [cat for cat in metrics.keys() if cat not in ['execution_times', 'progress_tracking']]

    if custom_categories:
        html_content.extend([
            "    <div class='section'>",
            "        <h2>Custom Metrics</h2>"
        ])

        for category in custom_categories:
            html_content.extend([
                f"        <h3>{category}</h3>"
            ])

            for metric_name, values in metrics[category].items():
                if values:
                    # Extract timestamps and values
                    try:
                        timestamps = [datetime.datetime.fromisoformat(v['timestamp']) for v in values]
                        metric_values = [float(v['value']) for v in values]

                        # Create metric chart
                        plt.figure(figsize=(12, 6))
                        plt.plot(timestamps, metric_values, marker='o')
                        plt.xlabel('Timestamp')
                        plt.ylabel('Value')
                        plt.title(f"{metric_name}")
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.tight_layout()

                        # Save the chart
                        metric_chart = os.path.join(dashboard_path, f"{category}_{metric_name}.png")
                        plt.savefig(metric_chart)
                        plt.close()

                        html_content.extend([
                            "        <div class='chart-container'>",
                            f"            <h4>{metric_name}</h4>",
                            f"            <img src='{os.path.basename(metric_chart)}' alt='{metric_name} Chart'>",
                            "        </div>"
                        ])
                    except (ValueError, TypeError):
                        # Skip metrics that can't be plotted
                        pass

        html_content.append("    </div>")

    # Finish HTML
    html_content.extend([
        "</body>",
        "</html>"
    ])

    # Write HTML file
    with open(os.path.join(dashboard_path, 'index.html'), 'w') as f:
        f.write('\n'.join(html_content))

    # Save metrics as JSON
    with open(os.path.join(dashboard_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return dashboard_path
