# Chess Puzzle Rating Prediction: Testing Infrastructure

This directory contains the testing infrastructure for the Chess Puzzle Rating Prediction project. The tests are organized by component and type, and they use pytest as the testing framework.

## Test Structure

The tests are organized into the following directories:

- `tests/features`: Tests for feature extraction functions
- `tests/models`: Tests for model components
- `tests/data`: Tests for data validation and pipeline
- `tests/integration`: Tests for end-to-end functionality
- `tests/cross_validation`: Tests for cross-validation strategies
- `tests/regression`: Tests for performance regression

## Running Tests

To run all tests, use the following command from the project root:

```bash
pytest tests/
```

To run tests for a specific component, use:

```bash
pytest tests/features/
pytest tests/models/
pytest tests/data/
pytest tests/integration/
pytest tests/cross_validation/
pytest tests/regression/
```

To run a specific test file, use:

```bash
pytest tests/features/test_position_features.py
```

To run a specific test function, use:

```bash
pytest tests/features/test_position_features.py::TestPawnStructureAnalysis::test_identify_pawn_chains_empty
```

## Test Fixtures

Common test fixtures are defined in `tests/conftest.py`. These include:

- `sample_puzzle_df`: A small sample DataFrame of chess puzzles for testing
- `sample_puzzle_df_with_timestamps`: A sample DataFrame with timestamps for time-based validation
- `sample_rating_ranges`: Sample rating ranges for stratified sampling
- `sample_feature_df`: A sample DataFrame of extracted features for testing

## Test Types

### Unit Tests

Unit tests verify that individual components work correctly in isolation. They test functions, classes, and methods with controlled inputs and expected outputs.

### Integration Tests

Integration tests verify that different components work together correctly. They test the interaction between components, such as the data pipeline, feature engineering, and model training.

### Cross-Validation Tests

Cross-validation tests verify that the cross-validation strategies work correctly. They test stratified sampling by rating range, time-based validation splits, and validation metrics for different rating ranges.

### Regression Tests

Regression tests detect performance degradation. They create baseline performance metrics, compare current performance with the baseline, and generate alerts for performance degradation.

## Adding New Tests

When adding new features or modifying existing ones, add or update the corresponding tests. Follow these guidelines:

1. Create a new test file in the appropriate directory
2. Use the existing test fixtures when possible
3. Test both normal cases and edge cases
4. Test for expected errors and exceptions
5. Keep tests independent and isolated
6. Use descriptive test names that explain what is being tested

## Continuous Integration

The tests are run automatically on every pull request and push to the main branch using GitHub Actions. The workflow is defined in `.github/workflows/tests.yml`.

## Test Coverage

To generate a test coverage report, use:

```bash
pytest --cov=chess_puzzle_rating tests/
```

To generate an HTML report, use:

```bash
pytest --cov=chess_puzzle_rating --cov-report=html tests/
```

The HTML report will be generated in the `htmlcov` directory.