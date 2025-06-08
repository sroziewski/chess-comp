from setuptools import setup, find_packages

setup(
    name="chess_puzzle_rating",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "chess",
        "tqdm",
        "scikit-learn",
        "lightgbm",
        "torch",
        "pyyaml",
        "jsonschema",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "pytest-mock",
        ],
    },
    author="Chess Puzzle Rating Team",
    author_email="simon@simon",
    description="A package for predicting chess puzzle ratings",
    keywords="chess, puzzle, rating, prediction",
    python_requires=">=3.6",
)
