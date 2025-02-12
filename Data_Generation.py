import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples per fruit
n_samples = 100

# Generate data for Apples
apples = pd.DataFrame({
    'width': np.random.normal(7.5, 0.5, n_samples),
    'height': np.random.normal(7.5, 0.5, n_samples),
    'color': np.random.normal(0.1, 0.05, n_samples),
    'label': ['apple'] * n_samples
})

# Generate data for Oranges
oranges = pd.DataFrame({
    'width': np.random.normal(7.5, 0.5, n_samples),
    'height': np.random.normal(7.5, 0.5, n_samples),
    'color': np.random.normal(0.3, 0.05, n_samples),
    'label': ['orange'] * n_samples
})

# Generate data for Bananas
bananas = pd.DataFrame({
    'width': np.random.normal(3.0, 0.3, n_samples),
    'height': np.random.normal(17.0, 1.0, n_samples),
    'color': np.random.normal(0.6, 0.05, n_samples),
    'label': ['banana'] * n_samples
})

# Combine datasets
data = pd.concat([apples, oranges, bananas], ignore_index=True)
data.to_csv('fruits.csv', index=False)
