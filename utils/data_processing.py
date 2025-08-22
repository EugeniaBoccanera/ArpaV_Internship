"""
Data processing utilities for meteorological analysis
"""
import xarray as xr
import numpy as np
import gc
from sklearn.decomposition import IncrementalPCA

############################################## preparing the data

def prepare_data_matrix(dataset):
    """
    Converts xarray dataset 4D to 2D matrix
    """
    data_matrices = {} # dictionary to save individual variable matrices

    # Loop over the variables
    for var in dataset.data_vars:
        print(f"   • Processing {var}...")
        var_data = dataset[var]   # var_data.dims = ('time', 'isobaricInhPa', 'latitude', 'longitude')
        
        # Reorganize dimensions: (time, features)
        if 'time' in var_data.dims:
            # Stack all non-temporal dimensions
            spatial_dims = [dim for dim in var_data.dims if dim != 'time']

            if spatial_dims:  # If there are spatial dimensions to stack
                stacked = var_data.stack(features=spatial_dims)       # From shape: (time=1827, pressure=3, lat=201, lon=321)
                # conversion to a numpy array
                matrix = stacked.values        # To:  shape(time=1827, features=193563)  (3×201×321=193563) for each variable

        data_matrices[var] = matrix
    
    # Concatenate all variables
    all_matrices = list(data_matrices.values())
    combined_matrix = np.concatenate(all_matrices, axis=1) # concatenate along the columns (horizontally)

    return combined_matrix, data_matrices


##################################################### global standardization

def apply_global_standardization(X):
    """
    Applies global standardization to the data matrix
    """
    global_mean = X.mean()    # Compute the global mean
    global_std = X.std()      # Compute the global standard deviation
    X_standardized = (X - global_mean) / global_std
    
    return X_standardized, global_mean, global_std

#################################################### incremental PCA

def perform_incremental_pca(X, n_components=30, batch_size=100):
    """
    Performs Incremental PCA on the input data matrix
    """

    # Initialize IncrementalPCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # Fit the PCA incrementally
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i+batch_size]
        ipca.partial_fit(batch)   # Incremental fit with X(batch)

        # Progress tracking and memory cleanup
        if i % (batch_size * 10) == 0:  # Every 10 batches
            gc.collect()

    # Transform data in batches to avoid memory issues
    X_pca = np.zeros((X.shape[0], n_components))

    for i in range(0, X.shape[0], batch_size):
        end_idx = min(i + batch_size, X.shape[0])
        batch = X[i:end_idx]
        # Apply dimensionality reduction to X
        X_pca[i:end_idx] = ipca.transform(batch)  # X is projected on the first principal components previously extracted from the training set
    
        # Progress tracking and memory cleanup
        if i % (batch_size * 10) == 0:  # Every 10 batches
            gc.collect()

    print(f"Original shape: {X.shape}")
    print(f"PCA shape: {X_pca.shape}")

    # Analyze explained variance
    explained_variance_ratio = ipca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)  # Cumulative sum

    # Show variance distribution for first components
    print(f"\nFirst 10 components variance: {explained_variance_ratio[:10]}")
    print(f"Total explained variance {n_components} components cumulative: {cumulative_variance[n_components-1]:.3f}")

    return X_pca, ipca, explained_variance_ratio, cumulative_variance

