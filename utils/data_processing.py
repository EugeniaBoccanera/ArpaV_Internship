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
        print(f"Processing {var}...")
        var_data = dataset[var]   # var_data.dims = ('time', 'pressure_z or _t', 'latitude', 'longitude')
        
        # Reorganize dimensions: (time, features)
        if 'time' in var_data.dims:
            # Stack all non-temporal dimensions
            spatial_dims = [dim for dim in var_data.dims if dim != 'time']

            if spatial_dims:  # If there are spatial dimensions to stack
                stacked = var_data.stack(features=spatial_dims)       # From shape: (time=1827, pressure=3, lat=201, lon=321)
                # conversion to a numpy array
                #matrix = stacked.values        # To:  shape(time=1827, features=193563)  (3×201×321=193563) for each variable
                matrix = stacked.values.astype(np.float32)
                print(f"     → {var}: {var_data.dims} → {matrix.shape}")

        data_matrices[var] = matrix
    
    # Concatenate all variables
    all_matrices = list(data_matrices.values())
    combined_matrix = np.concatenate(all_matrices, axis=1) # concatenate along the columns (horizontally)

    print(f"\nCombined matrix shape: {combined_matrix.shape}")
    

    return combined_matrix, data_matrices


##################################################### separate standardization

def apply_separate_standardization(X, spatial_size):
    """
    Applies separate standardization to temperature and geopotential data
    
    """
    
    # Split the matrix into temperature and geopotential parts
    X_temperature = X[:, :spatial_size]        # First half: temperature (T)
    X_geopotential = X[:, spatial_size:]       # Second half: geopotential (Z)
    
    print(f"Temperature matrix shape: {X_temperature.shape}")
    print(f"Geopotential matrix shape: {X_geopotential.shape}")
    
    # Compute statistics for each variable separately
    t_mean = X_temperature.mean()
    t_std = X_temperature.std()
    z_mean = X_geopotential.mean()
    z_std = X_geopotential.std()
    
    print(f"Temperature - Mean: {t_mean:.2f}, Std: {t_std:.2f}")
    print(f"Geopotential - Mean: {z_mean:.2f}, Std: {z_std:.2f}")
    
    # Standardize each variable separately
    X_temperature_std = (X_temperature - t_mean) / t_std
    X_geopotential_std = (X_geopotential - z_mean) / z_std
    
    # Recombine the standardized matrices
    X_standardized = np.concatenate([X_temperature_std, X_geopotential_std], axis=1)
    
    print(f"Combined standardized matrix shape: {X_standardized.shape}")
    
    # Verify standardization
    print(f"Temperature after standardization - Mean: {X_temperature_std.mean():.6f}, Std: {X_temperature_std.std():.6f}")
    print(f"Geopotential after standardization - Mean: {X_geopotential_std.mean():.6f}, Std: {X_geopotential_std.std():.6f}")
    
    return X_standardized, t_mean, t_std, z_mean, z_std

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

