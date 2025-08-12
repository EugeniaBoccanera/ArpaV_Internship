"""
Data processing utilities for meteorological analysis
"""
import xarray as xr
import numpy as np
import gc
from sklearn.decomposition import IncrementalPCA

def prepare_data_matrix(dataset):
    """
    Converts xarray dataset to 2D matrix
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Input dataset with meteorological variables
        
    Returns:
    --------
    combined_matrix : numpy.ndarray
        2D matrix with shape (time, features)
    data_matrices : dict
        Dictionary containing matrices for each variable
    """
    data_matrices = {}
    
    for var in dataset.data_vars:
        print(f"   • Processing {var}...")
        var_data = dataset[var]
        
        # Reorganize dimensions: (time, features)
        if 'time' in var_data.dims:
            # Stack all non-temporal dimensions
            spatial_dims = [dim for dim in var_data.dims if dim != 'time']
            if spatial_dims:
                stacked = var_data.stack(features=spatial_dims)      # From: var[time=1827, pressure=3, lat=201, lon=321]
                matrix = stacked.values  # shape: (time, features)   # To:  var[time=1827, features=193563]  (3×201×321=193563)
            else:
                matrix = var_data.values.reshape(-1, 1)  # For variables without spatial dimensions
        else:
            matrix = var_data.values.flatten().reshape(1, -1)
        
        data_matrices[var] = matrix
    
    # Concatenate all variables
    all_matrices = list(data_matrices.values())
    combined_matrix = np.concatenate(all_matrices, axis=1) # concatenate along the columns (horizontally)

    return combined_matrix, data_matrices

#######################################################################

def apply_global_standardization(X):
    """
    Applies global standardization to the data matrix
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data matrix
        
    Returns:
    --------
    X_standardized : numpy.ndarray
        Standardized data matrix
    global_mean : float
        Global mean of the data
    global_std : float  
        Global standard deviation of the data
    """
    global_mean = X.mean()
    global_std = X.std()
    X_standardized = (X - global_mean) / global_std
    
    return X_standardized, global_mean, global_std

#################################################################

def perform_incremental_pca(X, n_components=30, batch_size=100):
    """
    Performs Incremental PCA on the input data matrix
    """
    print(f"   • Target components: {n_components}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Data shape: {X.shape}")
    
    # Initialize IncrementalPCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # Fit the PCA incrementally
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i+batch_size]
        ipca.partial_fit(batch)
    
        # Progress tracking and memory cleanup
        if i % (batch_size * 10) == 0:  # Every 10 batches
            gc.collect()

    # Transform data in batches to avoid memory issues
    X_pca = np.zeros((X.shape[0], n_components))

    for i in range(0, X.shape[0], batch_size):
        end_idx = min(i + batch_size, X.shape[0])
        batch = X[i:end_idx]
        X_pca[i:end_idx] = ipca.transform(batch)
    
        # Progress tracking and memory cleanup
        if i % (batch_size * 10) == 0:  # Every 10 batches
            gc.collect()

    print(f"   • Original shape: {X.shape}")
    print(f"   • PCA shape: {X_pca.shape}")

    # Analyze explained variance
    explained_variance_ratio = ipca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print("\n   • VARIANCE ANALYSIS:")
    print(f"   • Total explained variance ({n_components} components): {cumulative_variance[-1]:.3f}")

    # Find components needed for different variance thresholds
    thresholds = [0.80, 0.85, 0.90, 0.95]
    for threshold in thresholds:
        n_comp_needed = np.argmax(cumulative_variance >= threshold) + 1
        if cumulative_variance[-1] >= threshold:
            print(f"   • {threshold*100}% variance: {n_comp_needed} components")
        else:
            print(f"   • {threshold*100}% variance: >{n_components} components needed")

    # Show variance distribution for first components
    print(f"\n   • First 10 components variance: {explained_variance_ratio[:10]}")
    print(f"   • First {n_components} components cumulative: {cumulative_variance[n_components-1]:.3f}")

    return X_pca, ipca, explained_variance_ratio, cumulative_variance

