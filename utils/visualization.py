"""
Visualization utilities for meteorological analysis
"""
import matplotlib.pyplot as plt
import numpy as np
import shapefile as shp

def plot_variance(explained_variance_ratio, cumulative_variance, n_components):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Individual explained variance ratio
    ax1.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'b-', alpha=0.7)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual Explained Variance per Component')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, n_components) 

    # Plot 2: Cumulative explained variance
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'r-', linewidth=2)
    ax2.axhline(y=0.80, color='orange', linestyle='--', alpha=0.7, label='80%')
    ax2.axhline(y=0.85, color='blue', linestyle='--', alpha=0.7, label='85%')
    ax2.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='90%')
    ax2.axhline(y=0.95, color='purple', linestyle='--', alpha=0.7, label='95%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, n_components)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

#########################################################################

def scatter_plot_2d(X_pca):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10))

    ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=20, c='navy')
    ax1.set_xlabel('PC1 (First Principal Component)')
    ax1.set_ylabel('PC2 (Second Principal Component)')
    ax1.set_title('PC1 vs PC2')
    ax1.grid(True, alpha=0.3)

    # PC1 vs PC3 scatter
    ax2.scatter(X_pca[:, 0], X_pca[:, 2], alpha=0.6, s=20, c='firebrick')
    ax2.set_xlabel('PC1 (First Principal Component)')
    ax2.set_ylabel('PC3 (Third Principal Component)')
    ax2.set_title('PC1 vs PC3')
    ax2.grid(True, alpha=0.3)

    # PC2 vs PC3 scatter
    ax3.scatter(X_pca[:, 1], X_pca[:, 2], alpha=0.6, s=20, c='green')
    ax3.set_xlabel('PC2 (Second Principal Component)')
    ax3.set_ylabel('PC3 (Third Principal Component)')
    ax3.set_title('PC2 vs PC3')
    ax3.grid(True, alpha=0.3)

    # PC1 vs PC4 scatter
    ax4.scatter(X_pca[:, 0], X_pca[:, 3], alpha=0.6, s=20, c='darkorange')
    ax4.set_xlabel('PC1 (First Principal Component)')
    ax4.set_ylabel('PC4 (Fourth Principal Component)')
    ax4.set_title('PC1 vs PC4')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

#################################################################################

def scatter_plot_3d(X_pca):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                    alpha=0.6, s=20, c=X_pca[:, 3], cmap='viridis')
    ax.set_xlabel('PC1 (First Principal Component)')
    ax.set_ylabel('PC2 (Second Principal Component)')
    ax.set_zlabel('PC3 (Third Principal Component)')
    ax.set_title('3D Visualization: PC1 vs PC2 vs PC3')

    # Add colorbar
    plt.colorbar(scatter, shrink=0.5, aspect=5)
    plt.show()

#################################################################################

# Function to add the Europe profile
def add_country_boundaries(ax, sf, color='black'):
    """
    Adds the country boundaries 
    """
    for shape in sf.shapeRecords():
        for i in range(len(shape.shape.parts)):
            i_start = shape.shape.parts[i]
            if i == len(shape.shape.parts) - 1:
                i_end = len(shape.shape.points)
            else:
                i_end = shape.shape.parts[i + 1]
                
            x = [point[0] for point in shape.shape.points[i_start:i_end]]
            y = [point[1] for point in shape.shape.points[i_start:i_end]]
                
            ax.plot(x, y, color=color, linewidth=0.8, alpha=0.7)

#################################################################################

def visualization_pca_coefficient(components_to_plot, pressure_levels, lats, lons,n_pressure, n_lat, n_lon, sf=None):
    for pc_coefficients, pc_name, pc_title in components_to_plot:

        # Forma originale dei coefficienti
        pc_reshaped = pc_coefficients.reshape(n_pressure, n_lat, n_lon)
    
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
        for i, (pressure, ax) in enumerate(zip(pressure_levels, axes)):
            # Dati per questo livello di pressione
            pc_level = pc_reshaped[i, :, :]
        
            # Coordinate per la mappa
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        
            # Contour plot
            contour_lines = ax.contour(lon_grid, lat_grid, pc_level, 
                                  levels=20, linewidths=0.5, colors='black', alpha=0.6)
        
            # Contour filled
            contour_filled = ax.contourf(lon_grid, lat_grid, pc_level, 
                                    levels=20, cmap="RdBu_r", extend='both')
        
            # Etichette delle isolinee
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
        
            if sf is not None:
                add_country_boundaries(ax, sf)

        
            ax.set_title(f'{pc_name} Coefficients at {pressure} hPa', fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude [°E]')
            ax.set_ylabel('Latitude [°N]')
            ax.set_xlim([lons.min(), lons.max()])
            ax.set_ylim([lats.min(), lats.max()])
            ax.grid(True, alpha=0.3)
        
            cbar = plt.colorbar(contour_filled, ax=ax, shrink=0.8)
            cbar.set_label(f'{pc_name} Coefficient', rotation=270, labelpad=15)
    
        # Titolo generale
        fig.suptitle(f'{pc_title} - Spatial pattern of coefficients', 
                 fontsize=16, fontweight='bold')
    
        plt.tight_layout()
        plt.show()

###########################################################################
