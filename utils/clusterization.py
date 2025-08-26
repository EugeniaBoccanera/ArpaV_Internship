"""
Functions for the clusterization process and clusterization analysis
"""
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score


########################################### kmeans

def kmeans(X_pca,K_range,n_init=20):
    """
    Performs K-means clustering on the PCA-transformed data.
    """
    inertia = []
    all_labels = {}
    all_centroids = {}

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init)
        labels = kmeans.fit_predict(X_pca)
        inertia.append(kmeans.inertia_)
        all_labels[k] = labels
        all_centroids[k] = kmeans.cluster_centers_

    # Save results for later analysis
    results = {
        "K_range": list(K_range),
        "inertia": inertia,
        "labels": all_labels,
        "centroids": all_centroids
    }
    return results


########################################### elbow analysis
def elbow_analysis(results_dict, min_significant_diff=0.8):
    """
    Analyzes the elbow method with automatic detection of derivative change point
    """
    K_range = results_dict["K_range"]
    inertias = results_dict["inertia"]

    # Calculate percentage reduction
    reduction_percentage = []
    for i in range(1, len(inertias)):
        reduction = (inertias[i-1] - inertias[i]) / inertias[i-1] * 100
        reduction_percentage.append(reduction)

    # Calculate differences between consecutive reductions (second derivative)
    reduction_diffs = []
    for i in range(1, len(reduction_percentage)):
        diff = abs(reduction_percentage[i-1] - reduction_percentage[i])
        reduction_diffs.append(diff)
    
    # Find the point where the difference drops drastically
    k_optimal_derivative = None
    for i in range(len(reduction_diffs) - 1):
        current_diff = reduction_diffs[i]
        next_diff = reduction_diffs[i + 1]
        
        # If the difference drops below the significant threshold
        if current_diff > min_significant_diff and next_diff < min_significant_diff:
            k_optimal_derivative = K_range[i + 2]  # +2 because reduction_diffs starts from k=3
            break

    # Geometric elbow method
    def find_elbow_point(x, y):
        first_point = np.array([x[0], y[0]])
        last_point = np.array([x[-1], y[-1]])
        distances = []
        for i in range(len(x)):
            point = np.array([x[i], y[i]])
            distance = np.abs(np.cross(last_point - first_point, first_point - point)) / np.linalg.norm(last_point - first_point)
            distances.append(distance)
        return x[np.argmax(distances)]
    
    k_elbow_geometric = find_elbow_point(K_range, inertias)
    
    # Plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Classic elbow
    ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(k_elbow_geometric, color='navy', linestyle='--', alpha=0.7, 
                label=f'Geometric Elbow: k={k_elbow_geometric}')
    if k_optimal_derivative:
        ax1.axvline(k_optimal_derivative, color='red', linestyle='--', alpha=0.7,
                    label=f'Derivative Method: k={k_optimal_derivative}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axis('off')

    # 2. Percentage reduction
    bars = ax3.bar(K_range[1:], reduction_percentage, alpha=0.7, color='orange')
    if k_optimal_derivative:
        # Highlight the bar for the optimal k
        idx = k_optimal_derivative - K_range[0] - 1
        if 0 <= idx < len(bars):
            bars[idx].set_color('red')
            bars[idx].set_alpha(1.0)
    
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Inertia Reduction (%)')
    ax3.set_title('Percentage Reduction of Inertia')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Values on the bars
    for k, reduction in zip(K_range[1:], reduction_percentage):
        ax3.text(k, reduction + 0.2, f'{reduction:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    
    
    # 3. Differences between consecutive reductions 
    if len(reduction_diffs) > 0:
        bars4 = ax4.bar(K_range[2:], reduction_diffs, alpha=0.7, color='green')
        ax4.axhline(min_significant_diff, color='red', linestyle='--', alpha=0.7,
                    label=f'Significance threshold: {min_significant_diff}%')
        
        if k_optimal_derivative:
            # Highlight the change point
            idx = k_optimal_derivative - K_range[0] - 2
            if 0 <= idx < len(bars4):
                bars4[idx].set_color('red')
                bars4[idx].set_alpha(1.0)

        ax4.set_xlabel('Number of Clusters (k)')
        ax4.set_ylabel('Difference in Reduction (%)')
        ax4.set_title('Rate of Change in Reduction (Second Derivative)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Values on bars
        for k, diff in zip(K_range[2:], reduction_diffs):
            ax4.text(k, diff + 0.05, f'{diff:.1f}%', 
                    ha='center', va='bottom', fontsize=9)
    
    
    plt.tight_layout()
    plt.show()

    return k_elbow_geometric, k_optimal_derivative, reduction_percentage, reduction_diffs

########################################### gap statistics

def gap_statistic(X_data, K_range,results, n_refs=20, random_state=42):
    """
    Calculate Gap Statistic to find the optimal number of clusters
    """
    np.random.seed(random_state)
    
    # Get data range for creating random datasets
    min_vals = X_data.min(axis=0)
    max_vals = X_data.max(axis=0)
    
    gaps = []
    errors = []
    
    for k in K_range:
        
        # Use already computed inertia from previous analysis
        real_inertia = results["inertia"][k-2]  
        
        # Generate random datasets and compute their inertias
        random_inertias = []
        for i in range(n_refs):
            # Create random data with same shape and range as real data
            random_data = np.random.uniform(min_vals, max_vals, size=X_data.shape)
            
            # Apply K-means to random data
            kmeans = KMeans(n_clusters=k, random_state=i, n_init=10)
            kmeans.fit(random_data)
            random_inertias.append(kmeans.inertia_)
        
        # Calculate gap statistic
        log_real = np.log(real_inertia)
        log_random = np.log(random_inertias)
        
        gap = np.mean(log_random) - log_real
        error = np.sqrt(1 + 1.0/n_refs) * np.std(log_random, ddof=1)
        
        gaps.append(gap)
        errors.append(error)
    
    # Find optimal k using the standard rule
    optimal_k = K_range[0]
    for i in range(len(K_range) - 1):
        if gaps[i] >= gaps[i+1] - errors[i+1]:
            optimal_k = K_range[i]
            break
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(K_range, gaps, yerr=errors, fmt='o-', capsize=5, color='indigo')
    plt.axvline(optimal_k, color='rebeccapurple', linestyle='--', 
                label=f'Optimal k = {optimal_k}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Gap Statistic')
    plt.title('Gap Statistic Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.show()
    
    return optimal_k, gaps, errors


############################################### silhouette analysis
def silhouette_analysis(X, k_values, random_state=42):
    """
    Analyzes the silhouette for different values of k
    """
    # Calculate the number of rows and columns needed
    n_plots = len(k_values)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols  
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
     # If we have only one row, axes might not be 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    scores = []

    for i, k in enumerate(k_values):
        # Apply K-means
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)

        # Calculate silhouette score
        sil_score = silhouette_score(X, labels)
        sil_samples = silhouette_samples(X, labels)
        scores.append(sil_score)
        
        # Create the plot
        ax = axes[i]
        y_pos = 0

        # For each cluster draw the silhouette
        for cluster in range(k):
            cluster_sil = sil_samples[labels == cluster]
            cluster_sil.sort()
            
            # Draw the cluster bar
            ax.fill_betweenx(range(y_pos, y_pos + len(cluster_sil)), 
                            0, cluster_sil, alpha=0.7)
            y_pos += len(cluster_sil)
        
        # Mean line
        ax.axvline(sil_score, color='red', linestyle='--')
        ax.set_title(f'k={k}, Mean Silhouette: {sil_score:.3f}')
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Sample Index')

    # Remove unused subplots
    for i in range(len(k_values), len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    plt.show()
    
    return scores


################################################# davies_bouldin index

def davies_bouldin_analysis(X_pca, k_values, results, random_state=42):
    """
    Compute the Davies-Bouldin index for different values of k.
    """
    db_scores = []
    
    for k in k_values:
        # Use the kmeans results
        if k in results["labels"]:
            labels = results["labels"][k]

            # Compute Davies-Bouldin score
            db_score = davies_bouldin_score(X_pca, labels)
            db_scores.append(db_score)

    # Find best k (lower score)
    if db_scores:
        optimal_k = k_values[np.argmin(db_scores)]
        min_score = min(db_scores)
        
        print(f"Best value for k={optimal_k} (DB = {min_score:.3f})")
        
        # Plots
        plt.figure(figsize=(10, 6))
        plt.plot(k_values[:len(db_scores)], db_scores, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Goodness limit (2.0)')

        # Highlight the smaller value
        min_idx = np.argmin(db_scores)
        plt.plot(k_values[min_idx], db_scores[min_idx], 'ro', markersize=12, 
                 label=f'Best for k={optimal_k}')
        
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Davies-Bouldin Index')
        plt.title('Davies-Bouldin Index')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(k_values[:len(db_scores)])
        plt.ylim(1.5,2.05)
        
        # Annotate the values
        for i, (k, score) in enumerate(zip(k_values[:len(db_scores)], db_scores)):
            plt.annotate(f'{score:.2f}', (k, score), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.show()
    
    return db_scores