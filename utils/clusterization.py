"""
Functions for the clusterization process and clusterization analysis
"""
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


########################################### kmeans

def kmeans(X_pca,K_range,n_init=20):
    inertia = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init)
        labels = kmeans.fit_predict(X_pca)
        inertia.append(kmeans.inertia_)

    # Save results for later analysis
    results = {
        "K_range": list(K_range),
        "inertia": inertia
    }
    return results


########################################### elbow analysis
def elbow_analysis(results_dict):
    """
    Analizza la curva dell'elbow method e calcola le riduzioni percentuali
    """
    K_range = results_dict["K_range"]
    inertias = results_dict["inertia"]
    
    # Calcolo riduzione percentuale
    reduction_pct = []
    for i in range(1, len(inertias)):
        reduction = (inertias[i-1] - inertias[i]) / inertias[i-1] * 100
        reduction_pct.append(reduction)
    
    # Trova il gomito usando il metodo della distanza massima
    def find_elbow_point(x, y):
        # Linea dal primo all'ultimo punto
        first_point = np.array([x[0], y[0]])
        last_point = np.array([x[-1], y[-1]])
        
        # Calcola distanze di ogni punto dalla linea
        distances = []
        for i in range(len(x)):
            point = np.array([x[i], y[i]])
            distance = np.abs(np.cross(last_point - first_point, first_point - point)) / np.linalg.norm(last_point - first_point)
            distances.append(distance)
        
        # Restituisce k con distanza massima
        max_idx = np.argmax(distances)
        return x[max_idx]
    
    k_elbow = find_elbow_point(K_range, inertias)
    
    # Grafici
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Grafico elbow
    ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(k_elbow, color='navy', linestyle='--', alpha=0.7, 
                label=f'Elbow at k = {k_elbow}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Valori sui punti
    for k, inertia in zip(K_range, inertias):
        ax1.annotate(f'{inertia:.0f}', (k, inertia), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=9)
    
    # Grafico riduzione percentuale
    ax2.bar(K_range[1:], reduction_pct, alpha=0.7, color='orange')
    ax2.axhline(3.5, color='darkred', linestyle=':', alpha=0.7, 
                label='3,5% threshold')
    ax2.axvline(k_elbow, color='navy', linestyle='--', alpha=0.6)
    ax2.axvline(9, color='darkred', linestyle='--', alpha=0.6)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Inertia Reduction (%)')
    ax2.set_title('Percentage Reduction of Inertia')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Valori sulle barre
    for k, reduction in zip(K_range[1:], reduction_pct):
        ax2.text(k, reduction + 0.5, f'{reduction:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return k_elbow, reduction_pct


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
    plt.errorbar(K_range, gaps, yerr=errors, fmt='o-', capsize=5, color='blue')
    plt.axvline(optimal_k, color='red', linestyle='--', 
                label=f'Optimal k = {optimal_k}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Gap Statistic')
    plt.title('Gap Statistic Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.show()
    
    return optimal_k, gaps, errors