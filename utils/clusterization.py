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
    ax2.axhline(3, color='darkred', linestyle=':', alpha=0.7, 
                label='3% threshold')
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