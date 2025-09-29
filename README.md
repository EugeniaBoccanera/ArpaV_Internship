# ArpaV_Internship
Project developed at ARPA Veneto from 21 July to 30 September, 2025. The goal is to design an unsupervised clustering algorithm for identifying meteorological patterns over Europe.


### INSTRUCTION TO NAVIGATE INTO THE FILES
**N.B.** The data are not included because they are too large.

_Repository contents_:

**B-60_years** → contains the 4 files required to run the analysis, as well as the folder for saving intermediate results.

**RISULTATI-K=44** → contains the analysis results for k (number of clusters) = 44.

**RISULTATI-K=7** → contains the analysis results for k (number of clusters) = 7.

**utils** → contains some of the functions used in the analysis (from B-60_years).

**world** → contains the files needed to plot the world boundaries.

**Report**: Unsupervised_Clustering_Methods_for_Meteorological_European_Configurations.pdf


In particular the _B-60_years folder_ contains the following files:
1.	**1-creazione_matrice_60y.ipynb**  → Script for generating the matrix for the 60 years period with a sampled dataset (1 day every 5). Use this code only if you need to analyze a different period.

2.	**2-application_to_10y.ipynb** → Code to obtain the normalized and PCA-reduced matrix X_pca_60y (saved in Mid_result_to_save). A separate normalization and PCA are performed for each of the six decades, and the six resulting datasets are then combined to produce a unified 60-year matrix that is already normalized and dimensionally reduced. Use this only if the period under analysis differs.

3.	**3-main_fit_60y** → Main script where K-means is applied to the 60-year matrix. It saves the centroids for further analysis and also computes the Elbow method, Silhouette score, and Calinski-Harabasz index. Use this script if you want to change the K-means parameters or visualize the physical maps of the centroids.

4.	**4-test_over_10y** → Analyzes temporal trends in atmospheric regime frequencies across six decades (1961–2020) by assigning daily data to the centroids identified in main_fit_60y. It also evaluates seasonal patterns across decades.

