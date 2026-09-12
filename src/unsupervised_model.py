import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_unsupervised(X_scaled):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    var_acum = var_exp.sum() * 100
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    sil_score = silhouette_score(X_pca, clusters)
    
    print("\n--- [NO SUPERVISADO: PCA + K-MEANS] ---")
    print(f"Varianza Acumulada 2D: {var_acum:.2f}%")
    print(f"Coeficiente de Silueta (k=3): {sil_score:.4f}")
    
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.6, edgecolors="k", s=35)
    plt.scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
        s=180, c="red", marker="X", label="Centroides"
    )
    plt.title(f"PCA + K-Means (k=3) | Silueta: {sil_score:.4f} | Varianza: {var_acum:.2f}%")
    plt.xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    plt.colorbar(scatter, label="Cluster")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("clustering_2d.png", dpi=300)
    plt.close()
    print("[+] Grafica guardada como clustering_2d.png")
    return X_pca, clusters, sil_score, var_acum