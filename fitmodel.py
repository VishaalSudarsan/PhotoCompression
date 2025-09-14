import numpy as np
from typing import List
from sklearn.decomposition import PCA
import pickle

def read_pixel_matrix(filename: str = "pixel_matrix.npy") -> np.ndarray:
    """Load the pixel matrix from a .npy file."""
    matrix = np.load(filename)
    print(f"Loaded pixel matrix with shape {matrix.shape}")
    return matrix

def apply_pca(vectors: List[List[int]], n_components: int) -> np.ndarray:
    """Apply PCA to the matrix of photo vectors."""
    X = np.array(vectors)
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced shape: {X_pca.shape}")
    return pca

n_components = 6000
pixel_matrix = read_pixel_matrix()
pca = apply_pca(pixel_matrix, n_components=n_components)

with open(f"pca_model_{n_components}.pkl", "wb") as f:
    pickle.dump(pca, f)
compresseddata = pca.transform(pixel_matrix)
np.save("compresseddata.npy", compresseddata)
print(f"Percentage variance explained by PCA: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
