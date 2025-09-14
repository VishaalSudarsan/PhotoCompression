import numpy as np
from typing import List
from sklearn.decomposition import PCA
import pickle

def read_pixel_matrix(filename: str = "pixel_matrix.npy") -> np.ndarray:
    """Load the pixel matrix from a .npy file."""
    matrix = np.load(filename)
    print(f"Loaded pixel matrix with shape {matrix.shape}")
    return matrix

def apply_pca(vectors: List[List[int]], n_components: int = 2000) -> np.ndarray:
    """Apply PCA to the matrix of photo vectors."""
    X = np.array(vectors)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced shape: {X_pca.shape}")
    return pca

pixel_matrix = read_pixel_matrix()
pca = apply_pca(pixel_matrix)

with open("pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)
compresseddata = pca.transform(pixel_matrix)
np.save("compresseddata.npy", compresseddata)
print(f"Percentage variance explained by PCA: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
