from pathlib import Path
from typing import List
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np

def photo_to_vector(photo_path: Path) -> List[int]:
    """Convert a photo to a grayscale pixel vector."""
    with Image.open(photo_path) as img:
        img = img.convert("L")
        pixels = list(img.getdata())
    return pixels

def read_photos_from_data_folder(data_folder: Path = Path("data")) -> List[List[int]]:
    """Read all photos from the data folder as pixel vectors."""
    vectors = []
    for photo_path in data_folder.glob("*.png"):
        vector = photo_to_vector(photo_path)
        vectors.append(vector)
        print(f"Read {photo_path.name}: {len(vector)} pixels")
    return vectors

def apply_pca(vectors: List[List[int]], n_components: int = 50) -> np.ndarray:
    """Apply PCA to the matrix of photo vectors."""
    X = np.array(vectors)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced shape: {X_pca.shape}")
    return X_pca

vectors = read_photos_from_data_folder()
print(f"Total photos processed: {len(vectors)}")
if vectors:
    X_pca = apply_pca(vectors)
    print("PCA transformation complete.")