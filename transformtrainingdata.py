from pathlib import Path
from typing import List
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np

def photo_to_vector(photo_path: Path) -> List[int]:
    # Green background
    bg = Image.new("RGBA", (240, 240), (0, 255, 0, 255))

    """Convert a photo to a grayscale pixel vector."""
    with Image.open(photo_path) as img:
        composite = Image.alpha_composite(bg, img)
        img = composite.convert("L")
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

def save_pixel_matrix(matrix: List[List[int]], filename: str = "pixel_matrix.npy") -> None:
    """Save the matrix of pixel vectors to a .npy file."""
    np.save(filename, np.array(matrix))
    print(f"Pixel matrix saved to {filename}")


vectors = read_photos_from_data_folder()
print(f"Total photos processed: {len(vectors)}")
if vectors:
    save_pixel_matrix(vectors)  # Save the pixel matrix
