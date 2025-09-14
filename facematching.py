import os
import numpy as np
import pickle
from PIL import Image
from pathlib import Path

def downsample_image(image_path: Path) -> Image:
    """Read a single image and downsample it to 240*240 pixel."""
    with Image.open(image_path) as img:
        img = img.resize((240, 240))
        #img.show()
    return img

def image_to_greyscale_pixel_matrix(img: Image) -> np.ndarray:
    """Convert an image into a grayscale pixel matrix."""
    width, height = img.size
    pixels = list(img.convert("L").getdata())
    matrix = np.array([pixels[i * width : (i + 1) * width] for i in range(height)])
    return matrix

def inverse_with_m(pca, Z, m):
    return np.dot(Z[:, :m], pca.components_[:m, :]) + pca.mean_

def get_image_from_pixel_matrix(pixel_matrix):
    """Get an image from a pixel matrix."""
    pixel_matrix.resize(240,240)
    img = Image.fromarray(pixel_matrix.astype(np.uint8))
    return img

# Load PCA model
n_components = 6000
with open(f'pca_model_{n_components}.pkl', 'rb') as f:
    pca = pickle.load(f)


inputphoto = downsample_image(Path("example.png"))
inputmatrix = image_to_greyscale_pixel_matrix(inputphoto)
compressed_image = pca.transform([inputmatrix.flatten()])

compressed_data = np.load('compresseddata.npy')

best_match = None
best_match_percent = 0
for i, row in enumerate(compressed_data):
    from skimage.metrics import structural_similarity as ssim
    diff = row - compressed_image[0]
    ssim = ssim(diff, compressed_image[0], data_range=diff.max() - diff.min())
    match_percent = ssim
    if match_percent > best_match_percent:
        best_match = i
        best_match_percent = match_percent

if best_match is not None:
    print(f"Best match is row {best_match} with {best_match_percent*100:.2f}% match")

# Load compressed data
restored_images = inverse_with_m(pca, compressed_data[[best_match]], m=6000)

img = get_image_from_pixel_matrix(restored_images.reshape(240, 240, 1))
img.show()