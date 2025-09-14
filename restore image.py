import numpy as np
import pickle
import os
from PIL import Image

# Load compressed data
compressed_data = np.load('compresseddata.npy')

# Load PCA model
with open('pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

def inverse_with_m(pca, Z, m):
    return np.dot(Z[:, :m], pca.components_[:m, :]) + pca.mean_

restored_images = inverse_with_m(pca, compressed_data, m=10)

print(restored_images.shape)

# Store restored images in a folder named "compresseddata"
folder_path = 'compresseddata'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def get_image_from_pixel_matrix(pixel_matrix):
    """Get an image from a pixel matrix."""
    pixel_matrix.resize(240,240)
    img = Image.fromarray(pixel_matrix.astype(np.uint8))
    return img

for i, img_data in enumerate(restored_images):
    img = get_image_from_pixel_matrix(img_data.reshape(240, 240, 1))
    img.save(os.path.join(folder_path, f'restored_image_{i}.png'))