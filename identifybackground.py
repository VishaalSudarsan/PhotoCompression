import numpy as np
import pickle
import os
from PIL import Image
from pathlib import Path
from typing import Tuple


def rpca(X, lam=None, mu=None, max_iter=1000, tol=1e-7):
    n, d = X.shape
    if lam is None:
        lam = 1 / np.sqrt(max(n, d))
    if mu is None:
        mu = (n * d) / (4.0 * np.linalg.norm(X, ord=1))

    # initialize
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)

    norm_X = np.linalg.norm(X, ord='fro')

    for _ in range(max_iter):
        # Singular Value Thresholding (for L)
        U, sigma, Vt = np.linalg.svd(X - S + (1/mu)*Y, full_matrices=False)
        sigma_thresh = np.maximum(sigma - 1/mu, 0)
        L = (U * sigma_thresh) @ Vt

        # Soft thresholding (for S)
        temp = X - L + (1/mu)*Y
        S = np.sign(temp) * np.maximum(np.abs(temp) - lam/mu, 0)

        # Dual update
        Z = X - L - S
        Y = Y + mu * Z

        if np.linalg.norm(Z, ord='fro') / norm_X < tol:
            break

    return L, S

def get_image_from_pixel_matrix(pixel_matrix, alpha=0.05):
    """Get an image from a pixel matrix."""
    pixel_matrix.resize(240,240)
    img = Image.fromarray(pixel_matrix.astype(np.uint8))
    return img

def read_pixel_matrix(filename: str = "pixel_matrix.npy") -> np.ndarray:
    """Load the pixel matrix from a .npy file."""
    matrix = np.load(filename)
    print(f"Loaded pixel matrix with shape {matrix.shape}")
    return matrix


def separate_background_subject(pixel_matrix: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Robust PCA to the pixel matrix to separate the background and subject."""
    background, subject = rpca(pixel_matrix)
    return background, subject

pixel_matrix = read_pixel_matrix()
background_matrix, subject_matrix = separate_background_subject(pixel_matrix, alpha=0.05)
background_image = get_image_from_pixel_matrix(background_matrix[0], alpha=0.05)
subject_image = get_image_from_pixel_matrix(subject_matrix[0], alpha=0.05)
background_image.show()
subject_image.show()
print("debug")