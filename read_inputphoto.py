#!/usr/bin/env python3
"""Utility to inspect inputphoto and convert it to a pixel matrix."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def main() -> None:
    path = Path("inputphoto.jpg")
    data = path.read_bytes()
    print(f"Read {len(data)} bytes from {path.name}")
    preview = " ".join(f"{byte:02x}" for byte in data[:20])
    print(f"First 20 bytes: {preview}")

    with Image.open(path) as img:
        img = img.convert("L")
        width, height = img.size
        if width != height:
            raise ValueError(
                f"Input photo must be square; got {width}x{height}"
            )
        pixels = list(img.getdata())

    matrix: List[List[int]] = [
        pixels[i * width : (i + 1) * width] for i in range(height)
    ]
    print(f"Image converted to {width}x{height} grayscale matrix")
    top_left = [row[:5] for row in matrix[:5]]
    print(f"Top-left 5x5 block: {top_left}")
    mid, left_matrix = find_vertical_symmetry(matrix)
    print(f"Best vertical symmetry at column {mid}")
    print(f"Left matrix up to mid has shape {len(left_matrix)}x{len(left_matrix[0]) if left_matrix else 0}")


def find_vertical_symmetry(matrix: List[List[int]]) -> Tuple[int, List[List[int]]]:
    """Return the column index forming the best vertical symmetry.

    For each possible ``mid`` column the routine computes a loss equal to the
    sum of squared differences between pixels mirrored across ``mid``. The
    ``mid`` that minimises this loss is returned along with the portion of the
    matrix up to (but not including) ``mid``.
    """

    if not matrix or not matrix[0]:
        return 0, []

    arr = np.asarray(matrix, dtype=float)
    height, width = arr.shape
    best_mid = 0
    best_loss = float("inf")

    for mid in range(width):
        max_a = min(mid, width - mid - 1)
        if max_a == 0:
            continue

        left = arr[:, mid - max_a : mid][:, ::-1]
        right = arr[:, mid + 1 : mid + max_a + 1]
        diff = left - right
        ones_row = np.ones((1, height))
        ones_col = np.ones((max_a, 1))
        loss = float(ones_row @ (diff ** 2) @ ones_col)

        if loss < best_loss:
            best_loss = loss
            best_mid = mid

    left_matrix = arr[:, :best_mid].tolist()
    return best_mid, left_matrix


if __name__ == "__main__":
    main()
