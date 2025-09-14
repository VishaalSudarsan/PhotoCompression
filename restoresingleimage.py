def downsample_image(image_path: Path) -> Image:
    """Read a single image and downsample it to 240*240 pixel."""
    with Image.open(image_path) as img:
        img = img.resize((240, 240), Image.ANTIALIAS)
    return img
