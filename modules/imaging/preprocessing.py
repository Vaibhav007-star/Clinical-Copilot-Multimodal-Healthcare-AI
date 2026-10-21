"""
Image preprocessing and validation module for Medical Imaging.
Validates file types, dimensions, format, and prepares tensors for model inference.
"""
from pathlib import Path
from typing import Tuple, Union
import io
from PIL import Image

def validate_image_file(
    file_bytes_or_path: Union[bytes, str, Path],
    filename: str,
    max_size_mb: int = 15,
    supported_extensions: tuple = ("png", "jpg", "jpeg", "webp")
) -> Tuple[bool, str, Image.Image]:
    """
    Validates uploaded medical image format, size, and integrity.
    Returns: (is_valid: bool, error_message: str, pil_image: Image.Image)
    """
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in supported_extensions:
        return False, f"Unsupported file type '.{ext}'. Supported types: {', '.join(supported_extensions)}", None

    try:
        if isinstance(file_bytes_or_path, (str, Path)):
            file_path = Path(file_bytes_or_path)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum limit of {max_size_mb}MB.", None
            image = Image.open(file_path)
        else:
            file_size_mb = len(file_bytes_or_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum limit of {max_size_mb}MB.", None
            image = Image.open(io.BytesIO(file_bytes_or_path))

        image.verify()  # Verify integrity
        # Reopen after verify because verify closes or moves the pointer in PIL
        if isinstance(file_bytes_or_path, (str, Path)):
            image = Image.open(file_path).convert("RGB")
        else:
            image = Image.open(io.BytesIO(file_bytes_or_path)).convert("RGB")

        w, h = image.size
        if w < 64 or h < 64:
            return False, f"Image resolution ({w}x{h}) is too low for reliable medical feature extraction (min 64x64).", None

        return True, "", image

    except Exception as e:
        return False, f"Corrupted or invalid image file: {str(e)}", None
