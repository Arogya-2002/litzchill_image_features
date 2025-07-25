from src.exceptions import CustomException
from src.logger import logging
import os
import sys

from src.components.image_upscaler import ImageUpscaler

from src.components.image_upscaler import ImageUpscaler
from src.exceptions import CustomException
from src.logger import logging
import sys

def initialte_upscaler_pipeline(image_path: str) -> str:
    """
    Runs the image upscaling pipeline on the given image path.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str: Path to the upscaled image.

    Raises:
        CustomException: If any error occurs during upscaling.
    """
    try:
        upscaler = ImageUpscaler()
        artifact = upscaler.upscale_image(image_path, use_tiling=True)

        logging.info(f"Image upscaling completed successfully: {artifact.upscaled_image_path}")
        return artifact.upscaled_image_path

    except Exception as e:
        logging.error(f"Error during image upscaling: {str(e)}")
        raise CustomException(f"Image upscaling failed: {str(e)}", sys) from e



