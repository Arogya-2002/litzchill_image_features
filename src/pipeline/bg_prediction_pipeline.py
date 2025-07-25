
from src.components.remove_bg import RemoveBg,AddBg  

from src.exceptions import CustomException
from src.logger import logging
import sys

def process_remove_bg(input_image_path: str) -> str:
    """Removes background from the given image path and returns path to the saved image."""
    try:
        logging.info(f"Starting background removal for: {input_image_path}")
        remove_bg_obj = RemoveBg()
        artifact = remove_bg_obj.remove_bg(input_image_path)
        return artifact.rmbg_img_path
    except Exception as e:
        logging.error(f"Error while removing background: {e}")
        raise CustomException(e, sys) from e

def process_add_bg(foreground_img_path: str, new_bg_path: str) -> str:
    """Adds new background to a transparent image and returns path to saved image."""
    try:
        logging.info(f"Starting background addition with foreground: {foreground_img_path} and background: {new_bg_path}")
        add_bg_obj = AddBg()
        artifact = add_bg_obj.change_bg(foreground_img_path, new_bg_path)
        return artifact.ch_bg_img_path
    except Exception as e:
        logging.error(f"Error while adding background: {e}")
        raise CustomException(e, sys) from e
