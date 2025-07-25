from src.exceptions import CustomException
import sys
from src.logger import logging

from src.entity.config import ConfigEntity, RemoveBgConfig, ChangeBgConfig
from src.entity.artifact import RemoveBgArtifact, ChangeBgArtifact
from src.utils import generate_unique_filename

from rembg import remove, new_session
from PIL import Image
import os
import shutil

class RemoveBg:
    def __init__(self):
        try:
            self.remove_bg_config = RemoveBgConfig(bg_config=ConfigEntity())
        except Exception as e:
            raise CustomException(e, sys)

    def remove_bg(self, input_image_path: str) -> RemoveBgArtifact:
        try:
            logging.info(f"Removing background from image: {input_image_path}")
            input_image = Image.open(input_image_path)

            # Create session with specific model
            session = new_session(model_name='u2net')

            output_image = remove(
                input_image,
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=5
            )

            # Ensure specific folder exists
            os.makedirs(self.remove_bg_config.img_path_folder, exist_ok=True)

            # Generate unique filename
            output_image_path = generate_unique_filename(
                input_image_path,
                self.remove_bg_config.img_path_folder
            )

            # Save final output image
            output_image.save(output_image_path)

            logging.info(f"Background removed and saved to: {output_image_path}")

            return RemoveBgArtifact(rmbg_img_path=output_image_path)

        except Exception as e:
            logging.error(f"Error in removing background: {e}")
            raise CustomException(e, sys) from e




class AddBg:
    def __init__(self):
        try:
            self.change_bg_config = ChangeBgConfig(bg_config=ConfigEntity())
        except Exception as e:
            raise CustomException(e, sys)

    def change_bg(self, img_path: str, bg_img_path: str) -> ChangeBgArtifact:
        try:
            logging.info("Changing background of the image.")

            # Load foreground and background
            foreground = Image.open(img_path).convert("RGBA")
            background = Image.open(bg_img_path).convert("RGBA")
            background = background.resize(foreground.size)

            # Composite images
            combined = Image.alpha_composite(background, foreground)

            # Ensure output folders exist
            os.makedirs(self.change_bg_config.img_path_folder, exist_ok=True)
            os.makedirs(self.change_bg_config.uploaded_path_folder, exist_ok=True)  # ✅ Ensure uploaded folder exists

            # Save the combined image
            output_image_path = generate_unique_filename(
                img_path,
                self.change_bg_config.img_path_folder
            )
            combined.save(output_image_path)

            # ✅ Save the uploaded background for reference
            uploaded_bg_path = generate_unique_filename(
                bg_img_path,
                self.change_bg_config.uploaded_path_folder
            )
            shutil.copy(bg_img_path, uploaded_bg_path)

            logging.info(f"Background changed and saved to: {output_image_path}")
            logging.info(f"Uploaded background saved to: {uploaded_bg_path}")

            return ChangeBgArtifact(ch_bg_img_path=output_image_path)

        except Exception as e:
            logging.error(f"Failed to change background: {e}")
            raise CustomException(e, sys)
