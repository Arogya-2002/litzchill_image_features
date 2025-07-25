from src.exceptions import CustomException
from src.logger import logging
import os
import sys
import torch
import gc
from PIL import Image
from src.entity.config import ImageUpscalerConfig, ConfigEntity

class ImageUtils:
    def __init__(self):
        self.image_upscaler_config = ImageUpscalerConfig(config=ConfigEntity())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")


    def tile_image(self, image, tile_size=512):
        """Split image into tiles for processing large images."""
        try:
            width, height = image.size
            tiles = []
            positions = []

            for y in range(0, height, tile_size - self.image_upscaler_config.tile_overlap):
                for x in range(0, width, tile_size - self.image_upscaler_config.tile_overlap):
                    x_end = min(x + tile_size, width)
                    y_end = min(y + tile_size, height)

                    tile = image.crop((x, y, x_end, y_end))
                    tiles.append(tile)
                    positions.append((x, y, x_end, y_end))

            logging.info(f"Split image of size {image.size} into {len(tiles)} tiles.")
            return tiles, positions
        except Exception as e:
            logging.error("Failed to tile the image.")
            raise CustomException(e, sys)


    def merge_tiles(self, tiles, positions, original_size, scale_factor=4):
        """Merge upscaled tiles back into a single image."""
        try:
            output_width = int(original_size[0] * scale_factor)
            output_height = int(original_size[1] * scale_factor)
            result = Image.new('RGB', (output_width, output_height))
            logging.info(f"Creating merged image of size ({output_width}, {output_height}).")

            for tile, (x, y, x_end, y_end) in zip(tiles, positions):
                out_x = int(x * scale_factor)
                out_y = int(y * scale_factor)
                out_x_end = int(x_end * scale_factor)
                out_y_end = int(y_end * scale_factor)

                expected_width = out_x_end - out_x
                expected_height = out_y_end - out_y

                if tile.size != (expected_width, expected_height):
                    tile = tile.resize((expected_width, expected_height), Image.LANCZOS)
                    logging.debug(f"Resized tile to ({expected_width}, {expected_height}) for pasting.")

                result.paste(tile, (out_x, out_y))

            logging.info("Successfully merged all tiles into final image.")
            return result
        except Exception as e:
            logging.error("Failed to merge tiles into final image.")
            raise CustomException(e, sys)

    def clear_gpu_memory(self):
            """Clear GPU memory to prevent OOM errors."""
            try:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                    logging.info("Cleared GPU memory.")
            except Exception as e:
                logging.error("Failed to clear GPU memory.")
                raise CustomException(e, sys)


    def resize_if_too_large(self,image):
            """Resize image if it's too large for processing."""
            try:
                width, height = image.size
                max_dim = max(width, height)

                if max_dim > self.image_upscaler_config.max_size:
                    scale_factor = self.image_upscaler_config.max_size / max_dim
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    logging.info(f"Resizing image from {image.size} to ({new_width}, {new_height}) to fit memory.")
                    return image.resize((new_width, new_height), Image.LANCZOS), scale_factor

                logging.info("Image size is within acceptable range; no resizing needed.")
                return image, 1.0
            except Exception as e:
                logging.error("Failed to resize image.")
                raise CustomException(e, sys)
