from src.exceptions import CustomException
from src.logger import logging
from src.entity.config import ImageUpscalerConfig, ConfigEntity
from src.entity.artifact import ImageUpscalerArtifact
from src.utils.common_functions import ImageUtils
import os
import sys
import torch
from PIL import Image
import numpy as np
from transformers import pipeline
import gc
import requests
from io import BytesIO


class ImageUpscaler:
    def __init__(self):
        try:
            self.image_upscaler_config = ImageUpscalerConfig(config=ConfigEntity())
            self.image_utils = ImageUtils()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Using device: {self.device}")

            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            try:
                self.upscaler = pipeline(
                    "image-to-image",
                    model=self.image_upscaler_config.upscaler_model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.upscaler_model_name = self.image_upscaler_config.upscaler_model_name
                logging.info(f"Loaded model: {self.upscaler_model_name}")
            except Exception as e:
                logging.warning(f"Failed to load model '{self.image_upscaler_config.upscaler_model_name}', trying fallbacks. Error: {e}")
                fallback_models = [
                    "caidas/swin2SR-classical-sr-x2-64",
                    "microsoft/swin2SR-classical-sr-x2-48"
                ]
                for fallback in fallback_models:
                    try:
                        self.upscaler = pipeline(
                            "image-to-image",
                            model=fallback,
                            device=0 if self.device == "cuda" else -1,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                        )
                        self.upscaler_model_name = fallback
                        logging.info(f"Loaded fallback model: {fallback}")
                        break
                    except Exception as fallback_e:
                        logging.error(f"Fallback model '{fallback}' also failed: {fallback_e}")
                else:
                    raise CustomException("All model loading attempts failed.", sys)

        except Exception as e:
            raise CustomException(e, sys)
        
    def upscale_image(self, image_path, use_tiling=True):
        try:
            # Load image
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
                logging.info(f"Loaded image from URL: {image_path}")
            else:
                image = Image.open(image_path)
                logging.info(f"Loaded image from disk: {image_path}")

            if image.mode != 'RGB':
                image = image.convert('RGB')

            logging.info(f"Original image size: {image.size}")
            original_size = image.size
            self.image_utils.clear_gpu_memory()

            processed_image, scale_factor = self.image_utils.resize_if_too_large(image)

            try:
                logging.info("Attempting to upscale image (resize if needed)...")
                upscaled_image = self.upscaler(processed_image)

                if scale_factor < 1.0:
                    target_size = (int(original_size[0] * 4), int(original_size[1] * 4))  # Assuming 4x upscale
                    upscaled_image = upscaled_image.resize(target_size, Image.LANCZOS)
                    logging.info(f"Rescaled to original size: {target_size}")

                logging.info(f"Upscaled image size: {upscaled_image.size}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and use_tiling:
                    logging.warning("OOM error encountered. Switching to tiling...")
                    self.image_utils.clear_gpu_memory()
                    tiles, positions = self.image_utils.tile_image(image, tile_size=512)
                    logging.info(f"Split image into {len(tiles)} tiles")

                    upscaled_tiles = []
                    for i, tile in enumerate(tiles):
                        logging.info(f"Upscaling tile {i + 1}/{len(tiles)}")
                        try:
                            upscaled_tile = self.upscaler(tile)
                            upscaled_tiles.append(upscaled_tile)
                        except RuntimeError as tile_e:
                            if "out of memory" in str(tile_e).lower():
                                logging.warning(f"Tile {i+1} OOM on GPU, using CPU fallback...")
                                cpu_upscaler = pipeline(
                                    "image-to-image",
                                    model=self.upscaler_model_name,
                                    device=-1
                                )
                                upscaled_tile = cpu_upscaler(tile)
                                upscaled_tiles.append(upscaled_tile)
                                del cpu_upscaler
                            else:
                                raise tile_e
                        self.image_utils.clear_gpu_memory()

                    logging.info("Merging tiles...")
                    upscaled_image = self.image_utils.merge_tiles(upscaled_tiles, positions, original_size)
                    logging.info(f"Final upscaled image size: {upscaled_image.size}")

                else:
                    logging.warning("OOM or unexpected error, falling back to CPU...")
                    self.image_utils.clear_gpu_memory()
                    cpu_upscaler = pipeline(
                        "image-to-image",
                        model=self.upscaler_model_name,
                        device=-1
                    )
                    upscaled_image = cpu_upscaler(processed_image)
                    del cpu_upscaler

            output_path = os.path.join(self.image_upscaler_config.output_dir,
                                       f"upscaled_{os.path.basename(image_path)}")
            os.makedirs(self.image_upscaler_config.output_dir, exist_ok=True)
            logging.info(f"Saving upscaled image to: {output_path}")
            upscaled_image.save(output_path, quality=95, optimize=True)
            logging.info(f"Upscaled image saved to: {output_path}")

            return ImageUpscalerArtifact(upscaled_image_path=output_path)

        except Exception as e:
            logging.error(f"Failed during image upscaling: {e}")
            raise CustomException(e, sys)
        finally:
            self.image_utils.clear_gpu_memory()


