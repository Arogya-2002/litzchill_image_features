from src.exceptions import CustomException
from src.logger import logging

from src.entity.config import ConfigEntity, SwapperModelConfig, ModelInitializerConfig
from src.entity.artifact import SwapperModelArtifact, ModelInitializationArtifact
from src.utils import download_weights_from_google_drive

import insightface
import sys
import cv2
import os
from datetime import datetime
from insightface.app import FaceAnalysis

class ModelInitializer:
    def __init__(self):
        try:
            logging.info("Creating ModelInitializerConfig...")
            self.model_initializer_config = ModelInitializerConfig(config=ConfigEntity())
        except Exception as e:
            logging.error("Failed to initialize ModelInitializerConfig", exc_info=True)
            raise CustomException(e, sys) from e

    def initialize_model(self) -> FaceAnalysis:
        try:
            logging.info("Starting FaceAnalysis model initialization...")

            app = FaceAnalysis(name=self.model_initializer_config.face_swap_model_name)
            app.prepare(
                ctx_id=self.model_initializer_config.ctx_id,
                det_size=self.model_initializer_config.det_size
            )

            logging.info("FaceAnalysis model initialized successfully.")

            # Create artifact for tracking (if needed later)
            artifact = ModelInitializationArtifact(
                model_name=self.model_initializer_config.face_swap_model_name
            )
            logging.info(f"ModelInitializationArtifact created: {artifact}")

            return app

        except Exception as e:
            logging.error("Error during model initialization", exc_info=True)
            raise CustomException(e, sys) from e




class FaceSwap:
    def __init__(self):
        try:
            logging.info("Creating SwapperModelConfig...")
            self.swapper_model_config = SwapperModelConfig(config=ConfigEntity())
            logging.info(f"SwapperModelConfig initialized with model path: {self.swapper_model_config.swapper_model_dir}")
        except Exception as e:
            logging.error("Failed to initialize SwapperModelConfig", exc_info=True)
            raise CustomException(e, sys) from e

    def perform_face_swapping(self, app, img_multi_faces, img_single_face) -> SwapperModelArtifact:
        try:
            # Load model
            model_path = self.swapper_model_config.swapper_model_dir
            if not os.path.exists(model_path):
                download_weights_from_google_drive()
            
            logging.info(f"Loading face swapper model from: {model_path}")
            swapper = insightface.model_zoo.get_model(model_path, download=False)
            logging.info("Face swapper model loaded successfully.")

            # Preprocess images
            logging.info("Converting images to RGB format...")
            img_multi_faces_rgb = cv2.cvtColor(img_multi_faces, cv2.COLOR_BGR2RGB)
            img_single_face_rgb = cv2.cvtColor(img_single_face, cv2.COLOR_BGR2RGB)

            # Detect faces
            logging.info("Detecting faces in multi-face image...")
            faces_multi = app.get(img_multi_faces_rgb)
            logging.info("Detecting face in single-face image...")
            faces_single = app.get(img_single_face_rgb)

            if not faces_multi:
                logging.warning("No faces detected in the multi-face image.")
                raise ValueError("No faces detected in the multi-face image!")
            if not faces_single:
                logging.warning("No faces detected in the single-face image.")
                raise ValueError("No faces detected in the single-face image!")

            logging.info(f"{len(faces_multi)} face(s) detected in multi-face image.")
            logging.info(f"{len(faces_single)} face(s) detected in single-face image.")

            # Perform face swapping
            source_face = faces_single[0]
            result_image = img_multi_faces_rgb.copy()
            logging.info("Performing face swap...")
            for idx, face in enumerate(faces_multi):
                logging.info(f"Swapping face {idx + 1}...")
                result_image = swapper.get(result_image, face, source_face, paste_back=True)

            # Save result image
            if not self.swapper_model_config.output_dir or not self.swapper_model_config.result_image_dir:
                raise ValueError("Output directory or result image directory is not configured properly.")

            result_dir = os.path.join(self.swapper_model_config.output_dir, self.swapper_model_config.result_image_dir)

            os.makedirs(result_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_image_path = os.path.join(result_dir, f"swapped_face_{timestamp}.jpg")

            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
            cv2.imwrite(result_image_path, result_bgr)
            logging.info(f"Swapped image saved to: {result_image_path}")

            # Create and return artifact
            return SwapperModelArtifact(result_image_path=result_image_path)

        except Exception as e:
            logging.error("Error during face swapping operation", exc_info=True)
            raise CustomException(e, sys) from e
