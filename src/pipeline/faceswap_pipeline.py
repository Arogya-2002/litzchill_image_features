import cv2
import sys
from src.components.faceswap import FaceSwap,ModelInitializer
from src.logger import logging
from src.exceptions import CustomException

def initiate_face_swapper(multi_face_img_path:str,single_face_img_path:str):
    try:
        logging.info("=== Starting Face Swap Pipeline ===")

        # Step 1: Initialize FaceAnalysis model
        logging.info("Initializing FaceAnalysis model...")
        model_initializer = ModelInitializer()
        face_analysis_app = model_initializer.initialize_model()

        # Step 2: Load input images
        logging.info("Loading input images...")

        img_multi_faces = cv2.imread(multi_face_img_path)
        img_single_face = cv2.imread(single_face_img_path)

        if img_multi_faces is None:
            raise FileNotFoundError(f"Multi-face image not found: {multi_face_img_path}")
        if img_single_face is None:
            raise FileNotFoundError(f"Single-face image not found: {single_face_img_path}")

        logging.info("Input images loaded successfully.")

        # Step 3: Perform face swapping
        face_swapper = FaceSwap()
        artifact = face_swapper.perform_face_swapping(
            app=face_analysis_app,
            img_multi_faces=img_multi_faces,
            img_single_face=img_single_face
        )

        logging.info(f"Face swap completed. Result saved at: {artifact.result_image_path}")

        logging.info("=== Pipeline Completed Successfully ===")
        return artifact.result_image_path

    except Exception as e:
        logging.error("Pipeline execution failed.", exc_info=True)
        raise CustomException(e, sys) from e

# if __name__ == "__main__":
#     initiate_face_swapper(multi_face_img_path="/Users/vamshi/Desktop/image_tasks/data/multiimages.jpg",
#                            single_face_img_path="/Users/vamshi/Desktop/image_tasks/data/brahmi.jpg")