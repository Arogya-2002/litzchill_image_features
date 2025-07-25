
####################bg add and remove constants################
import os
from typing import Dict, Set
# Root artifact directory
ARTIFACTS_DIR = "artifacts"

# Subdirectories
REMOVED_BG_DIR = os.path.join(ARTIFACTS_DIR, "removed_bg_images")
CHANGED_BG_DIR = os.path.join(ARTIFACTS_DIR, "changed_bg_images")
UPLOADED_BG_DIR = os.path.join(ARTIFACTS_DIR, "user_uploaded_bg_images")

# Image filenames
RMBG_IMG_NAME = "removed_bg.png"
BG_IMG_NAME = "background.png"
CH_BG_IMG_NAME = "changed_bg.png"

# Default path folder (can be anything general)
IMG_PATH_FOLDER = ARTIFACTS_DIR


###################Image upscaler constants#####################
UPSCALER_MODEL_NAME = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
MAX_SIZE = 1024
TILE_OVERLAP = 64
OUTPUT_DIR = "artifacts"

#######################Image face swap constants#######################
FACE_SWAP_MODEL_NAME = "buffalo_l"

SWAPPER_MODEL_DIR ="weights/inswapper_128.onnx"

CTX_ID = 0  # Use 0 for CPU, 0+ for GPU
DET_SIZE = (640, 640)  # Detection size for the face analysis model
RESULT_IMAGE_DIR = "results"
DIRECT_URL ="https://drive.usercontent.google.com/download?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF&export=download&authuser=0"

####################Meme style changer constants#######################

import numpy as np


CUSTOM_KERNEL = np.array([[-1, -1, -1], 
                            [-1, 9, -1], 
                            [-1, -1, -1]])


PENCIL_SKETCH_DIR = "pencil_sketch"
GLITCH_EFFECT_DIR = "glitch_effect"
ANIME_SKETCH_DIR = "anime_sketch"
MODEL_DIR = "weights"
ANIME_GAN_HAYAO = "AnimeGANv2_Hayao.onnx"
ANIME_GAN_PAPRIKA = "AnimeGANv2_Paprika.onnx"
ANIME_GAN_SHINKAI = "AnimeGANv2_Shinkai.onnx"
DOWNSIZE_RATIO = 1.0

PROTOXT_PATH = "deploy.prototxt"
CAFFEMODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"


####################Inpainting constants########################

CHECKPOINT_PATH ="weights/states_tf_places2.pth"

##################### Validation constants #######################
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Validation Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
ALLOWED_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
ALLOWED_MIME_TYPES: Set[str] = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 
    'image/tiff', 'image/webp'}