from src.exceptions import CustomException
from src.logger import logging
import sys
import uuid

from src.constants import *

class ConfigEntity:
    def __init__(self):
        self.rmbg_img_name = RMBG_IMG_NAME
        self.bg_img_name = BG_IMG_NAME
        self.ch_bg_img_name = CH_BG_IMG_NAME
        self.img_path_folder = IMG_PATH_FOLDER
        self.removed_bg_dir = REMOVED_BG_DIR
        self.changed_bg_dir = CHANGED_BG_DIR
        self.uploaded_bg_dir = UPLOADED_BG_DIR
        self.upscaler_model_name = UPSCALER_MODEL_NAME
        self.max_size = MAX_SIZE
        self.tile_overlap = TILE_OVERLAP
        self.output_dir = OUTPUT_DIR
        self.face_swap_model_name = FACE_SWAP_MODEL_NAME
        self.swapper_model_dir = SWAPPER_MODEL_DIR
        self.ctx_id = CTX_ID
        self.det_size = DET_SIZE
        self.result_image_dir= RESULT_IMAGE_DIR
        self.custom_kernel = CUSTOM_KERNEL
        self.pencil_sketch_dir = PENCIL_SKETCH_DIR
        self.model_dir = MODEL_DIR
        self.anime_gan_hayao = ANIME_GAN_HAYAO
        self.anime_gan_paprika = ANIME_GAN_PAPRIKA
        self.anime_gan_shinkai = ANIME_GAN_SHINKAI
        self.anime_sketch_dir = ANIME_SKETCH_DIR
        self.downsize_ratio = DOWNSIZE_RATIO
        self.glitch_effect_dir = GLITCH_EFFECT_DIR
        self.check_point_path = CHECKPOINT_PATH
        self.inpaint_output_dir = os.path.join(OUTPUT_DIR,f"{uuid.uuid4().hex}.jpg")




class RemoveBgConfig:
    def __init__(self,bg_config:ConfigEntity):
        self.rmbg_img_name = bg_config.rmbg_img_name
        self.bg_img_name = bg_config.bg_img_name
        self.img_path_folder = bg_config.removed_bg_dir


class ChangeBgConfig:
    def __init__(self, bg_config: ConfigEntity):
        self.ch_bg_img_name = bg_config.ch_bg_img_name
        self.img_path_folder = bg_config.changed_bg_dir
        self.bg_img_name = bg_config.bg_img_name
        self.uploaded_path_folder: str = bg_config.uploaded_bg_dir


class ImageUpscalerConfig:
    def __init__(self,config: ConfigEntity):
        self.upscaler_model_name = config.upscaler_model_name
        self.max_size = config.max_size
        self.tile_overlap = config.tile_overlap
        self.output_dir = config.output_dir 

class ModelInitializerConfig:
    def __init__(self,config: ConfigEntity):
        self.face_swap_model_name = config.face_swap_model_name
        self.ctx_id = config.ctx_id
        self.det_size = config.det_size

class SwapperModelConfig:
    def __init__(self, config: ConfigEntity):
        self.swapper_model_dir = config.swapper_model_dir
        self.output_dir = config.output_dir
        self.result_image_dir = config.result_image_dir


class PencilSketchConfig:
    def __init__(self,config: ConfigEntity):
        self.custom_kernel = config.custom_kernel
        self.output_dir = config.output_dir
        self.pencil_sketch_dir = config.pencil_sketch_dir



class AnimeGANConfig:
    def __init__(self,config: ConfigEntity):
        self.model_hayao_path = os.path.join(config.model_dir, config.anime_gan_hayao)
        self.model_paprika_path = os.path.join(config.model_dir, config.anime_gan_paprika)
        self.model_shinkai_path = os.path.join(config.model_dir, config.anime_gan_shinkai)
        self.anime_sketch_dir = config.anime_sketch_dir
        self.model_dir = config.model_dir
        self.downsize_ratio = config.downsize_ratio
        self.output_dir = config.output_dir



class GlitchEffectConfig:
    def __init__(self,config:ConfigEntity):
        self.deploy_protox_apth = os.path.join(config.model_dir, "deploy.prototxt.txt")
        self.deploy_caffemodel_path = os.path.join(config.model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        self.shape_predictor_path = os.path.join(config.model_dir, "shape_predictor_68_face_landmarks.dat")
        self.output_dir = config.output_dir
        self.glitch_effect_dir = config.glitch_effect_dir