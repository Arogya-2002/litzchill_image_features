from dataclasses import dataclass

@dataclass
class RemoveBgArtifact:
    rmbg_img_path:str

@dataclass
class ChangeBgArtifact:
    ch_bg_img_path: str


@dataclass
class ImageUpscalerArtifact:
    upscaled_image_path: str


@dataclass
class ModelInitializationArtifact:
    model_name: str

@dataclass
class SwapperModelArtifact:
    result_image_path: str


@dataclass
class PencilSketchArtifact:
    blur_sigma_10_img_path: str
    blur_sigma_8_img_path: str
    custom_kernel_img_path: str


@dataclass
class AnimeGANArtifact:
    model_hayao_img_path: str
    model_paprika_img_path: str
    model_shinkai_img_path: str

@dataclass
class GlitchEffectArtifact:
    glitch_effect_img_path: str


@dataclass
class InpaintEntityArtifact:
    out_put_path:str