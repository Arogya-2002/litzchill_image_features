from src.exceptions import CustomException
from src.logger import logging

from src.components.sketch_effects import AnimeSketch,PencilSketch,GlitchEffect

import os,sys

from src.entity.artifact import AnimeGANArtifact, GlitchEffectArtifact, PencilSketchArtifact


pencil_sketch = PencilSketch()
anime_sketch = AnimeSketch()
glitch_effect = GlitchEffect()

def run_pencil_sketch(image_path: str) -> PencilSketchArtifact:
    try:
        logging.info("Running pencil sketch transformation")
        return pencil_sketch.initiate_pencil_sketch(image_path)
    except Exception as e:
        raise CustomException(e, sys)
    
def run_anime_sketch(image_path: str) -> AnimeGANArtifact:
    try:
        logging.info("Running anime sketch transformation")
        return anime_sketch.initiate_anime_sketch(image_path)
    except Exception as e:
        raise CustomException(e, sys)
    
def run_glitch_effect(image_path: str) -> GlitchEffectArtifact:
    try:
        logging.info("Running glitch effect transformation")
        return glitch_effect.initiate_glitch_effect(image_path)
    except Exception as e:
        raise CustomException(e, sys)