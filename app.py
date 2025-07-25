from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, status
import shutil
import os
import io
import uuid
import logging
import uvicorn
import sys

import tempfile
from pathlib import Path
import imghdr
from typing import Dict, Set
import logging
import io
import uuid
import urllib.parse
from PIL import Image
import zipfile

from src.constants import *

from src.pipeline.bg_prediction_pipeline import process_remove_bg, process_add_bg
from src.pipeline.upscaler_pipeline import initialte_upscaler_pipeline
from src.pipeline.faceswap_pipeline import initiate_face_swapper
from src.pipeline.inpaint_pipeline import inpaint_image
from src.pipeline.meme_style_transfer_pipeline import (
    run_pencil_sketch,
    run_anime_sketch,
    run_glitch_effect
)
from src.entity.artifact import (
    PencilSketchArtifact,
    AnimeGANArtifact,
    GlitchEffectArtifact
)
from src.exceptions import CustomException

from src.utils import get_temp_path, save_file_validated, get_temp_path, cleanup_file
from src.utils import validate_file_extension, validate_mime_type, save_file_with_validation,validate_image_content,validate_image_dimensions

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
@app.get("/", response_class=HTMLResponse)
def root():
    return "<h1>FastAPI Image Processing Service</h1>"

@app.post("/remove-background/")
def remove_background(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    temp_input = get_temp_path("input", file.filename)
    output_path = None

    try:
        save_file_validated(file, temp_input)
        output_path = process_remove_bg(temp_input)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Background removal failed. Output not found.")

        with open(output_path, "rb") as f:
            image_bytes = f.read()

        background_tasks.add_task(cleanup_file, temp_input)
        background_tasks.add_task(cleanup_file, output_path)

        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=no_bg_{file.filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in remove-background: {e}")
        cleanup_file(temp_input)
        if output_path:
            cleanup_file(output_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-background/")
def add_background(background_tasks: BackgroundTasks,
                   foreground: UploadFile = File(...),
                   background: UploadFile = File(...)):
    temp_fg = get_temp_path("fg", foreground.filename)
    temp_bg = get_temp_path("bg", background.filename)
    output_path = None

    try:
        save_file_validated(foreground, temp_fg)
        save_file_validated(background, temp_bg)

        output_path = process_add_bg(temp_fg, temp_bg)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Add background failed. Output not found.")

        with open(output_path, "rb") as f:
            image_bytes = f.read()

        for path in [temp_fg, temp_bg, output_path]:
            background_tasks.add_task(cleanup_file, path)

        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=with_bg_{foreground.filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in add-background: {e}")
        for path in [temp_fg, temp_bg, output_path]:
            if path:
                cleanup_file(path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upscale")
def upscale_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    temp_input = get_temp_path("upscale", file.filename)

    try:
        save_file_validated(file, temp_input)
        logging.info(f"Saved input for upscaling: {temp_input}")

        output_path = initialte_upscaler_pipeline(temp_input)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Upscaling failed. Output not found.")

        with open(output_path, "rb") as f:
            image_bytes = f.read()

        background_tasks.add_task(cleanup_file, temp_input)
        background_tasks.add_task(cleanup_file, output_path)

        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename=upscaled_{file.filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error("Error during upscale:", exc_info=True)
        cleanup_file(temp_input)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/face-swap")
def face_swap(background_tasks: BackgroundTasks,
              multi_face_img: UploadFile = File(...),
              single_face_img: UploadFile = File(...)):
    multi_path = get_temp_path("multi", multi_face_img.filename)
    single_path = get_temp_path("single", single_face_img.filename)
    output_path = None

    try:
        save_file_validated(multi_face_img, multi_path)
        save_file_validated(single_face_img, single_path)

        output_path = initiate_face_swapper(
            multi_face_img_path=multi_path,
            single_face_img_path=single_path
        )

        if not output_path or not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Face swap failed. Output not found.")

        with open(output_path, "rb") as f:
            image_bytes = f.read()

        background_tasks.add_task(cleanup_file, multi_path)
        background_tasks.add_task(cleanup_file, single_path)
        background_tasks.add_task(cleanup_file, output_path)

        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename={os.path.basename(output_path)}"}
        )

    except HTTPException:
        raise
    except CustomException as ce:
        logging.error("Custom error during face-swap", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(ce)})

    except Exception as e:
        logging.error("Unhandled error during face-swap", exc_info=True)
        raise CustomException(e, sys)

    finally:
        # Safety net if StreamingResponse fails to run cleanup
        for path in [multi_path, single_path, output_path]:
            if path:
                cleanup_file(path)

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file with validation."""
    extension = Path(file.filename).suffix.lower() if file.filename else ".png"
    
    # Validate before saving
    validate_file_extension(file.filename)
    validate_mime_type(file)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension, dir=TEMP_UPLOAD_DIR)

    try:
        # Check file size during copy
        total_size = 0
        chunk_size = 8192
        
        while chunk := file.file.read(chunk_size):
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File size exceeds maximum allowed size ({MAX_FILE_SIZE} bytes)"
                )
            temp_file.write(chunk)
        
        temp_file.close()
        
        # Validate image content and dimensions
        validate_image_content(temp_file.name)
        validate_image_dimensions(temp_file.name)
        
        return temp_file.name
    except HTTPException:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise
    except Exception as e:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

def image_to_bytesio(image_path: str) -> io.BytesIO:
    try:
        with Image.open(image_path) as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            return img_byte_arr
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to convert image: {str(e)}")

def create_zip_with_images(image_paths: Dict[str, str], zip_name: str) -> io.BytesIO:
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, image_path in image_paths.items():
                if os.path.exists(image_path):
                    zip_file.write(image_path, f"{filename}.png")
        zip_buffer.seek(0)
        return zip_buffer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP file: {str(e)}")

@app.post("/transform/pencil-sketch")
async def transform_pencil_sketch(image: UploadFile = File(...)):
    temp_file_path = None
    try:
        temp_file_path = await save_uploaded_file(image)
        artifact = run_pencil_sketch(temp_file_path)
        image_paths = {
            "blur_sigma_10": artifact.blur_sigma_10_img_path,
            "blur_sigma_8": artifact.blur_sigma_8_img_path,
            "custom_kernel": artifact.custom_kernel_img_path
        }
        zip_buffer = create_zip_with_images(image_paths, "pencil_sketch")
        filename = f"pencil_sketch_{uuid.uuid4().hex[:8]}.zip"
        return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{urllib.parse.quote(filename)}"'})
    except HTTPException:
        raise
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/transform/anime-sketch")
async def transform_anime_sketch(image: UploadFile = File(...)):
    temp_file_path = None
    try:
        temp_file_path = await save_uploaded_file(image)
        artifact = run_anime_sketch(temp_file_path)
        image_paths = {
            "model_hayao": artifact.model_hayao_img_path,
            "model_paprika": artifact.model_paprika_img_path,
            "model_shinkai": artifact.model_shinkai_img_path
        }
        zip_buffer = create_zip_with_images(image_paths, "anime_sketch")
        filename = f"anime_sketch_{uuid.uuid4().hex[:8]}.zip"
        return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{urllib.parse.quote(filename)}"'})
    except HTTPException:
        raise
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/transform/glitch-effect")
async def transform_glitch_effect(image: UploadFile = File(...)):
    temp_file_path = None
    try:
        temp_file_path = await save_uploaded_file(image)
        artifact = run_glitch_effect(temp_file_path)
        image_bytesio = image_to_bytesio(artifact.glitch_effect_img_path)
        filename = f"glitch_effect_{uuid.uuid4().hex[:8]}.png"
        return StreamingResponse(image_bytesio, media_type="image/png", headers={"Content-Disposition": f'attachment; filename="{urllib.parse.quote(filename)}"'})
    except HTTPException:
        raise
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/api/v1/inpaint/")
async def inpaint_endpoint(image: UploadFile = File(...), mask: UploadFile = File(...)):
    try:
        logging.info("Received inpainting request.")

        # Save both files with validation
        input_image_path = await save_file_with_validation(image, "Input image")
        mask_image_path = await save_file_with_validation(mask, "Mask image")

        # Load model + run inference
        logging.info("Running inpainting model")
        artifact = inpaint_image(input_image_path, mask_image_path)

        logging.info(f"Inpainting done. Output at: {artifact.out_put_path}")
        return FileResponse(artifact.out_put_path, media_type="image/png", filename="inpainted.png")

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unhandled exception during inpainting: {e}", exc_info=True)
        raise CustomException(e, sys)

# ----------------------------------
# Configuration endpoint
# ----------------------------------
@app.get("/validation-config")
def get_validation_config():
    """Get current validation configuration."""
    return {
        "max_file_size_bytes": MAX_FILE_SIZE,
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "allowed_mime_types": list(ALLOWED_MIME_TYPES),
        "max_image_dimensions": "4096x4096"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)