# utils.py
import os
from pathlib import Path
import time,io
from datetime import datetime
import uuid
import os
import requests
from tqdm import tqdm
import re
from urllib.parse import urlencode
from src.constants import *

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, status
from PIL import Image
import shutil 
from src.logger import logging

def generate_unique_filename(input_image_path: str, output_folder: str) -> str:
    """
    Generates a unique filename for the background removed image.
    
    Parameters:
        input_image_path (str): The path of the original input image.
        output_folder (str): The folder where the processed image will be saved.
    
    Returns:
        str: A unique output file path.
    """
    input_stem = Path(input_image_path).stem  # Extracts the base filename without extension
    timestamp = int(time.time())  # Current time in seconds
    unique_name = f"{input_stem}_rmbg_{timestamp}.png"
    return os.path.join(output_folder, unique_name)






def download_weights(url, save_path):
    """
    Download pre-trained model weights from a Google Drive URL and save them locally.
    
    Args:
        url (str): Direct URL to download the weights from.
        save_path (str): Local path to save the weights.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"Weights already exist at {save_path}")
        return

    try:
        session = requests.Session()
        response = session.get(url, allow_redirects=True)

        if "Virus scan warning" in response.text or "Download anyway" in response.text:
            confirm_match = re.search(r'<input type="hidden" name="confirm" value="([^"]+)"', response.text)
            uuid_match = re.search(r'<input type="hidden" name="uuid" value="([^"]+)"', response.text)
            id_match = re.search(r'<input type="hidden" name="id" value="([^"]+)"', response.text)
            export_match = re.search(r'<input type="hidden" name="export" value="([^"]+)"', response.text)
            authuser_match = re.search(r'<input type="hidden" name="authuser" value="([^"]+)"', response.text)

            if confirm_match and uuid_match:
                form_data = {
                    "id": id_match.group(1) if id_match else "1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF",
                    "export": export_match.group(1) if export_match else "download",
                    "authuser": authuser_match.group(1) if authuser_match else "0",
                    "confirm": confirm_match.group(1),
                    "uuid": uuid_match.group(1)
                }
                print(f"Extracted form parameters: {form_data}")
                
                form_action = "https://drive.usercontent.google.com/download"
                download_url = f"{form_action}?{urlencode(form_data)}"

                response = session.get(download_url, stream=True, allow_redirects=True)
            else:
                print("Could not extract required form parameters.")
                print(f"Confirm: {confirm_match.group(1) if confirm_match else 'Not found'}")
                print(f"UUID: {uuid_match.group(1) if uuid_match else 'Not found'}")
                return

        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            if total_size == 0:
                print("Warning: Content-length not provided, progress bar may not be accurate.")

            with open(save_path, "wb") as file, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

            print(f"Weights downloaded successfully and saved at {save_path}")
        else:
            print(f"Failed to download weights. Status code: {response.status_code}")
            print(f"Response content (first 500 chars): {response.text[:500]}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def download_weights_from_google_drive():
    # Google Drive download URL
    direct_url = DIRECT_URL 
    # Determine the project root from src/utils/__init__.py
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))

    # Target weights directory and file path
    weights_dir = os.path.join(project_root, "weights")
    weights_filename = "inswapper_128.onnx"
    save_path = os.path.join(weights_dir, weights_filename)

    # Attempt download
    print(f"Attempting to download from: {direct_url}")
    print(f"Saving to: {save_path}")
    download_weights(direct_url, save_path)




def generate_unique_filename1(prefix: str = "file", ext: str = ".jpg", directory: str = "") -> str:
    """
    Generate a unique .jpg filename using current date, time, and UUID suffix.

    Args:
        prefix (str): Prefix for the filename (e.g., "blur_sigma_10").
        ext (str): File extension (default: '.jpg'). Can be '.jpeg' too.
        directory (str): Directory to place the file in. Created if it doesn't exist.

    Returns:
        str: Full path to the uniquely named file.
    """
    # Normalize extension
    ext = ext.lower()
    if ext not in [".jpg", ".jpeg"]:
        ext = ".jpg"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{prefix}_{timestamp}_{unique_id}{ext}"

    if directory:
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)
    
    return filename



def validate_file_size(file: UploadFile) -> None:
    """Validate file size doesn't exceed maximum allowed size."""
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({file.size} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE} bytes)"
        )

def validate_file_extension(filename: str) -> None:
    """Validate file has an allowed extension."""
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File extension '{extension}' not allowed. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )

def validate_mime_type(file: UploadFile) -> None:
    """Validate file MIME type."""
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"MIME type '{file.content_type}' not allowed. Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"
        )

def validate_image_content(file_path: str) -> None:
    """Validate that the file is actually a valid image by trying to open it."""
    try:
        with Image.open(file_path) as img:
            # Verify the image by loading it
            img.verify()
        
        # Re-open for format validation since verify() closes the file
        with Image.open(file_path) as img:
            if img.format.lower() not in ['jpeg', 'png', 'bmp', 'tiff', 'webp']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image format '{img.format}' is not supported"
                )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )

def validate_image_dimensions(file_path: str, max_width: int = 4096, max_height: int = 4096) -> None:
    """Validate image dimensions don't exceed maximum allowed dimensions."""
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            if width > max_width or height > max_height:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image dimensions ({width}x{height}) exceed maximum allowed dimensions ({max_width}x{max_height})"
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error validating image dimensions: {str(e)}"
        )

def comprehensive_image_validation(file: UploadFile, temp_path: str) -> None:
    """Perform all image validations."""
    validate_file_size(file)
    validate_file_extension(file.filename)
    validate_mime_type(file)
    validate_image_content(temp_path)
    validate_image_dimensions(temp_path)

# ----------------------------------
# Utils
# ----------------------------------
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

async def save_file_with_validation(file: UploadFile, label: str) -> str:
    """Save file with comprehensive validation."""
    # Read file content
    file_content = await file.read()
    
    # Check file size from content if not available from file object
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({len(file_content)} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE} bytes)"
        )
    
    # Validate file extension and MIME type
    validate_file_extension(file.filename)
    validate_mime_type(file)
    
    # Save with UUID
    ext = os.path.splitext(file.filename)[-1].lower() if file.filename else ".png"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(ARTIFACT_DIR, unique_name)
    with open(save_path, "wb") as f:
        f.write(file_content)

    # Validate image content and dimensions
    validate_image_content(save_path)
    validate_image_dimensions(save_path)

    logging.info(f"{label} saved at {save_path}")
    return save_path

async def save_file_no_validation(file: UploadFile, label: str) -> str:
    """Legacy function - now with validation for security."""
    return await save_file_with_validation(file, label)

def get_temp_path(prefix: str, filename: str) -> str:
    return os.path.join(TEMP_UPLOAD_DIR, f"{prefix}_{uuid.uuid4().hex}_{filename}")

def save_file_validated(file: UploadFile, path: str):
    """Save file with validation."""
    # Perform initial validations
    validate_file_extension(file.filename)
    validate_mime_type(file)
    
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Validate the saved file
    comprehensive_image_validation(file, path)

def cleanup_file(path: str):
    if os.path.exists(path):
        os.remove(path)