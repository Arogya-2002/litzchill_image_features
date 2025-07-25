# Advanced Image Processing Service

A comprehensive image processing service built with Python and FastAPI that provides intelligent background removal, background replacement, image inpainting, face swapping, image upscaling, and artistic style transfer capabilities. Upload images through REST API endpoints to automatically process images with state-of-the-art AI models.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)  
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)  
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## ✨ Features

- **🎯 Smart Background Removal**: Leverage the powerful U2NET model via `rembg` library for precise background removal  
- **🖼️ Background Replacement**: Seamlessly add custom backgrounds to transparent images  
- **🤖 Face Swapping**: Advanced face swapping technology for replacing faces in images
- **📈 Image Upscaling**: Enhance image resolution using AI-powered upscaling algorithms
- **🎨 Artistic Style Transfer**: Transform images with multiple artistic effects:
  - Pencil Sketch (multiple blur variants)
  - Anime Style (Hayao, Paprika, Shinkai models)
  - Glitch Effects
- **🖌️ Image Inpainting**: Fill masked regions in images intelligently
- **🚀 REST API**: FastAPI-powered endpoints for programmatic access  
- **🛡️ Robust Error Handling**: Comprehensive logging and exception management  
- **🗂️ Smart File Management**: Automatic temporary file cleanup and organized storage
- **📏 File Validation**: Comprehensive validation for file types, sizes, and dimensions
- **⚡ Background Tasks**: Asynchronous cleanup for better performance

---

## 📁 Project Structure

```
.
├── app.py                          # FastAPI application and route definitions
├── src/
│   ├── components/
│   │   ├── inpaint/
│   │   │   └── __init__.py
│   │   ├── faceswap.py            # Face swapping logic
│   │   ├── image_upscaler.py      # Image upscaling functionality
│   │   ├── remove_bg.py           # Background removal logic
│   │   └── sketch_effects.py     # Sketch and artistic effects
│   ├── constants/
│   │   └── __init__.py            # Application constants and configuration
│   ├── entity/
│   │   ├── artifact.py            # Data classes and artifacts
│   │   └── config.py              # Configuration management
│   ├── exceptions/
│   │   └── __init__.py            # Custom exception definitions
│   ├── logger/
│   │   └── __init__.py            # Logging configuration
│   ├── pipeline/
│   │   ├── bg_prediction_pipeline.py      # Background processing pipeline
│   │   ├── faceswap_pipeline.py           # Face swapping pipeline
│   │   ├── inpaint_pipeline.py            # Image inpainting pipeline
│   │   ├── meme_style_transfer_pipeline.py # Style transfer pipeline
│   │   └── upscaler_pipeline.py           # Image upscaling pipeline
│   └── utils/
│       ├── app_utils.py           # Application utilities
│       ├── common_functions.py    # Shared utility functions
│       ├── data.py                # Data handling utilities
│       └── misc.py                # Miscellaneous helpers
├── .gitignore
├── Dockerfile                     # Docker configuration
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher  
- Git  

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Arogya-2002/litzchill_image_features.git
   cd litzchill_image_features
   ```

2. **Set up virtual environment** (recommended)
   ```bash
   # Using conda
   conda create -p venv python==3.11 -y
   conda activate venv/
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note**: The AI models may require additional system dependencies:
   > - **Ubuntu/Debian**: `sudo apt-get install libglib2.0-0 libgl1-mesa-glx`
   > - **macOS**: Usually works out of the box
   > - **Windows**: May require Visual C++ Build Tools

### Running the Application

1. **Start the FastAPI server**
   ```bash
   uvicorn app:app --reload
   ```
   or
   '''bash
   python app.py
   '''

2. **Access the API documentation**
   Open your browser and navigate to: `http://0.0.0.0:8000/docs`

---

## 📖 API Usage

### Core Endpoints

#### `GET /`
Returns basic service information.

#### `POST /remove-background/`
Remove background from an uploaded image.

**Parameters:**
- `file`: Image file (multipart/form-data)

**Response:** Image with background removed (PNG format)

**Example:**
```bash
curl -X POST "http://127.0.0.1:8000/remove-background/" \
  -F "file=@your_image.jpg"
```

#### `POST /add-background/`
Add a new background to a foreground image.

**Parameters:**
- `foreground`: Foreground image file (preferably with transparent background)
- `background`: Background image file

**Response:** Composite image with new background

**Example:**
```bash
curl -X POST "http://127.0.0.1:8000/add-background/" \
  -F "foreground=@foreground.png" \
  -F "background=@background.jpg"
```

### Advanced Processing Endpoints

#### `POST /upscale`
Enhance image resolution using AI upscaling.

**Parameters:**
- `file`: Image file to upscale

**Response:** Upscaled image (JPEG format)

#### `POST /face-swap`
Swap faces between two images.

**Parameters:**
- `multi_face_img`: Image containing the target face(s)
- `single_face_img`: Image containing the source face

**Response:** Image with swapped faces

#### `POST /api/v1/inpaint/`
Fill masked regions in images intelligently.

**Parameters:**
- `image`: Original image file
- `mask`: Mask image (white regions will be inpainted)

**Response:** Inpainted image (PNG format)

### Artistic Style Transfer

#### `POST /transform/pencil-sketch`
Convert image to pencil sketch with multiple variants.

**Response:** ZIP file containing multiple sketch versions

#### `POST /transform/anime-sketch`
Transform image using anime style models.

**Response:** ZIP file with Hayao, Paprika, and Shinkai style variants

#### `POST /transform/glitch-effect`
Apply glitch effects to the image.

**Response:** Single glitch-effect image (PNG format)

### Configuration

#### `GET /validation-config`
Get current file validation settings.

**Response:** JSON with file size limits, allowed formats, and dimension constraints

---

## ⚙️ File Validation & Limits

The service includes comprehensive file validation:

- **Maximum file size**: Configurable (default: varies by endpoint)
- **Supported formats**: JPEG, PNG, WebP, BMP, TIFF
- **Maximum dimensions**: 4096x4096 pixels
- **MIME type validation**: Ensures files match their extensions
- **Content validation**: Verifies actual image content

---

## 🔧 Technical Details

### AI Models & Libraries
- **Background Removal**: U2NET via `rembg`
- **Face Swapping**: Advanced face detection and swapping algorithms
- **Image Upscaling**: AI-powered super-resolution models
- **Style Transfer**: Multiple artistic style models
- **Inpainting**: Deep learning-based image completion

### Framework & Infrastructure
- **Web Framework**: FastAPI with automatic API documentation
- **Image Processing**: Pillow (PIL) for image manipulation
- **File Handling**: Automatic cleanup with background tasks
- **Error Management**: Custom exceptions with detailed logging
- **Validation**: Multi-layer file and content validation
- **Logging**: Structured logging with configurable levels

### Performance Features
- **Asynchronous Processing**: Non-blocking file operations
- **Background Tasks**: Automatic cleanup without blocking responses
- **Memory Management**: Efficient handling of large image files
- **Temporary File Management**: Secure temporary file handling

---

## 🛠️ Development

### Project Architecture

The application follows a modular, pipeline-based architecture:

- **Components**: Core functionality modules for different image processing tasks
- **Entities**: Data classes, artifacts, and configuration management
- **Pipelines**: Main processing workflows that orchestrate components
- **Utils**: Helper functions, validation, and common utilities
- **Exceptions**: Custom error handling and logging

### Adding New Features

1. Implement core processing logic in appropriate `src/components/` module
2. Create data artifacts in `src/entity/artifact.py`
3. Build processing pipeline in `src/pipeline/`
4. Add API endpoints in `app.py`
5. Update validation constants in `src/constants/`

### Error Handling

The service uses a multi-tier error handling approach:
- **Validation Errors**: File format, size, and content validation
- **Processing Errors**: AI model and image processing failures  
- **HTTP Exceptions**: Proper status codes and error messages
- **Custom Exceptions**: Detailed error tracking and logging

---

## 🐳 Docker Support

```bash
# Build the Docker image
docker build -t image-processing-service .

# Run the container
docker run -p 8000:8000 image-processing-service
```

---

## 🚀 Future Roadmap

- [ ] **Batch Processing**: Support multiple image processing in single requests
- [ ] **WebSocket Support**: Real-time processing updates
- [ ] **Cloud Storage Integration**: Support for AWS S3, Google Cloud Storage
- [ ] **Advanced Models**: Integration with newer AI models (SAM, DALL-E, etc.)
- [ ] **User Authentication**: API key management and rate limiting
- [ ] **Caching System**: Result caching for improved performance
- [ ] **Video Processing**: Extend capabilities to video files
- [ ] **Custom Model Training**: Allow users to fine-tune models
- [ ] **GraphQL API**: Alternative API interface
- [ ] **Monitoring & Analytics**: Usage tracking and performance metrics

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes following the project architecture
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

### Code Standards

- Follow PEP 8 for Python code style
- Add comprehensive error handling
- Include logging for debugging
- Document all new API endpoints
- Add type hints where applicable

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Arogya Vamshi**  
- Email: [arogyavamshi2002@gmail.com](mailto:arogyavamshi2002@gmail.com)  
- GitHub: [@Arogya-2002](https://github.com/Arogya-2002)  

---

## 🙏 Acknowledgments

- [rembg](https://github.com/danielgatis/rembg) - For the excellent background removal library
- [FastAPI](https://fastapi.tiangolo.com/) - For the modern, fast web framework
- [U2NET](https://github.com/xuebinqin/U-2-Net) - For the underlying segmentation model
- Various AI research communities for the face swapping and style transfer models

---

---

⭐ **Star this repository if you found it helpful!**