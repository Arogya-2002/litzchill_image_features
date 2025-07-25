# Background Removal, Replacement, and Inpainting Service

A powerful image processing service built with Python and FastAPI that provides intelligent background removal, background replacement, and image inpainting capabilities. Upload images through a clean web interface or REST API to automatically remove backgrounds, add custom ones, or fill masked regions.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)  
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)  
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## ✨ Features

- **🎯 Smart Background Removal**: Leverage the powerful U2NET model via `rembg` library for precise background removal  
- **🖼️ Background Replacement**: Seamlessly add custom backgrounds to transparent images  
- **🚀 REST API**: FastAPI-powered endpoints for programmatic access  
- **🌐 Web Interface**: Clean, responsive HTML interface for easy image uploads  
- **📁 Modular Architecture**: Well-organized codebase with reusable components  
- **🛡️ Robust Error Handling**: Comprehensive logging and exception management  
- **🗂️ Smart File Management**: Automatic temporary file cleanup and organized storage  

---

## 📁 Project Structure

```
.
├── app.py                          # FastAPI application and route definitions
├── static/                         # Static assets
│   └── style.css                   # Styling for web interface
├── templates/                      # Jinja2 HTML templates
│   └── index.html                  # Main web interface
├── src/
│   ├── components/
│   │   ├── add_bg.py              # Background addition logic
│   │   ├── remove_bg.py           # Background removal logic
│   ├── entity/
│   │   ├── artifact.py         # Data classes and artifacts
│   │   └── config.py           # Configuration management
│   ├── exceptions.py              # Custom exception definitions
│   ├── logger.py                  # Logging configuration
│   ├── pipeline/
│   │   └── bg_prediction_pipeline.py  # Main processing pipeline
│   └── utils.py                   # Utility functions and helpers
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
   git clone https://github.com/Arogya-2002/BG_add_-_remove.git
   cd BG_add_-_remove
   ```

2. **Set up virtual environment** (recommended)
   ```bash
   # Using conda
   conda create -p venv python==3.10 -y
   conda activate venv/
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note**: The `rembg` and inpainting libraries may require additional system dependencies:
   > - **Ubuntu/Debian**: `sudo apt-get install libglib2.0-0`
   > - **macOS**: Usually works out of the box
   > - **Windows**: May require Visual C++ Build Tools

### Running the Application

1. **Start the FastAPI server**
   ```bash
   uvicorn app:app --reload
   ```

2. **Access the web interface**
   Open your browser and navigate to: `http://127.0.0.1:8000`

---

## 📖 Usage

### Web Interface

1. Navigate to `http://127.0.0.1:8000` in your browser  
2. Choose one of the following actions:  
   - **Remove Background**: Upload an image and click "Remove Background"  
   - **Add Background**: Upload a foreground image and a background image  
3. Download your processed images  

### API Endpoints

#### `GET /`
Serves the main HTML interface.

#### `POST /remove-background/`
Remove background from an uploaded image.

**Parameters:**
- `file`: Image file (multipart/form-data)

**Response:** Image with background removed (PNG format)

#### `POST /add-background/`
Add a new background to a foreground image.

**Parameters:**
- `foreground`: Foreground image file (preferably with transparent background)
- `background`: Background image file

**Response:** Composite image with new background
---

## ⚙️ Configuration

Configuration is managed through classes in `src/entity/bg_config.py`. You can customize:

- **File paths**: Where processed images are saved  
- **Temporary directories**: Location for intermediate processing files  
- **Model settings**: Background removal and inpainting model parameters  
- **Logging levels**: Adjust verbosity of application logs  

---

## 🔧 Technical Details

- **Background Removal Model**: U2NET via `rembg`  
- **Image Processing**: Pillow (PIL) for image manipulation  
- **Web Framework**: FastAPI with Jinja2 templating  
- **File Handling**: Automatic cleanup of temporary files  
- **Error Management**: Custom exceptions with detailed stack traces  
- **Logging**: Structured logging with configurable levels  

---

## 🛠️ Development

### Project Architecture

The application follows a modular architecture:

- **Components**: Core functionality for background and inpainting operations  
- **Entities**: Data classes and configuration management  
- **Pipeline**: Main processing workflows  
- **Utils**: Helper functions and utilities  
- **Exceptions**: Custom error handling  

### Adding New Features

1. Implement core logic in appropriate `src/components/` module  
2. Add configuration options in `src/entity/bg_config.py`  
3. Create pipeline functions in `src/pipeline/`  
4. Add API endpoints in `app.py`  

---

## 🚀 Future Roadmap

- [ ] **User Authentication**: Add user accounts and session management  
- [ ] **Batch Processing**: Support multiple image processing  
- [ ] **Advanced Models**: Integration with SAM, Deeplab, etc.  
- [ ] **Cloud Storage**: Support for AWS S3, Google Cloud Storage  
- [ ] **Image Formats**: Extended format support (WebP, HEIC, etc.)  
- [ ] **Real-time Processing**: WebSocket support for live processing  
- [ ] **Docker Support**: Containerization for easy deployment  
- [ ] **Enhanced UI**: Modern React/Vue.js frontend  

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request  

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

---

⭐ **Star this repository if you found it helpful!**