https://youtu.be/uOdji7DuDKc

# 🔬 Skin Issue Detection

A FastAPI-based web application that uses YOLOv8 computer vision and OpenAI's language models to detect skin issues from images and provide personalized skincare recommendations.

## 📋 Overview

This project combines deep learning object detection with AI-powered skincare guidance. Upload an image of your skin, and the application will:

1. **Detect skin issues** using a fine-tuned YOLOv8 model
2. **Analyze** the detected issues with confidence scores
3. **Generate personalized skincare routines** using OpenAI's language models
4. **Provide visual feedback** with annotated images showing detected regions

**⚠️ Disclaimer:** This tool is for educational purposes only and is NOT a medical diagnostic tool. Always consult a dermatologist or licensed healthcare professional for diagnosis and treatment of skin conditions.

## ✨ Features

- **Real-time skin issue detection** using YOLOv8 object detection
- **AI-powered skincare recommendations** using OpenAI language models
- **Web-based interface** with interactive image upload via webcam or file
- **Annotated image output** showing detected skin issues with bounding boxes
- **CORS-enabled API** for easy frontend integration
- **Production-ready FastAPI backend** with comprehensive error handling

## 📦 Requirements

- Python 3.8+
- TensorFlow (with CUDA support for GPU acceleration)
- PyTorch (required by ultralytics/YOLO)
- OpenAI API key

See `req.txt` for the complete dependency list.

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/WTL04/CECS458---Skin-Issue-Detection.git
cd CECS458---Skin-Issue-Detection
pip install -r req.txt
```

### 2. Environment Setup

Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY="sk-your-api-key-here"
YOLO_MODEL_PATH="runs/skin_yolo_run_relabeled_dataset/weights/best.pt"
OPENAI_MODEL="gpt-5-nano"
```

**Note:** You'll need to train or provide your custom YOLO model weights at the specified path.

### 3. Running the Backend

Start the FastAPI server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 4. Using the Web Interface

Open `index.html` in your browser or serve it using a local web server:

```bash
# Using Python's built-in server
python -m http.server 5500
# Then navigate to http://localhost:5500
```

## 📖 API Endpoints

### POST `/analyze`

Upload an image for skin issue detection and analysis.

**Request:**
- `file` (multipart/form-data): Image file (PNG, JPG, etc.)

**Response:**
```json
{
  "detections": [
    {
      "label": "acne",
      "confidence": 0.92
    }
  ],
  "gpt_routine": "## Acne\n\n1) Acne appears as...",
  "annotated_image": "iVBORw0KGgoAAAANSUhEUg..."
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid image file
- `500`: Server error (YOLO model not loaded, API error, etc.)

## 📁 Project Structure

```
CECS458---Skin-Issue-Detection/
├── main.py                          # FastAPI backend application
├── yolo.py                          # YOLO model training script
├── index.html                       # Web interface frontend
├── req.txt                          # Python dependencies
├── model.ipynb                      # Jupyter notebook for model exploration
├── yolo.ipynb                       # Jupyter notebook for YOLO training
├── yolov8n.pt                       # YOLOv8 nano pretrained weights
├── yolov8s.pt                       # YOLOv8 small pretrained weights
├── yolo11n.pt                       # YOLOv11 nano pretrained weights
├── runs/                            # Training outputs and logs
└── .gitignore                       # Git ignore file
```

## 🛠️ How It Works

### Backend Flow

1. **Image Upload** → FastAPI receives the image file
2. **YOLO Detection** → Image is processed by YOLOv8 model
3. **Detection Extraction** → Bounding boxes and class labels are extracted
4. **Image Annotation** → YOLO creates a visual overlay with detections
5. **LLM Processing** → Detection results are sent to OpenAI for skincare routine generation
6. **Response Assembly** → Detections, routine, and annotated image are returned as JSON

### Skincare Routine Generation

The LLM is prompted to provide:
- Short explanations of detected concerns
- Key ingredients that help address each issue
- Example over-the-counter products
- Simple morning and evening routines
- Common mistakes to avoid

## 🧠 Model Training

To train a custom YOLO model:

1. Prepare your dataset in YOLO format
2. Update the dataset path in `yolo.py`
3. Run the training script:

```bash
python yolo.py
```

Trained weights will be saved in the `runs/` directory.

## 🔧 Configuration

Edit `main.py` to customize:

- **Model Path:** `MODEL_PATH` environment variable
- **OpenAI Model:** `OPENAI_MODEL` environment variable
- **API Behavior:** Adjust CORS settings, model parameters, or prompt engineering

## 📝 Dependencies Overview

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 object detection framework |
| `fastapi[standard]` | Web framework for API |
| `uvicorn[standard]` | ASGI server |
| `pillow` | Image processing |
| `openai>=1.0.0` | OpenAI API client |
| `tensorflow[and-cuda]` | Deep learning framework |
| `keras` | Neural network API |
| `numpy`, `pandas` | Data processing |
| `python-dotenv` | Environment variable management |

## ⚙️ Troubleshooting

**YOLO model not loading:**
- Ensure the model path is correct and file exists
- Verify ultralytics is installed: `pip install ultralytics --upgrade`

**OpenAI API errors:**
- Check your API key is valid and has sufficient credits
- Verify internet connection

**CORS errors:**
- The app is configured to allow all origins by default (`allow_origins=["*"]`)
- For production, restrict to specific frontend URLs

**GPU not being used:**
- Install CUDA: `pip install tensorflow[and-cuda]`
- Verify GPU availability: `nvidia-smi`

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

## 📄 License

This project is part of CECS 458 at California State University, Long Beach.

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome! Please feel free to open an issue or submit a pull request.

## ⚖️ Disclaimer

This application is intended for **educational purposes only**. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare professional for:

- Medical diagnosis of skin conditions
- Treatment recommendations
- Prescription medications
- Concerns about your health

The developers of this application assume no responsibility for any health decisions made based on this tool.

---

**Created for:** CECS 458 - Machine Learning Course  
**Last Updated:** 2026-04-29 05:18:32