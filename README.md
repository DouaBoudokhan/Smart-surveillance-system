# 🔒 Smart Surveillance System

A comprehensive AI-powered surveillance system that integrates multiple advanced detection and recognition technologies into a unified web application. Built with Flask and modern computer vision frameworks, this system provides real-time monitoring and analysis capabilities for enhanced security.

## 🌟 Features

### 1. **Object Detection & Classification**
- **Real-time Detection**: Live camera feed with object detection and tracking
- **Video Analysis**: Upload and process video files with object detection
- **Image Analysis**: Detect suspicious objects in uploaded images
- **Suspicious Object Detection**: Specialized model for detecting threats like weapons, masked faces, and dangerous items
- **Normal Object Detection**: YOLOv8-based detection for everyday objects
- **Alert System**: Automatic alert generation when suspicious objects are detected

### 2. **Facial Recognition**
- Face detection and recognition system
- Real-time facial identification
- Person tracking and identification

### 3. **Gait Analysis**
- Person identification through walking patterns
- Video-based gait recognition
- Confidence-based person matching

### 4. **License Plate Recognition**
- Tunisian license plate detection and recognition
- Video-based vehicle tracking
- OCR-powered text extraction from license plates

### 5. **OCR (Optical Character Recognition)**
- Arabic and French text recognition
- ID card information extraction
- Support for multiple document types including:
  - Full names (Arabic)
  - ID numbers
  - Date of birth
  - Place of birth
  - Addresses
  - Mother's name
  - Issue dates
  - Profession

### 6. **Audio Classification**
- Sound event detection
- Audio file analysis
- Confidence-based classification

## 🛠️ Technologies Used

### Core Frameworks
- **Flask 3.0.0** - Web application framework
- **OpenCV** - Computer vision and image processing
- **PyTorch** - Deep learning framework
- **TensorFlow** - Machine learning framework

### AI/ML Libraries
- **YOLOv8 (Ultralytics)** - Object detection
- **EasyOCR** - Text recognition (Arabic & French)
- **face_recognition** - Facial recognition
- **scikit-learn** - Machine learning utilities
- **FastAI** - Deep learning library
- **Librosa** - Audio analysis

### Additional Libraries
- **NumPy** - Numerical computing
- **Pillow** - Image processing
- **Matplotlib** - Data visualization
- **Flask-CORS** - Cross-origin resource sharing

## 📁 Project Structure

```
Smart-surveillance-system/
├── app.py                          # Main Flask application
├── face_detection.py               # Facial recognition module
├── gait.py                        # Gait analysis module
├── lp.py                          # License plate processing
├── predict.py                     # Audio prediction module
├── requirements.txt               # Python dependencies
├── static/                        # Static assets
│   ├── uploads/                   # Uploaded files
│   ├── detections/                # Detection results
│   ├── alerts/                    # Alert history
│   ├── best.pt                    # Suspicious object model
│   └── yolov8n.pt                 # Normal object detection model
├── templates/                     # HTML templates
│   ├── index.html
│   ├── detection_type.html
│   ├── detectobj.html
│   ├── upload_video.html
│   ├── real_time.html
│   ├── face.html
│   ├── gait.html
│   ├── ocr.html
│   ├── sound.html
│   └── licenseplate.html
├── gait_recognition/              # Gait recognition models
├── ranim.pt                       # ID card detection model
├── lp_recognition.pt              # License plate recognition model
├── tunisian_lp_detector.pt        # Tunisian LP detector
├── audio_classifier.pkl           # Audio classification model
└── uploads/                       # User uploads directory
```

## 🚀 Installation

### Prerequisites
- Python 3.11+
- Webcam (for real-time detection)
- GPU support recommended for optimal performance

### Step 1: Clone the Repository
```bash
git clone https://github.com/DouaBoudokhan/Smart-surveillance-system.git
cd Smart-surveillance-system
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install dlib (Windows)
```bash
pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
```

### Step 4: Create Required Directories
```bash
mkdir -p static/uploads static/detections static/alerts uploads
```

### Step 5: Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 📖 Usage

### Object Detection

#### Image Detection
1. Navigate to **Detection Type** page
2. Select **Upload Image**
3. Choose your model (Normal/Suspicious Objects)
4. Upload an image
5. View detection results and alerts

#### Video Detection
1. Navigate to **Upload Video** page
2. Select a video file
3. System processes video and highlights detected objects
4. Download annotated video

#### Real-time Detection
1. Navigate to **Real-time Detection**
2. Grant camera permissions
3. View live detection feed
4. Detected objects appear in real-time with bounding boxes
5. Alerts trigger automatically for suspicious objects

### Facial Recognition
1. Navigate to **Face Recognition** page
2. Click **Start Detection**
3. System launches facial recognition module
4. Detects and identifies faces in real-time

### Gait Analysis
1. Navigate to **Gait Analysis** page
2. Upload a video containing walking person
3. System analyzes gait patterns
4. Returns person ID and confidence score

### License Plate Recognition
1. Navigate to **License Plate** page
2. Upload a video with vehicles
3. System detects and reads license plates
4. Returns annotated video with recognized plates

### OCR (ID Card Recognition)
1. Navigate to **OCR** page
2. Upload an ID card image
3. System extracts all text fields
4. Displays structured information (name, ID, DOB, etc.)

### Audio Classification
1. Navigate to **Sound Detection** page
2. Upload an audio file
3. System classifies the sound event
4. Returns prediction and confidence score

## 🎯 Features in Detail

### Alert System
- Suspicious objects are tracked and logged
- Alert history stored in `static/alerts/alert_history.txt`
- Timestamps and object types recorded
- Visual alerts displayed in web interface

### Supported Suspicious Objects
- Pistol
- Grenade
- Knife
- RPG
- Machine Guns
- Masked Face
- Bat

### OCR Field Detection
The system can extract the following fields from ID cards:
- **first_name**: First name in Arabic
- **last_name**: Last name in Arabic
- **full_name**: Complete name
- **mother_name**: Mother's name in Arabic
- **id**: 8-digit ID number
- **dob**: Date of birth
- **pob**: Place of birth
- **address**: Full address with numbers
- **profession**: Profession in Arabic
- **issue_date**: Document issue date

## 🔧 Configuration

### Model Switching
The system supports switching between two detection modes:
- **Normal Mode**: YOLOv8n for general object detection
- **Suspicious Mode**: Custom-trained model for threat detection

### Custom Paths
Update paths in `app.py` if needed:
```python
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DETECTION_FOLDER'] = 'static/detections'
```

## 🎥 Video Processing

### Supported Formats
- MP4 (recommended)
- AVI
- MOV

### Output Format
- Codec: H.264 (avc1)
- Container: MP4
- Frame rate: Preserved from input

## 🤖 Models

### Pre-trained Models Included
- `best.pt` - Suspicious object detection model
- `yolov8n.pt` - YOLOv8 Nano model
- `ranim.pt` - ID card field detection
- `lp_recognition.pt` - License plate character recognition
- `tunisian_lp_detector.pt` - Tunisian license plate detector
- `audio_classifier.pkl` - Audio event classifier

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/detection_type` | GET | Detection mode selection |
| `/detectobj` | POST | Image object detection |
| `/upload_video_actual` | POST | Video processing |
| `/start_camera` | GET | Real-time detection page |
| `/video_feed` | GET | Live camera stream |
| `/get_detected_classes` | GET | Get detected objects (JSON) |
| `/alert_status` | GET | Get alert status (JSON) |
| `/analyze` | POST | Gait analysis |
| `/ocr` | POST | OCR processing |
| `/sound` | POST | Audio classification |
| `/licenseplate.html` | POST | License plate recognition |
| `/start-detection` | POST | Start facial recognition |
| `/switch_model/<type>` | GET | Switch detection model |

## 👥 Team

This project was developed by a team of specialists:
- **Ala** - Facial recognition & Flask backend
- **Doua** - Gait analysis & core architecture
- **Ranim** - OCR & ID card recognition
- **Hazem** - Face recognition integration
- **Aziz** - Audio classification
- **Nour** - License Plate Recognition

## 📄 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- EasyOCR for Arabic OCR capabilities
- OpenCV community
- Flask framework developers

---

**Built with ❤️ for enhanced security and surveillance**
