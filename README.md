# CE224.Q11 - People Counting System

A real-time people counting application powered by YOLOv11 and WiFi CSI (Channel State Information), featuring ESP32-CAM integration, edge computing with WebSocket streaming, FastAPI backend, and Next.js frontend with live bounding box visualization and CSI motion detection.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-purple)
![ESP32](https://img.shields.io/badge/ESP32-CAM-orange?logo=espressif)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Run-blue?logo=google-cloud)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [CSI Motion Detection](#-csi-motion-detection)
- [API Documentation](#-api-documentation)
- [Cloud Deployment](#-cloud-deployment)
- [Dataset](#-dataset)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Camera-based Detection

- **Real-time People Detection**: YOLOv11 nano model optimized for edge devices
- **NCNN Backend**: CPU-optimized inference using NCNN library
- **Live Bounding Boxes**: Real-time annotated video with detection overlays
- **Edge Computing**: On-device inference for low-latency processing

### WiFi CSI Sensing

- **Motion Detection**: Non-visual human detection using WiFi signal variance
- **Real-time Monitoring**: Live RSSI and amplitude visualization with charts
- **Variance-based Detection**: Statistical motion detection without ML training
- **Calibration API**: Adjustable thresholds for different environments

### Streaming & Communication

- **Direct WebSocket Streaming**: Frontend receives frames directly from edge device
- **Fallback HTTP Polling**: Automatic fallback when WebSocket unavailable
- **ESP32-CAM Integration**: Wireless camera + CSI streaming via WebSocket

### Web Interface

- **Live Video Stream**: Real-time annotated video with bounding boxes
- **CSI Motion Chart**: Interactive charts showing RSSI, amplitude, and motion level
- **Motion Status Banner**: Visual indicators for motion detection status
- **Responsive Design**: Modern UI built with Next.js and Tailwind CSS

## 🏗 Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │     │                 │
│   ESP32-CAM     │────▶│  Edge Device    │═══════════════════════════▶│  Next.js GUI    │
│  Camera + CSI   │     │  (YOLO Infer)   │     │                 │     │  (Port 3000)    │
│   WebSocket     │     │  ws_server.py   │     │                 │     │                 │
│                 │     │   Port 8080     │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └─────────────────┘     └───────┬─────────┘
                                 │                                              │
                                 │ REST API                                     │ Poll
                                 ▼                                              ▼
                        ┌─────────────────┐                            ┌─────────────────┐
                        │  FastAPI Server │◀──────────────────────────▶│  CSI Motion     │
                        │  (Port 8000)    │         REST API           │  Detection API  │
                        │  Google Cloud   │                            │                 │
                        └─────────────────┘                            └─────────────────┘
```

### Data Flow

1. **ESP32-CAM** captures JPEG frames (~10 FPS) + CSI data (every 500ms), sends via WebSocket
2. **Edge Device** (`ws_server.py`) receives frames, runs YOLOv11 inference
3. **Direct Streaming**: Edge broadcasts results directly to frontend via WebSocket (`/viewer` endpoint)
4. **Backend Sync**: Edge sends results to backend for storage and CSI motion analysis
5. **Frontend** displays live video, bounding boxes, people count, and CSI motion status

## 📁 Project Structure

```
CE224.Q11_People_Counting/
├── edge_side/
│   ├── camera/                    # ESP32-CAM Firmware (ESP-IDF 5.x)
│   │   ├── main/
│   │   │   ├── main.c            # Camera + CSI capture & WebSocket
│   │   │   └── camera_pins.h     # Hardware pin definitions
│   │   ├── sdkconfig             # CSI enabled configuration
│   │   └── CMakeLists.txt
│   │
│   └── infra/                     # Edge ML Infrastructure
│       ├── ws_server.py          # WebSocket server + YOLO inference
│       ├── test_cam.py           # Camera client for testing
│       ├── weights/              # Model weights
│       │   └── yolo11n_ncnn_model_coco/
│       └── csi_data/             # CSI training data
│
├── server_side/
│   ├── backend/                   # FastAPI Backend
│   │   ├── main.py               # Application entry
│   │   ├── routers/
│   │   │   ├── count_people.py   # Camera counting endpoints
│   │   │   └── csi.py            # CSI motion detection endpoints
│   │   ├── Dockerfile            # Container config
│   │   └── cloudbuild.yaml       # Cloud Build config
│   │
│   └── frontend/                  # Next.js Frontend
│       ├── src/
│       │   ├── app/page.tsx      # Main page
│       │   ├── components/
│       │   │   ├── LiveVideoStream.tsx   # WebSocket video stream
│       │   │   ├── CSIChart.tsx          # Motion detection charts
│       │   │   └── BoundingBoxCanvas.tsx # Detection overlays
│       │   └── types/
│       ├── Dockerfile            # Container config
│       └── cloudbuild.yaml       # Cloud Build config
│
├── training_process/              # Model Training
│   ├── optimize_model.py         # NCNN export with INT8 quantization
│   ├── train_csi_model.py        # CSI ML model training
│   └── htn-object-counting.ipynb # Training notebook
│
└── requirements.txt              # Python dependencies
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- ESP-IDF 5.x (for ESP32-CAM firmware)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/chisphung/CE224.Q11_People_Counting.git
cd CE224.Q11_People_Counting
```

### 2. Set Up Python Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 3. Set Up Frontend

```bash
cd server_side/frontend
npm install

# Create environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
echo "NEXT_PUBLIC_WS_URL=ws://YOUR_EDGE_IP:8080/viewer" >> .env.local
```

### 4. Set Up ESP32-CAM

#### 4.1 Configure Wi-Fi and Server

Edit `edge_side/camera/main/main.c`:

```c
#define WIFI_SSID "your_wifi_name"
#define WIFI_PASS "your_wifi_password"
#define SERVER_URI "ws://192.168.x.x:8080"  // Edge server IP
```

#### 4.2 Build and Flash

```bash
cd edge_side/camera

# Using Docker (Recommended)
docker run --rm -v $PWD:/project -w /project espressif/idf:v5.3 idf.py build
docker run --rm -v $PWD:/project -w /project \
  --device=/dev/ttyUSB0 --privileged \
  espressif/idf:v5.3 idf.py -p /dev/ttyUSB0 flash monitor
```

## 📖 Usage

### Option 1: Full Pipeline

```bash
# Terminal 1: Backend server
cd server_side/backend
python main.py

# Terminal 2: Edge WebSocket server
cd edge_side/infra
python ws_server.py --display  # --display for local preview

# Terminal 3: Frontend
cd server_side/frontend
npm run dev

# Power on ESP32-CAM (connects automatically)
```

Open `http://localhost:3000` for live video with people counting and CSI motion detection.

### Option 2: Test with Webcam

```bash
cd edge_side/infra

# Run edge server
python ws_server.py --port 8080

# In another terminal, run test camera
python test_cam.py  # Uses webcam as input
```

## 📡 CSI Motion Detection

WiFi CSI (Channel State Information) enables non-visual motion detection by analyzing WiFi signal variations caused by human movement.

### How It Works

The motion detection algorithm uses **variance-based statistical analysis**:

1. **Sliding Window**: Analyzes last N CSI samples (default: 10)
2. **RSSI Variance**: Calculates variance of RSSI values over time
3. **Amplitude Variance**: Calculates variance of mean CSI amplitudes
4. **Threshold Comparison**: Motion detected if variance exceeds thresholds
5. **Motion Level**: Weighted combination (40% RSSI + 60% amplitude)

### Motion Level Formula

```
M = 0.4 × min(100, σ²_RSSI/τ_RSSI × 50) + 0.6 × min(100, σ²_amp/τ_amp × 50)
```

Where:

- `τ_RSSI = 5 dB` (default RSSI variance threshold)
- `τ_amp = 50` (default amplitude variance threshold)

### Calibration

```bash
# Adjust thresholds via API
curl -X POST "http://localhost:8000/api/v1/csi/calibrate?rssi_threshold=3&amplitude_threshold=30&window_size=15"
```

### Motion Status Levels

| Status          | Motion Level | Indicator |
| --------------- | ------------ | --------- |
| No Motion       | 0-10%        | ⚪ Gray   |
| Low Motion      | 10-40%       | 🟢 Green  |
| Moderate Motion | 40-70%       | 🟡 Yellow |
| High Motion     | 70-100%      | 🔴 Red    |

## 📚 API Documentation

### Camera Endpoints

| Method | Endpoint               | Description                 |
| ------ | ---------------------- | --------------------------- |
| `POST` | `/api/v1/count`        | Count from file/URL         |
| `POST` | `/api/v1/count/upload` | Count from uploaded image   |
| `POST` | `/api/v1/count/edge`   | Receive from edge device    |
| `GET`  | `/api/v1/count/latest` | Latest count result         |
| `GET`  | `/api/v1/stream/frame` | Live frame for streaming    |
| `GET`  | `/api/v1/count/fusion` | Fusion count (camera + CSI) |

### CSI Motion Detection Endpoints

| Method | Endpoint                    | Description                      |
| ------ | --------------------------- | -------------------------------- |
| `POST` | `/api/v1/csi/data`          | Receive CSI data + detect motion |
| `GET`  | `/api/v1/csi/motion`        | **Real-time motion status**      |
| `GET`  | `/api/v1/csi/stats`         | Collection statistics            |
| `GET`  | `/api/v1/csi/buffer`        | Recent CSI samples               |
| `POST` | `/api/v1/csi/calibrate`     | **Adjust detection thresholds**  |
| `GET`  | `/api/v1/csi/training-data` | Training file info               |

### WebSocket Endpoints (Edge Server)

| Endpoint                | Description                          |
| ----------------------- | ------------------------------------ |
| `ws://HOST:8080/`       | Camera client connection (ESP32-CAM) |
| `ws://HOST:8080/viewer` | **Frontend viewer connection**       |

Interactive API docs: `http://localhost:8000/docs`

## ☁️ Cloud Deployment

The system can be deployed to Google Cloud Run for public access.

### Deploy Backend

```bash
cd server_side/backend
gcloud builds submit --config=cloudbuild.yaml
```

### Deploy Frontend

```bash
cd server_side/frontend

# Update API URL in cloudbuild.yaml first
gcloud builds submit --config=cloudbuild.yaml
```

### Environment Variables

| Variable              | Description                             |
| --------------------- | --------------------------------------- |
| `NEXT_PUBLIC_API_URL` | Backend API URL                         |
| `NEXT_PUBLIC_WS_URL`  | Edge WebSocket URL for direct streaming |

## 📊 Dataset

Camera model uses YOLOv11n pre-trained on **MS COCO dataset**, filtering only `person` class (ID 0) for optimal generalization.

Custom dataset training was tested but showed domain shift issues in real deployment. Using COCO pre-trained weights provides better accuracy across diverse environments.

- **Model**: YOLOv11n (nano)
- **Backend**: NCNN with INT8 quantization
- **Input Resolution**: 320×320 (optimized for edge)
- **Confidence Threshold**: 0.25

## 🛠 Technologies

### Hardware

- **ESP32-CAM** - Wi-Fi camera with CSI support

### Edge Computing

- **YOLOv11** - Object detection (Ultralytics)
- **NCNN** - Optimized CPU inference (Tencent)
- **WebSocket** - Real-time streaming

### Backend

- **FastAPI** - Python web framework
- **Pydantic** - Data validation
- **Google Cloud Run** - Serverless deployment

### Frontend

- **Next.js 14** - React framework
- **Tailwind CSS** - Styling
- **SVG Charts** - Custom visualization

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

## 👥 Authors

- **Chi Phung** - [GitHub](https://github.com/chisphung)

---

<p align="center">
  Made with ❤️ for CE224.Q11 - Real-time People Counting with YOLOv11 & WiFi CSI
</p>
