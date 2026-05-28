# SmartLock Fuzzy

SmartLock Fuzzy is an intelligent, local, face-recognition-based smart lock system. It integrates high-speed local computer vision, a Mamdani fuzzy logic decision engine, and Raspberry Pi 4 B hardware integration (OLED display, 4x4 matrix keypad, and RC Servo) with a modern Next.js web dashboard.

---

## 🚀 Key Features

* **Real-time Computer Vision Pipeline**:
  * Local face detection via OpenCV **Haar Cascade**.
  * Face recognition using an **LBPH (Local Binary Patterns Histograms)** model.
  * Real-time estimation of **Face Illumination** (mean ROI brightness) and **Facial Angle** (asymmetric profile estimation).
* **Mamdani Fuzzy Logic Decision Engine**:
  * Powered by `pyfuzzylite` to dynamically evaluate three inputs: *Model Confidence*, *Illumination*, and *Facial Angle*.
  * Uses **Gaussian membership functions** with a custom-scaled Model Confidence range of `[0.0, 85.0]`.
  * Fires **13 fuzzy rules** to output a defuzzified *Security Risk* score (using Centroid defuzzification).
  * Includes a built-in static logic fallback to guarantee fail-safe operation when `pyfuzzylite` is unavailable.
* **Hardware & Peripheral Control**:
  * **Interrupt-driven 4x4 Matrix Keypad scanning** (utilizing GPIO interrupts with software debouncing instead of polling to save CPU).
  * **SSD1306 OLED display** showing system status, registration progress, PIN prompts, door opening timer, and lockout count downs.
  * **RC Servo motor control** using software PWM with an anti-vibration feature (automatically disabling the duty cycle pulses after rotation to eliminate motor buzz and save power).
* **PIN & Password Management**:
  * Local 6-digit passcode authentication, stored and compared using secure **SHA-256 hashes**.
  * Dynamic keypad passcode change interface embedded in the dashboard, communicating with the FastAPI backend endpoint.
* **Diagnostics & Benchmarking**:
  * **Cross-platform Pipeline Benchmark Utility**: Profiles step-by-step pipeline latency (ms), RAM usage delta (MB), CPU load (%), and estimates RPi 4 B power (Watts & energy/frame in mJ). Includes hardware mocks to run on development PCs (Windows/Linux/macOS).
  * **Servo Diagnostic Tool**: A menu-driven script (`test_servo.py`) to calibrate and test duty cycles, angles, and electrical grounds.

---

## 📂 Project Structure

```text
SmartLock_Fuzzy/
├── backend/
│   ├── infra/
│   │   ├── fuzzy_controller.py      # Core Mamdani fuzzy engine (Gaussian, 13 rules)
│   │   └── test_cam.py              # Camera smoke-test script
│   ├── routers/
│   │   └── get_camera.py            # API endpoints for frame streaming & metadata
│   ├── services/
│   │   ├── face_detection.py        # Haar Cascade & LBPH recognition wrapper
│   │   ├── fuzzy_logic.py           # Fuzzy engine wrapper & fallback static logic
│   │   ├── hardware_io.py           # Keypad matrix interrupts & Servo PWM controller
│   │   ├── hardware_schematic.md    # Detailed wiring diagrams & electrical notes
│   │   ├── local_camera.py          # Background worker coordinating camera, oled, and hardware
│   │   ├── oled_display.py          # Luma.oled display UI drawer (enter PIN, lockout, etc.)
│   │   └── registration.py          # Face sample collector & LBPH training manager
│   ├── main.py                      # FastAPI application main entry point
│   └── benchmark_pipeline.py        # Pipeline latency, memory, and power profiling
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   └── page.tsx             # Main dashboard UI integrating video & forms
│   │   └── components/
│   │       ├── LiveVideoStream.tsx  # Video frame poller & decoder
│   │       └── PasswordChange.tsx   # Secured keypad password changer component
│   └── .env.local                   # Frontend API URL configuration
├── custom_models/
│   └── smartlock_lbph_model.xml     # Trained LBPH model file
├── registered_faces/                # Raw directories of captured face samples
├── test_servo.py                    # Independent Servo test & calibration script
└── requirements.txt                 # Python dependencies
```

---

## 🔌 Hardware Connections & Pins

The system is configured for the **Raspberry Pi 4 Model B**. 

### Quick Wiring Matrix

| Component | Pin Function | Raspberry Pi BCM GPIO | Physical Pin Number |
| --- | --- | --- | --- |
| **Servo Motor** | Signal (SIG) | `GPIO 18` | Pin 12 |
| **SSD1306 OLED** | SDA | `GPIO 2` (SDA) | Pin 3 |
| **SSD1306 OLED** | SCL | `GPIO 3` (SCL) | Pin 5 |
| **4x4 Keypad** | Row 1, 2, 3, 4 | `GPIO 17`, `27`, `22`, `5` | Pin 11, 13, 15, 29 |
| **4x4 Keypad** | Col 1, 2, 3, 4 | `GPIO 6`, `13`, `19`, `26` | Pin 31, 33, 35, 37 |

> [!IMPORTANT]
> **Common Ground Requirement**: Always power the RC Servo motor using a dedicated external 5V power supply. Connecting the servo's power line directly to the Pi's 5V pin can trigger brownout resets. Ensure a common ground wire connects the external supply's negative (-) terminal to any GND pin on the Raspberry Pi.
>
> *For an ASCII schematic and implementation notes, read the [Hardware Schematic Guide](file:///d:/SmartLock_Fuzzy/backend/services/hardware_schematic.md).*

---

## 🧠 Fuzzy Logic Decision Engine

The security decision pipeline is modeled using **Mamdani fuzzy inference** and defuzzified into a crisp value in range `[0.0, 1.0]`.

```text
  INPUTS (Antecedents)                      ENGINE                      OUTPUT & MAPPING
┌────────────────────────┐
│ Confidence:  [0 - 85]  ├──────┐
├────────────────────────┤      │     ┌──────────────────┐      Security Risk [0.0 - 1.0]
│ Illumination: [0 - 255]├──────┼────>│  Mamdani Engine  ├────> ┌────────────────────────┐
├────────────────────────┤      │     │ (13 Fuzzy Rules) │      │ < 0.30: Unlock         │
│ Facial Angle: [0 - 90] ├──────┘     └──────────────────┘      │ < 0.60: OTP Challenge  │
└────────────────────────┘                                      │ < 0.85: Access Deny    │
                                                                │ >=0.85: Lockout & Alarm│
                                                                └────────────────────────┘
```

### Antecedents (Membership Functions)
* **Model Confidence ($C$)** $[0, 85]$:
  * `LOW`: Gaussian $[0.0, 17.0]$
  * `MEDIUM`: Gaussian $[42.5, 10.0]$
  * `HIGH`: Gaussian $[85.0, 17.0]$
* **Illumination ($I$)** $[0, 255]$:
  * `DARK`: Gaussian $[0.0, 40.0]$
  * `NORMAL`: Gaussian $[128.0, 45.0]$
  * `BRIGHT`: Gaussian $[255.0, 40.0]$
* **Facial Angle ($\theta$)** $[0, 90]$:
  * `FRONTAL`: Gaussian $[0.0, 20.0]$
  * `MARGINAL`: Gaussian $[90.0, 40.0]$

### Consequent (Membership Function)
* **Security Risk** $[0.0, 1.0]$: (Defaults to $1.0$ - maximum risk - in case of error).
  * `MINIMUM`: Gaussian $[0.0, 0.1]$
  * `AVERAGE`: Gaussian $[0.5, 0.1]$
  * `MAXIMUM`: Gaussian $[1.0, 0.08]$

---

## 🛠️ Setup & Execution

### 1. Backend Setup

Prerequisites: OpenCV, python3, pip, virtualenv.

```bash
# Set up a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # Or `.venv\Scripts\activate` on Windows

# Install python dependencies
pip install -r requirements.txt

# Run the FastAPI server (binds on port 8000)
python backend/main.py
```

### 2. Frontend Setup

Prerequisites: Node.js (v18+ recommended) and npm.

```bash
# Navigate to the frontend directory
cd frontend

# Install Node dependencies
npm install

# Set up local environment variables
cp .env.example .env.local
```

Ensure that `NEXT_PUBLIC_API_URL` in `frontend/.env.local` points to the running backend (default: `http://localhost:8000`). If accessing the UI from another laptop on the network, update `localhost` to your Raspberry Pi's local IP.

```bash
# Run the development server
npm run dev
```

Open `http://localhost:3000` to access the dashboard.

---

## 📡 API Endpoints

FastAPI exposes the following endpoints (default base URL: `http://localhost:8000`). Fully documented Swagger UI is available at `/docs`.

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/v1/camera/frame` | Returns the latest camera metadata along with the base64-encoded JPEG frame (`frame_base64`) used by the frontend to render the live feed. |
| `GET` | `/api/v1/camera/latest` | Returns the latest camera metadata *excluding* the base64 frame payload (for lightweight polling). |
| `GET` | `/api/v1/camera/history` | Returns the recent history of face detection, recognition, and fuzzy decision records (up to last 100 entries). |
| `GET` | `/camera/status` | Returns the status of the camera background worker, including database identity statistics, device index, FPS, and keypad/hardware status. |
| `POST` | `/api/v1/register/start` | Initiates the interactive face sample capture process. Request Body: `{ "name": "string", "samples_required": 30 }` |
| `POST` | `/api/v1/register/cancel` | Cancels the active face registration session. |
| `GET` | `/api/v1/register/status` | Returns the current face registration progress and status. |
| `POST` | `/api/v1/keypad/password` | Updates the 6-digit keypad passcode. Request Body: `{ "current_password": "xxxxxx", "new_password": "yyyyyy" }` |
| `GET` | `/api/v1/keypad/status` | Returns keypad runtime metrics (keypad activity, lockout timers, failed attempts, and the last keypad event). |

---

## 🧪 Diagnostics & Verification

### Pipeline Benchmarking
The dashboard includes a **Benchmark** panel that can start a benchmark run, poll progress, and visualize latency, FPS, CPU, RAM, and edge telemetry from the latest `summary.json`.

To profile latency, FPS, RAM, CPU utilization, and Raspberry Pi telemetry on the edge device, run:
```bash
python3 backend/benchmark_pipeline.py \
  --source camera \
  --camera /dev/video0 \
  --duration 60 \
  --warmup 20 \
  --width 640 \
  --height 480 \
  --capture-fps 10 \
  --output-dir benchmark_results/pi_camera
```

For a quick smoke test without a camera:
```bash
python3 backend/benchmark_pipeline.py --source synthetic --frames 100 --output-dir benchmark_results/synthetic
```

The benchmark groups frames into no-face and face-detected cases automatically. To populate both report columns in one run, leave the camera empty for part of the run and stand in front of it for the remaining part. It writes `summary.json`, `summary.md`, `summary_table.tex`, and per-frame `frames.csv` into the output directory.


### Face Registration
1. In the Web UI, enter a username and click `Register`.
2. The UI enters sample collection mode. Look directly at the camera and slowly shift your head.
3. The system captures 30 valid frames (ensuring they pass quality, blur, and lighting checks) and automatically saves them.
4. The system triggers training of `custom_models/smartlock_lbph_model.xml` and reloads it instantly without interrupting the video feed.
