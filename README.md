# SmartLock Fuzzy

Local OpenCV smart-lock prototype with a single backend process and a Next.js web UI.

The backend opens the camera directly:

```python
CAMERA_INDEX = "/dev/video1"
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
```

It streams annotated frames to the browser, lets users register their face from the live camera feed, trains an LBPH recognizer, and uses fuzzy logic to decide whether to unlock, request OTP, deny, or lock out.

## Pipeline

```text
/dev/video1
  -> FastAPI camera worker
  -> Haar face detection
  -> LBPH face recognition
  -> fuzzy security decision
  -> Next.js web UI polls /api/v1/camera/frame
```

The active runtime is only the local camera smart-lock flow.

## Structure

```text
backend/
  main.py                         FastAPI app and registration endpoints
  services/
    local_camera.py               owns cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
    face_detection.py             Haar detection and LBPH recognition
    registration.py               live sample collection and model training
    fuzzy_logic.py                fuzzy smart-lock decision wrapper
  routers/
    get_camera.py                 latest frame/history API

frontend/
  src/app/page.tsx                smart-lock dashboard
  src/components/LiveVideoStream.tsx
```

## Setup

```bash
cd /home/tamminh/SmartLock_Fuzzy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
cd frontend
npm install
cp .env.example .env.local
```

## Run

Terminal 1:

```bash
cd /home/tamminh/SmartLock_Fuzzy
source .venv/bin/activate
cd backend
python main.py
```

Terminal 2:

```bash
cd /home/tamminh/SmartLock_Fuzzy/frontend
npm run dev
```

Open `http://localhost:3000`.

## Registration

1. Enter a name in the web UI.
2. Click `Register`.
3. Look at the camera and slowly move your head until enough samples are collected.
4. The backend trains `custom_models/smartlock_lbph_model.xml` and reloads it automatically.

Registered face samples are stored in `registered_faces/`.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/v1/camera/frame` | Latest annotated frame, detections, fuzzy decision |
| `GET` | `/api/v1/camera/latest` | Latest metadata without frame payload |
| `GET` | `/api/v1/camera/history` | Recent recognition/decision history |
| `GET` | `/camera/status` | Camera worker and registration status |
| `POST` | `/api/v1/register/start` | Start live face registration |
| `POST` | `/api/v1/register/cancel` | Cancel registration |
| `GET` | `/api/v1/register/status` | Registration status |

## Environment

| Variable | Default | Description |
| --- | --- | --- |
| `CAMERA_INDEX` | `/dev/video1` | Camera device opened by OpenCV |
| `CAMERA_AUTO_START` | `true` | Set `false` to run API without opening the camera |
| `CAMERA_PROCESS_FPS` | `10` | Camera worker processing rate |
| `CAMERA_FLIP` | `false` | Horizontally flip frames |
| `REGISTERED_FACES_DIR` | `registered_faces` | Training sample directory |
| `RECOGNIZER_PATH` | `custom_models/smartlock_lbph_model.xml` | LBPH model path |

## Camera Smoke Test

```bash
cd /home/tamminh/SmartLock_Fuzzy/backend/infra
python test_cam.py
```
