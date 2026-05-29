# SmartLock Fuzzy System - Static API Specification

This document maps all APIs of the **SmartLock Fuzzy** system structured across four layers: **DRIVER**, **SYSTEM INTERFACE**, **SERVICE**, and **APPLICATION**.

---

## 1. DRIVER Layer (Hardware Drivers)

Direct hardware-level control interfaces for the Raspberry Pi 4 Model B GPIO pins, SPI serial interface, and software PWM timer.

### 1.1. GPIO Control
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `GPIO.setup` | `def setup(pin: int, mode: int, pull_up_down: int) -> None` | `0x1101` | Sync | Non-reentrant | `pin`: BCM pin (int), `mode`: e.g. `GPIO.IN`/`GPIO.OUT` (int), `pull_up_down`: e.g. `GPIO.PUD_DOWN` (int) | None | None | `None` | Configures the direction and electrical pull-up/pull-down settings of a GPIO pin. | `RPi.GPIO` library |
| `GPIO.output` | `def output(pin: int, state: int) -> None` | `0x1102` | Sync | Non-reentrant | `pin`: BCM pin (int), `state`: e.g. `GPIO.HIGH`/`GPIO.LOW` (int) | None | None | `None` | Drives a configured output GPIO pin level to logical HIGH (3.3V) or LOW (0V). | `RPi.GPIO` library |
| `GPIO.input` | `def input(pin: int) -> int` | `0x1103` | Sync | Reentrant | `pin`: BCM pin (int) | None | None | `int` (1 or 0) | Reads the current logical level of an input GPIO pin. | `RPi.GPIO` library |
| `GPIO.add_event_detect` | `def add_event_detect(pin: int, edge: int, callback: Callable, bouncetime: int) -> None` | `0x1104` | Sync | Non-reentrant | `pin`: BCM pin (int), `edge`: e.g. `GPIO.RISING` (int), `callback`: ISR (Callable), `bouncetime`: bounce filter ms (int) | None | None | `None` | Binds a hardware interrupt service routine callback to trigger on rising/falling edge transitions. | `RPi.GPIO` library |
| `GPIO.remove_event_detect` | `def remove_event_detect(pin: int) -> None` | `0x1105` | Sync | Non-reentrant | `pin`: BCM pin (int) | None | None | `None` | Disconnects event listener callbacks and interrupts from a GPIO pin. | `RPi.GPIO` library |

### 1.2. PWM (Software PWM Timer)
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `GPIO.PWM` | `def PWM(pin: int, frequency: int) -> PWM_instance` | `0x1201` | Sync | Non-reentrant | `pin`: BCM pin (int), `frequency`: base frequency in Hz (int) | None | None | `PWM` instance | Instantiates software PWM configuration on a target pin. | `RPi.GPIO` class |
| `PWM.start` | `def start(duty_cycle: float) -> None` | `0x1202` | Sync | Non-reentrant | `duty_cycle`: duty percentage 0.0 - 100.0 (float) | None | None | `None` | Starts PWM signal output at the specified initial duty cycle. | `RPi.GPIO.PWM` |
| `PWM.ChangeDutyCycle` | `def ChangeDutyCycle(duty_cycle: float) -> None` | `0x1203` | Sync | Non-reentrant | `duty_cycle`: target duty cycle 0.0 - 100.0 (float) | None | None | `None` | Modifies active duty cycle of the PWM timer to change servo angles. | `RPi.GPIO.PWM` |
| `PWM.stop` | `def stop(self) -> None` | `0x1204` | Sync | Non-reentrant | None | None | None | `None` | Disables PWM signal generation on the target channel. | `RPi.GPIO.PWM` |

### 1.3. SPI / I2C Serial Interface
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `spi` | `def spi(port: int, device: int, gpio_DC: int, gpio_RST: int) -> spi_interface` | `0x1301` | Sync | Non-reentrant | `port` (int), `device` (int), `gpio_DC` (int), `gpio_RST` (int) | None | None | `spi` object | Opens connection to raw SPI bus device node for screen writing. | `luma.core.interface.serial` |
| `ssd1306` | `def ssd1306(serial: spi, width: int, height: int) -> ssd1306_device` | `0x1302` | Sync | Non-reentrant | `serial` (spi), `width` (int), `height` (int) | None | None | `ssd1306` device | Initialise SSD1306 hardware registers over serial. | `luma.oled.device` |

---

## 2. SYSTEM INTERFACE Layer (OS & Library Wrappers)

OS resource libraries and computer vision wrappers interfacing external resources.

### 2.1. OS Runtime & Threading
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Thread` | `threading.Thread(target=func, args=(), daemon=True)` | `0x2101` | Sync | Reentrant | `target` (Callable), `args` (tuple), `daemon` (bool) | None | None | `Thread` instance | Spawns a parallel kernel task executing target callback routines. | Python `threading` |
| `Lock` | `threading.Lock()` | `0x2102` | Sync | Reentrant | None | None | None | `Lock` object | Instantiates mutex synchronization objects for state guards. | Python `threading` |
| `Event` | `threading.Event()` | `0x2103` | Sync | Reentrant | None | None | None | `Event` object | Instantiates a state thread-safety control flag. | Python `threading` |
| `time.sleep` | `time.sleep(seconds)` | `0x2104` | Sync (blocking) | Reentrant | `seconds`: sleep duration (float) | None | None | `None` | Suspends active calling thread context execution. | Python `time` |

### 2.2. Camera Capture Interface
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `VideoCapture` | `cv2.VideoCapture(index, cap_v4l2)` | `0x2201` | Sync | Non-reentrant | `index` (int/str), `cap_v4l2`: capture backend (int) | None | None | `VideoCapture` | Locks OS device node and configures video buffer capture. | OpenCV (`cv2`) |
| `VideoCapture.read` | `def read() -> Tuple[bool, np.ndarray]` | `0x2202` | Sync | Non-reentrant | None | None | None | `Tuple[bool, np.ndarray]` | Pulls latest BGR image array from the device frame buffer. | OpenCV `VideoCapture` |
| `VideoCapture.release`| `def release() -> None` | `0x2203` | Sync | Non-reentrant | None | None | None | `None` | Releases locks on OS camera descriptors and cleans memory. | OpenCV `VideoCapture` |
| `cv2.flip` | `cv2.flip(frame, 1)` | `0x2204` | Sync | Reentrant | `frame` (np.ndarray), `flipCode` (int) | None | None | `np.ndarray` | Performs horizontal/vertical mirrors on pixel matrices. | OpenCV (`cv2`) |
| `cv2.imencode` | `cv2.imencode('.jpg', frame, params)` | `0x2205` | Sync | Reentrant | `ext` (str), `img` (np.ndarray), `params` (list) | None | None | `Tuple[bool, np.ndarray]` | Compresses pixel array to compressed JPEG byte block structures. | OpenCV (`cv2`) |

### 2.3. Image Drawing overlays
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cv2.rectangle` | `cv2.rectangle(img, pt1, pt2, color, thickness)` | `0x2301` | Sync | Non-reentrant | `img` (np.ndarray), `pt1`, `pt2` (tuples), `color` (BGR tuple), `thickness` (int) | None | `img`: target frame matrix | `np.ndarray` | Draws bounding boxes overlay directly on frame image matrices. | OpenCV (`cv2`) |
| `cv2.putText` | `cv2.putText(img, text, org, font, scale, color, thickness)` | `0x2302` | Sync | Non-reentrant | `img` (np.ndarray), `text` (str), `org` (tuple), `font` (int), `scale` (float), `color` (tuple), `thickness` (int) | None | `img`: target frame matrix | `np.ndarray` | Renders font glyph text characters over target image coordinates. | OpenCV (`cv2`) |

### 2.4. OLED Graphic Interface
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ssd1306.display` | `def display(image: Image)` | `0x2401` | Sync | Non-reentrant | `image`: 1-bit PIL canvas image | None | None | `None` | Decodes 1-bit bitmap frame buffers and writes pixels to screen via SPI. | `ssd1306` device |
| `ssd1306.clear` | `def clear() -> None` | `0x2402` | Sync | Non-reentrant | None | None | None | `None` | Wipes active display memory buffers on SSD1306. | `ssd1306` device |
| `ssd1306.hide` | `def hide() -> None` | `0x2403` | Sync | Non-reentrant | None | None | None | `None` | Shuts down active display driver power nodes. | `ssd1306` device |

---

## 3. SERVICE Layer (Core Domain Logic)

Nuggets of smart lock business rules: scanning, auth database hashing, Mamdani systems, and OLED drawing functions.

### 3.1. Face Recognition Diagnostics (`FaceDetection`)
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reload_recognizer` | `def reload_recognizer(self, recognizer_path: str) -> None` | `0x3101` | Sync | Non-reentrant | `recognizer_path`: file path (str) | None | None | `None` | Hot-reloads the newly trained LBPH XML model and identity mappings. | [face_detection.py](file:///d:/SmartLock_Fuzzy/backend/services/face_detection.py) |
| `analyze` | `def analyze(self, image: np.ndarray) -> dict` | `0x3102` | Sync | Reentrant | `image`: BGR camera matrix (np.ndarray) | None | None | `dict` (detection bounds, names, risk details) | Master wrapper performing Haar detection, ROI illumination averages, angles, and LBPH predictions. | [face_detection.py](file:///d:/SmartLock_Fuzzy/backend/services/face_detection.py) |
| `extract_registration_face` | `def extract_registration_face(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], dict]` | `0x3103` | Sync | Reentrant | `image`: BGR frame matrix (np.ndarray) | None | None | `Tuple[np.ndarray, dict]` (equalised crop, details) | Assesses blur (Laplacian variance), illumination, size, and yaw angle to approve samples. | [face_detection.py](file:///d:/SmartLock_Fuzzy/backend/services/face_detection.py) |

### 3.2. Fuzzy Security Engine (`FuzzySecurityController`)
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `evaluate` | `def evaluate(self, confidence: float, illumination: float, facial_angle: float) -> dict` | `0x3201` | Sync | Reentrant | `confidence` `[0-85]`, `illumination` `[0-255]`, `facial_angle` `[0-90]` | None | None | `dict` (risk, action, details) | Computes Membership Functions, fires 13 Mamdani rules via `pyfuzzylite`, and defuzzifies crisp risk score. | [fuzzy_controller.py](file:///d:/SmartLock_Fuzzy/backend/infra/fuzzy_controller.py) |

### 3.3. Keypad & Actuator Manager (`SmartLockHardware`)
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `start` | `def start(self) -> None` | `0x3301` | Sync | Non-reentrant | None | None | None | `None` | Initializes output rows, pulldown cols, attaches rising ISR callbacks, and locks servo. | [hardware_io.py](file:///d:/SmartLock_Fuzzy/backend/services/hardware_io.py) |
| `_col_interrupt` | `def _col_interrupt(self, col_pin: int) -> None` | `0x3302` | Async | Non-reentrant (lock guarded) | `col_pin`: BCM col pin number (int) | None | None | `None` | Callback checking debounce bounds, scanning coordinate keys, and updating passcode queues. | [hardware_io.py](file:///d:/SmartLock_Fuzzy/backend/services/hardware_io.py) |
| `_scan_key` | `def _scan_key(self, col_pin: int) -> Optional[str]` | `0x3303` | Sync | Non-reentrant | `col_pin`: Col pin number (int) | None | None | `Optional[str]` (key resolved) | Sequentially drives row pins HIGH, reading column state values to locate pressed keypad indexes. | [hardware_io.py](file:///d:/SmartLock_Fuzzy/backend/services/hardware_io.py) |
| `_submit_password` | `def _submit_password(self) -> None` | `0x3304` | Sync | Non-reentrant | None | None | None | `None` | Hashes PIN using SHA-256, validates lockout limits, resets fails, or sets 30s lockout countdown. | [hardware_io.py](file:///d:/SmartLock_Fuzzy/backend/services/hardware_io.py) |
| `unlock_door` | `def unlock_door(self, source: str = "unknown") -> None` | `0x3305` | Async (spawned thread) | Non-reentrant (lock guarded) | `source`: e.g. `"keypad"` or `"face"` (str) | None | None | `None` | Actuates Servo to 7.5% duty, waits 5 seconds, and returns it to 2.5% duty. Uses anti-vibration release. | [hardware_io.py](file:///d:/SmartLock_Fuzzy/backend/services/hardware_io.py) |
| `set_password` | `def set_password(self, current_pin: str, new_pin: str) -> dict` | `0x3306` | Sync | Non-reentrant | `current_pin` (str), `new_pin` (str) | None | None | `dict` (success status, msg) | Updates stored passcode SHA-256 hash upon verification checks. | [hardware_io.py](file:///d:/SmartLock_Fuzzy/backend/services/hardware_io.py) |

### 3.4. OLED display engine (`OLEDDisplay`)
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `_render` | `def _render(self, draw_fn) -> None` | `0x3401` | Sync | Non-reentrant (lock guarded) | `draw_fn`: canvas drawer (Callable) | None | None | `None` | Allocates new monochrome Pillow canvas image, draws elements, and calls SPI output buffers. | [oled_display.py](file:///d:/SmartLock_Fuzzy/backend/services/oled_display.py) |
| `show_enter_pin` | `def show_enter_pin(self, digits: int, total: int = 6) -> None` | `0x3402` | Sync | Non-reentrant | `digits` entered (int), `total` length (int) | None | None | `None` | Draws filled/hollow dots on OLED canvas indicating passcode progress. | [oled_display.py](file:///d:/SmartLock_Fuzzy/backend/services/oled_display.py) |
| `show_lockout` | `def show_lockout(self, seconds_remaining: int) -> None` | `0x3403` | Sync | Non-reentrant | `seconds_remaining` (int) | None | None | `None` | Renders lockout screen warning and wait countdown. | [oled_display.py](file:///d:/SmartLock_Fuzzy/backend/services/oled_display.py) |
| `show_registration` | `def show_registration(self, name: str, accepted: int, required: int) -> None` | `0x3404` | Sync | Non-reentrant | `name` (str), `accepted` (int), `required` (int) | None | None | `None` | Renders horizontal progress bar illustrating user sample extraction. | [oled_display.py](file:///d:/SmartLock_Fuzzy/backend/services/oled_display.py) |

### 3.5. Registration Manager (`FaceRegistrationManager`)
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `start` | `def start(self, name: str, samples: int = 30) -> dict` | `0x3501` | Sync | Non-reentrant | `name`: slug name (str), `samples` count (int) | None | None | `dict` (session start details) | Allocates directory bounds and prepares collector session struct values. | [registration.py](file:///d:/SmartLock_Fuzzy/backend/services/registration.py) |
| `process_frame` | `def process_frame(self, frame: np.ndarray, detector: FaceDetection) -> Optional[dict]` | `0x3502` | Sync | Non-reentrant | `frame` array, `detector` class | None | None | `Optional[dict]` (progress event) | Calls ROI diagnostics checks, writes image files to directories, and increments valid counts. | [registration.py](file:///d:/SmartLock_Fuzzy/backend/services/registration.py) |
| `train_model` | `def train_model(self) -> dict` | `0x3503` | Sync | Non-reentrant | None | None | None | `dict` (training summaries) | Reads directories, trains `LBPHFaceRecognizer`, and writes XML and label map files. | [registration.py](file:///d:/SmartLock_Fuzzy/backend/services/registration.py) |

---

## 4. APPLICATION Layer (API endpoints & UI interface)

FastAPI router, camera workers coordinating threads, and Next.js front-end triggers.

### 4.1. Camera Background Thread Orchestrator (`LocalCameraWorker`)
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `start` | `def start(self) -> None` | `0x4101` | Sync | Non-reentrant | None | None | None | `None` | Initializes display and keypad modules, and triggers camera capture loops on thread. | [local_camera.py](file:///d:/SmartLock_Fuzzy/backend/services/local_camera.py) |
| `_run` | `def _run(self) -> None` | `0x4102` | Async (loop run) | Non-reentrant | None | None | None | `None` | Background loop: captures frame, analyzes face metrics, evaluates fuzzy logic rules, and acts. | [local_camera.py](file:///d:/SmartLock_Fuzzy/backend/services/local_camera.py) |
| `update_latest_camera_result` | `def update_latest_camera_result(faces_count, detections, timestamp, camera_id, frame_base64, fuzzy, registration)` | `0x4103` | Sync | Non-reentrant (mutex protected) | Frame count, detections metadata list, timestamp, device identifier, base64 data, fuzzy outputs, registration state | None | None | `None` | Serializes frame analysis outputs and updates the main in-memory cache structure. | [get_camera.py](file:///d:/SmartLock_Fuzzy/backend/routers/get_camera.py) |

### 4.2. FastAPI HTTP API endpoints
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `get_latest_camera_frame` | `@router.get("/camera/frame")` | `0x4201` | Async | Reentrant | None | None | None | `JSON` (Success tag, base64 frame, fuzzy outputs) | Returns latest base64 JPEG camera frame image and metrics cache. | `GET /api/v1/camera/frame` |
| `start_registration` | `@app.post("/api/v1/register/start")` | `0x4202` | Async | Non-reentrant | Body: `RegisterStartRequest` | None | None | `JSON` (Started status details) | Triggers background registration buffers and starts tracking name bounds. | `POST /api/v1/register/start` |
| `change_keypad_password`| `@app.post("/api/v1/keypad/password")`| `0x4203` | Async | Non-reentrant | Body: `PasswordChangeRequest` | None | None | `JSON` (PIN update success state) | Validates credentials and modifies stored SHA-256 hash database logs. | `POST /api/v1/keypad/password` |

### 4.3. Next.js React Dashboard Client Hooks
| Service Name | Syntax | Service ID [hex] | Sync/Async | Reentrancy | Parameters (in) | Parameters (out) | Parameters (inout) | Return value | Description | Available via |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fetchFrame` | `const fetchFrame = useCallback(async () => { ... })` | `0x4301` | Async | Non-reentrant (polling drop logic) | None | None | None | `None` (triggers UI states) | Interval hook poller calling backend camera frames and feeding base64 image displays. | [LiveVideoStream.tsx](file:///d:/SmartLock_Fuzzy/frontend/src/components/LiveVideoStream.tsx) |
| `handleSubmit` | `const handleSubmit = async (e: React.FormEvent) => { ... }` | `0x4302` | Async | Non-reentrant (button disabling) | Form event (e) | None | None | `None` (triggers form states) | Submits password form body to backend passcode endpoints, handling API warnings. | [PasswordChange.tsx](file:///d:/SmartLock_Fuzzy/frontend/src/components/PasswordChange.tsx) |
