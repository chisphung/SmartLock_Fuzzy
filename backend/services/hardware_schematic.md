# Hardware Wiring & Schematic

This document provides detailed hardware wiring diagrams and electrical safety guidelines for connecting the **Raspberry Pi 4 Model B** to external peripheral components in the SmartLock Fuzzy system.

---

## 1. Pin Mapping Matrix

The GPIO pins below are configured directly in the source code of the [SmartLockHardware](file:///d:/SmartLock_Fuzzy/backend/services/hardware_io.py#L57) and [OLEDDisplay](file:///d:/SmartLock_Fuzzy/backend/services/oled_display.py#L32) classes.

| Device Name | Device Pin | Raspberry Pi GPIO (BCM ID) | Physical Pin ID | Signal Type (I/O) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4x4 Keypad** | Row 1 | **GPIO 17** | Pin 11 | Output (HIGH) | Row scan output |
| | Row 2 | **GPIO 27** | Pin 13 | Output (HIGH) | Row scan output |
| | Row 3 | **GPIO 22** | Pin 15 | Output (HIGH) | Row scan output |
| | Row 4 | **GPIO 5** | Pin 29 | Output (HIGH) | Row scan output |
| | Col 1 | **GPIO 6** | Pin 31 | Input (Pull-Down) | Triggers `RISING` interrupt |
| | Col 2 | **GPIO 13** | Pin 33 | Input (Pull-Down) | Triggers `RISING` interrupt |
| | Col 3 | **GPIO 19** | Pin 35 | Input (Pull-Down) | Triggers `RISING` interrupt |
| | Col 4 | **GPIO 26** | Pin 37 | Input (Pull-Down) | Triggers `RISING` interrupt |
| **Servo Motor**| PWM (Signal) | **GPIO 18** | Pin 12 | Output Hardware PWM | Hardware PWM Channel 0 |
| | VCC (Power) | **5V External** | Independent external power | Power | **DO NOT** connect directly to the Pi |
| | GND (Ground) | **GND External**| Shared ground pin | Ground | Shared ground with the Pi |
| **OLED Display** | GND (Ground) | **GND** | Pin 9 / 14 / 20 / 25 | Ground | Ground |
| (SSD1306 SPI) | VCC (Power) | **3.3V** | Pin 1 / 17 | Power | 3.3V Power Supply |
| | SCL / SCLK | **GPIO 11** (SCLK)| Pin 23 | SPI0 SCLK | SPI Clock |
| | SDA / MOSI | **GPIO 10** (MOSI)| Pin 19 | SPI0 MOSI | SPI Data |
| | RST (Reset) | **GPIO 25** | Pin 22 | Output | Display reset |
| | D/C (Data/Cmd) | **GPIO 24** | Pin 18 | Output | Data/Command select |
| | CS (Chip Select)| **GPIO 8** (CE0) | Pin 24 | SPI0 CE0 | SPI Chip Select |

---

## 2. Visual Schematic (ASCII Schematic Diagram)

```text
                      RASPBERRY PI 4 (Header 40-pin)
                               +-----+-----+
                  (3.3V Power) | 1   | 2   | (5V Power)
                               | 3   | 4   | (5V Power)
                               | 5   | 6   | (GND)
                               | 7   | 8   | 
        OLED GND ------------->| 9   | 10  | 
                               | 11  | 12  | -------------> SERVO PWM (GPIO 18)
       KEYPAD ROW 2 (GPIO 27) ->| 13  | 14  | 
       KEYPAD ROW 3 (GPIO 22) ->| 15  | 16  | 
        OLED VCC ------------->| 17  | 18  | -------------> OLED D/C (GPIO 24)
        OLED SDA ------------->| 19  | 20  | 
                               | 21  | 22  | -------------> OLED RST (GPIO 25)
        OLED SCLK ------------>| 23  | 24  | -------------> OLED CS (GPIO 8)
                               | 25  | 26  | 
                               | 27  | 28  | 
       KEYPAD ROW 4 (GPIO 5) ->| 29  | 30  | 
       KEYPAD COL 1 (GPIO 6) ->| 31  | 32  | 
       KEYPAD COL 2 (GPIO 13) ->| 33  | 34  | 
       KEYPAD COL 3 (GPIO 19) ->| 35  | 36  | 
       KEYPAD COL 4 (GPIO 26) ->| 37  | 38  | 
                               | 39  | 40  | 
                               +-----+-----+

       [ 4x4 MATRIX KEYPAD ]
       Row 1, 2, 3, 4  ----> Connect to GPIO pins 17, 27, 22, 5
       Col 1, 2, 3, 4  ----> Connect to GPIO pins 6, 13, 19, 26

       [ SSD1306 SPI OLED DISPLAY ]
       GND, VCC, SCL, SDA, RST, D/C, CS ----> Refer to the detailed pin matrix table above

       [ SERVO MOTOR ]
       PWM Signal      ----> Connect to GPIO pin 18
       VCC (+)         ----> Connect to the Positive (+) terminal of the external 5V power supply
       GND (-)         ----> Connect to the Negative (-) terminal of the external 5V power supply & bridge to Raspberry Pi GND
```

---

## 3. Design Principles & Electrical Safety Rules

> [!CAUTION]
> **SERVO MOTOR POWER SUPPLY WARNING:**
> Servo motors (such as SG90, MG90S) have very high startup and stall currents (potentially up to 500mA - 1A). If you connect the VCC of the Servo directly to the 5V power pins of the Raspberry Pi, this current draw spike can cause severe voltage drops on the Pi (brownout), leading to system hangs, crashes, or damage to the Raspberry Pi's Power Management Integrated Circuit (PMIC).
> 
> **Safe Solution:**
> 1. Use an independent external 5V power supply (e.g., MB102 power module, power bank, or 5V adapter) dedicated strictly to the Servo motor.
> 2. **REQUIRED**: Connect the ground (GND) of this external supply to any GND pin of the Raspberry Pi. This establishes a common reference voltage level for the PWM control signal.

> [!IMPORTANT]
> **LOGIC LEVELS:**
> - The Raspberry Pi's GPIO pins operate strictly at a **3.3V** logic level. Any voltage exceeding 3.3V applied to a GPIO pin can damage that pin or permanently destroy the Raspberry Pi board.
> - The 4x4 matrix keypad uses the Pi's internal pull-down resistors (`GPIO.PUD_DOWN`) and is scanned using 3.3V logic levels from the Pi's output pins, making it completely safe.
> - The SSD1306 OLED display using SPI communication also needs to be powered with 3.3V from physical pin 17 or 1 of the Raspberry Pi to ensure the SCLK, MOSI, and CS signal lines operate at 3.3V logic levels.

---

## 4. Hardware Interaction Programming Details

1. **Matrix Keypad**:
   - Initialized in [hardware_io.py](file:///d:/SmartLock_Fuzzy/backend/services/hardware_io.py) by setting the row pins to `GPIO.OUT` (defaulting to `HIGH` state) and column pins to `GPIO.IN` with `pull_up_down=GPIO.PUD_DOWN`.
   - Utilizes `GPIO.add_event_detect(col, GPIO.RISING, callback=..., bouncetime=250)` to catch rising-edge interrupts, minimizing CPU load compared to constant scanning (polling).
2. **Lock Servo**:
   - Uses the `RPi.GPIO` library to setup Hardware PWM on GPIO pin 18 at a frequency of 50Hz.
   - The `unlock_door` method runs on a separate background thread (`threading.Thread`) to prevent blocking the main camera execution thread. The PWM signal is completely disabled (`ChangeDutyCycle(0)`) after a successful open/lock sequence to suppress mechanical vibrations (buzzing noise) and conserve power.
3. **OLED Display**:
   - Communicates over the SPI bus via the `luma.core.interface.serial.spi` module.
   - Leverages a synchronization lock (`self._lock = threading.Lock()`) to prevent conflicts from concurrent writes by the keypad thread and the camera thread.
