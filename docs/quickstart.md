# HPM Quick-Start Guide

This guide gets you from unboxed hardware to a verified, running session in about 30 minutes.

---

## What you need

### Hardware

| Item | Purpose |
|---|---|
| Arduino Uno R3 (or compatible) | Microcontroller and data acquisition hub |
| CJMCU-6107 GSR sensor board | Galvanic skin response measurement |
| ADS1115 16-bit ADC breakout | High-resolution GSR digitisation |
| Olimex SHIELD-EKG-EMG | ECG front-end |
| GSR finger electrodes (3.5mm audio jack) | Skin contact sensors |
| USB data blocker | Prevents ground loops via laptop USB |
| Battery power bank (5V/2A) | Isolated power for the Arduino |
| Enclosure with strain-relieved cable exits | Houses the hardware |

### Software

- Python 3.11 or later ([python.org](https://python.org))
- Arduino IDE 2.x ([arduino.cc](https://arduino.cc))
- This repository

---

## Step 1 — Flash the firmware

1. Open Arduino IDE.
2. Install the required libraries via **Sketch → Include Library → Manage Libraries**:
   - `Adafruit ADS1X15`
   - `Adafruit BusIO`
3. Open `firmware/arduino_physio_bridge_v3_3/arduino_physio_bridge_v3_3.ino`.
4. Select your board: **Tools → Board → Arduino Uno**.
5. Select the correct port: **Tools → Port → /dev/cu.usbmodem...** (macOS) or **COM...** (Windows).
6. Click **Upload** (right-arrow button).
7. Open **Tools → Serial Monitor** at **115200 baud** and confirm you see:

   ```
   # ARDUINO PHYSIO BRIDGE v3.3
   # VCC : 5.xxx V
   # Ready. Waiting for 'experiment_start' marker.
   ```

8. Close the Serial Monitor before launching the GUI (both cannot hold the port simultaneously).

---

## Step 2 — Install Python dependencies

```bash
cd desktop/gui
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs all required packages — numpy, scipy, neurokit2, pyserial, and others.

---

## Step 3 — Connect the hardware

1. Connect the Arduino to the battery pack via USB.
2. Connect the battery pack's data port to your laptop via the **USB data blocker**.  
   The data blocker passes data but blocks the power line, preventing ground loops.
3. Plug the **GSR electrode cable** (3.5mm audio jack) into the CJMCU-6107 jack.  
   > ⚠️ **Important:** Seat the audio jack fully until it clicks. A partially-seated jack
   > causes a fixed ~40 µS false reading that does not change with finger contact.
4. Place the electrodes on the participant's non-dominant hand (index and middle finger pads).
5. Verify the ECG electrode pads are attached and connected to the Olimex shield.

---

## Step 4 — Launch the GUI

```bash
python hpm_gui.py
```

The HPM GUI will open. Select the session type and participant ID, then proceed to
**Signal Quality Check**.

---

## Step 5 — Signal Quality Check

1. Click **Signal Quality Check** from the main menu.
2. Select the correct serial port from the dropdown (the Arduino port).
3. Click **Connect**.
4. Verify both indicators turn green:
   - **ECG** — shows a BPM reading (typically 55–100 BPM at rest)
   - **GSR** — shows "OK" with a µS value (typically 2–20 µS at rest)
5. If both are green, click **Continue** to proceed.

> 💡 If GSR shows a value above 30 µS without finger contact, reseat the audio jack
> and click **Connect** again to reset the EMA. See `troubleshooting.md` for details.

---

## Step 6 — Run a session

1. On the session setup page, enter the participant ID and choose session parameters.
2. Start the Pavlovia experiment in the browser.
3. Click **Start Recording** in the GUI.
4. The LED on the Arduino will illuminate when recording begins.
5. When the experiment ends, the GUI auto-stops recording and saves a CSV to the output folder.

---

## Step 7 — Analyse the data

Run the analysis pipeline on the saved CSV:

```bash
python psychophysiology_pipeline_v7_17_2.py --input path/to/session_data.csv
```

Output figures and a summary CSV will be saved alongside the input file.

---

## Quick-reference: expected signal values

| Signal | Expected range at rest | Action if outside range |
|---|---|---|
| ECG BPM | 55–100 BPM | Check electrode contact and placement |
| GSR (EDA) | 2–20 µS | Check electrode contact; reseat audio jack |
| GSR (no contact) | ≈ 0 µS | Normal — electrodes disconnected |
| GSR (stuck high, e.g. 40+ µS) | Not real signal | Reseat audio jack fully; reconnect in GUI |

---

## Next steps

- See `troubleshooting.md` for solutions to common hardware and software issues.
- See `hardware/bom/` for the full bill of materials and part numbers.
- See `hardware/wiring/` for wiring diagrams.
