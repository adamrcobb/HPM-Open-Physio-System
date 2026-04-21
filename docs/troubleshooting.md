# HPM Troubleshooting Guide

This guide covers the most common hardware and software issues encountered with the HPM system.

---

## GSR signal issues

### GSR is stuck at a high value (~40 µS) regardless of finger contact

**Cause:** The 3.5mm audio jack connecting the GSR electrodes to the CJMCU-6107 board is
partially seated. This creates a resistive short between the tip and ring contacts, adding a
fixed low-resistance path in parallel with the skin and producing a falsely high conductance
reading.

**Fix:**
1. Press the audio jack firmly into the socket until it clicks or seats fully.
2. In the GUI Signal Quality page, click **Connect** again (or disconnect and reconnect).
   The EMA resets on every new connection, so the displayed value will refresh immediately.

> This is the most common GSR issue. Always check jack seating first.

---

### GSR reads 0 µS or near-zero with electrodes attached

**Cause:** Electrodes are disconnected, the cable is broken, or the ADS1115 is not responding.

**Fix:**
1. Check that the electrode cable is firmly connected at both the electrode end and the jack end.
2. Open the Arduino Serial Monitor (115200 baud) and look for:
   ```
   # GSR startup: raw=... V=...V gsr=0.00 uS
   # Note: near VCC = electrodes disconnected.
   ```
   Near-VCC readings mean the electrode circuit is open. Check the cable continuity.
3. If the ADS1115 is not detected, the Serial Monitor will show:
   ```
   # FATAL: ADS1115 not found!
   ```
   In that case, check the I2C wiring: SDA → A4, SCL → A5, VDD → 5V, GND → GND, ADDR → GND.

---

### GSR signal degrades or disappears when the enclosure lid is closed

**Cause:** The enclosure lid is physically pressing on the audio jack or electrode cable, partially
dislodging the connector. This can also be caused by the power cable shifting when the lid closes.

**Fix:**
1. Check that the audio jack cable exits the enclosure without being pinched or kinked.
2. Add strain relief (a cable tie anchor or foam pad) so the cable cannot move when the lid closes.
3. Check the power cable routing — resoldering the power connector may be needed if it is prone
   to shorting when the enclosure is closed.
4. Run on battery power (not mains-plugged laptop) to rule out 50/60 Hz interference from the
   power adapter coupling into the high-impedance GSR signal path.

---

### GSR value is plausible but noisy / very spiky

**Cause:** Poor electrode contact, dry skin, or the EMA smoothing alpha is set too high.

**Fix:**
1. Ensure the participant's fingers are clean and dry before applying electrodes.
2. Apply a small amount of electrolyte gel if available.
3. If spikiness persists, the EMA alpha in `hpm_gui.py` can be adjusted:
   - Current: `alpha = 0.85` (fast, responsive)
   - Smoother: `alpha = 0.90` (slower settling, less noise)
   - Search for `0.85` in the file to find all three locations.

---

## ECG signal issues

### ECG shows a flat line or "Weak signal" in Signal Quality Check

**Cause:** Electrode pads are dry, not making contact, or the Olimex shield is not seated correctly.

**Fix:**
1. Check that the ECG electrode pads are firmly attached to the participant's wrists/ankles.
2. Moisten the electrode pads slightly if they appear dry.
3. Confirm the Olimex SHIELD-EKG-EMG is correctly reading on pin A1.
4. Open the Arduino Serial Monitor to verify the ECG column (column 4) is changing.

---

### ECG BPM detection shows "Signal OK — detecting…" but no BPM

**Cause:** The buffer is accumulating but the R-peak detector has not found enough peaks yet.
This is normal for the first 5–10 seconds after connecting.

**Fix:** Wait 10 seconds after connecting before judging signal quality. If no BPM appears after
30 seconds, check electrode contact and placement.

---

## Serial / connection issues

### The port dropdown shows "No ports found"

**Cause:** The Arduino is not connected, the USB data blocker is blocking data, or the driver is
not installed.

**Fix:**
1. Confirm the USB data blocker is plugged in correctly — some blockers block data in both
   directions if installed backwards.
2. On Windows, install the Arduino USB driver from the Arduino IDE installer.
3. Try a different USB cable — some cables are charge-only and carry no data.
4. On macOS, the port will appear as `/dev/cu.usbmodem...`. If it does not appear, check
   **System Settings → Privacy & Security** for any USB device blocking.

---

### "Connecting to port…" never resolves / GUI hangs

**Cause:** The Arduino Serial Monitor is open in the Arduino IDE at the same time as the GUI,
holding the port.

**Fix:** Close the Arduino Serial Monitor before launching the GUI. Only one application can
hold a serial port at a time.

---

### GUI connects but receives no data

**Cause:** Baud rate mismatch, or the firmware is not running the v3.3 sketch.

**Fix:**
1. Confirm the firmware is flashed and the Serial Monitor shows the v3.3 header at 115200 baud.
2. Confirm the GUI is set to 115200 baud (this is the default and should not need changing).
3. Press the **Reset** button on the Arduino and reconnect.

---

## Software / installation issues

### `pip install -r requirements.txt` fails on `neurokit2`

**Cause:** Older pip or missing build tools.

**Fix:**
```bash
pip install --upgrade pip
pip install neurokit2
```

---

### `import cv2` error when launching the GUI

**Cause:** `opencv-python` was not installed or the wrong variant was installed.

**Fix:**
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.11.0.86
```

---

### `websockets` ImportError or version mismatch in the Pavlovia bridge

**Cause:** The bridge requires the v12+ websockets API. Older versions use a different connection
handler signature.

**Fix:**
```bash
pip install "websockets>=12.0"
```

---

## Still stuck?

Open an issue on the GitHub repository at `https://github.com/adamrcobb/Open-Psychophys`
and include:
- Your OS and Python version
- The HPM version (tag or commit hash)
- The full error message or Serial Monitor output
- A brief description of what you expected vs. what happened
