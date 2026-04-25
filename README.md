# HPM Open Physio System

HPM (open Hardware Psychophysiology Monitor) is an open-source, low-cost psychophysiology system for collecting ECG and GSR in browser-based experiments (e.g., Pavlovia / PsychoPy). It is designed so that any lab can assemble and run a research-grade setup with off-the-shelf parts and free software.

## Features

- Low-cost ECG, EDA, and rPPG recording with Arduino-based hardware
- Machine learning algorithm using ECG as groundtruth for enhanced rPPG accuracy
- Browser-based task integration via a Pavlovia bridge
- Desktop GUI for RA and participant-facing workflows
- Signal quality checks for ECG and EDA before each session
- CSV recording compatible with automated analysis pipeline


## Supported hardware

- Arduino-compatible microcontroller running the HPM physio bridge firmware
- GSR sensor board (CJMCU-6107 or Grove-style GSR)
- ECG front-end (e.g., AD8232-based module)
- ADS1115 ADC for high-resolution GSR acquisition
- Battery power pack and USB data blocker (recommended)

<img width="4271" height="2493" alt="HPM-Open-Physio-System" src="https://github.com/user-attachments/assets/d5941abe-bb04-47f8-90da-c55d7ceb6983" />

See `hardware/` for wiring diagrams, bill of materials, and enclosure notes.

## Supported platforms

- Windows 10/11 (64-bit)
- macOS 12+ (Intel and Apple Silicon)

Linux may work from source but is not a primary support target.

## Quick start (end users)

1. Download the latest release from the **Releases** page.
2. Install the desktop app for your platform (Windows `.exe` or macOS `.dmg`).
3. Connect the HPM hardware via USB.
4. Launch the app and select **Signal Quality Check** to verify ECG and GSR.
5. When both indicators are green, you are ready to start an experiment.

## Quick start (developers / builders)

1. Clone this repository.
2. Install Python 3.11+ and `pip`.
3. In `desktop/gui/`, create a virtual environment and install requirements:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Run the GUI directly:

   ```bash
   python hpm_gui.py
   ```

5. To build standalone apps, see `desktop/packaging/`.

## Repository layout

- `firmware/` – Arduino sketches and board-specific notes
- `desktop/` – GUI, bridge server, and packaging configs
- `hardware/` – BOM, wiring diagrams, enclosure CAD, and photos
- `docs/` – Quick-start, troubleshooting, and validation notes
- `examples/` – Sample data and analysis outputs

## Citing HPM

If you use HPM in a publication, please cite this repository. A machine-readable citation is provided in `CITATION.cff` and via the **Cite this repository** button on GitHub.

## License

- Software (desktop app and firmware): MIT License
- Hardware designs (PCBs, wiring diagrams, models): CERN OHL v2 or later
- Documentation and figures: CC BY 4.0

See `LICENSE`, `LICENSE-HARDWARE`, and `LICENSE-DOCS` for details.
