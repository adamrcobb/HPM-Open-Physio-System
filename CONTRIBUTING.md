# Contributing to HPM

Thank you for considering a contribution to the HPM open physio system. Contributions from researchers, students, and developers are welcome.

## Ways to contribute

- Report bugs (software, firmware, or hardware)
- Improve documentation and examples
- Add support for additional sensors or boards
- Improve performance, robustness, or packaging

## Getting started

1. Fork the repository and create a feature branch.
2. If you change behavior, add or update documentation in `docs/`.
3. For code changes, run existing tests (if present) or add minimal checks.
4. Open a pull request with a clear description of the change and testing.

## Coding style

- Python: follow PEP 8 where practical.
- Arduino: keep sketches readable, comment sensor assumptions.
- Do not introduce new dependencies lightly; discuss major changes in an issue first.

## Hardware contributions

If you submit changes to wiring diagrams, enclosure models, or PCBs:

- Include updated source files (e.g., KiCad, CAD, SVG) and exported images.
- Update the BOM and wiring notes if part numbers or connections change.
- Add a brief note to `docs/validation/` describing how you tested the change.

## Issue reporting

When reporting a bug, please include:

- OS version and platform
- HPM version (tag or commit hash)
- A short description of what you expected vs. what happened
- Relevant logs, screenshots, or sample data, if possible

## Code of conduct

This project expects respectful, inclusive behavior in issues and pull requests. Please see `CODE_OF_CONDUCT.md`.
