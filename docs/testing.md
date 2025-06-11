# Testing Guide

This document explains how to run and extend the test-suite for **Audio-Aligner**.

---

## 1. Quick Start

```bash
# install optional test dependencies
pip install .[test]

# run fast unit + integration tests only
pytest

# include benchmarks
pytest -m benchmark

# generate HTML coverage report
pytest --cov=audio_aligner --cov-report=html
```

## 2. Test Layout

```
tests/
├── fixtures/          # synthetic audio & metadata generators
├── unit/              # quick, isolated tests
├── integration/       # CLI-level tests with Click's runner
├── performance/       # speed & memory benchmarks (opt-in)
└── __init__.py
```

### Key Fixtures

* **mock_audio_data.py** – produces NumPy arrays for sine, noise and onset patterns.
* **mock_media_info.py** – returns MediaInfo-like dictionaries with arbitrary codecs/FPS.

## 3. Benchmarks

Performance tests are tagged with `@pytest.mark.benchmark` and **skipped by default** via `pytest.ini`. Run them explicitly:

```bash
pytest -m benchmark
```

## 4. Continuous Integration

GitHub Actions workflow `.github/workflows/test.yml` executes the suite on:

* Windows, macOS, Linux
* Python 3.11 & 3.12
* Fails if coverage drops below 90 % (enforced via Codecov or future policy).

## 5. Adding Tests

1. Place new files under `tests/unit/` (or other appropriate folder).
2. Prefer Hypothesis for data-driven properties.
3. Mock external tools (`ffmpeg`, `eac3to`, etc.) with `pytest-mock` or `unittest.mock.patch`.
4. Keep runtime under **300 ms** per unit test; use `-k slow` markers if needed. 