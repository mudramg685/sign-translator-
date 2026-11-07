# Real-Time ISL to Text and Speech Translator

An academic project scaffold for recognizing a documented subset of Indian Sign Language (ISL) gestures from hand landmarks, displaying confirmed signs as text, speaking the text, and saving translation history.

This repository is being built in phases. At this stage it contains the project structure and configuration; gesture recognition and application features are not implemented yet. It will not claim full ISL translation. The initial vocabulary and model quality will be documented after data collection and evaluation.

## Modules

- `java-client`: JavaFX desktop application.
- `java-backend`: Spring Boot REST API and MySQL persistence.
- `python-ml-service`: FastAPI service, camera/landmark pipeline, dataset tools, and model artifacts.
- `database`: database setup and schema resources.
- `docs`: architecture and operating notes.

## Technology baseline

- Java 21 LTS, Spring Boot 3.5.16, JavaFX 21.0.7, Maven.
- Python 3.11, FastAPI, OpenCV, MediaPipe, PyTorch, NumPy, and Pandas.
- MySQL 8.4 LTS for local development.

The baseline versions are pinned or bounded in the module build files. Python packages may require platform-specific installation options for GPU acceleration; CPU execution is the default.

## Planned data flow

The Python service captures frames and extracts hand landmarks, then returns a model prediction and confidence as JSON. The Java backend filters and processes predictions, assembles confirmed signs into text, and stores translation history. The JavaFX client provides the live translator, editing controls, history, and speech output.

See [docs/architecture.md](docs/architecture.md) for the module diagram, API outline, and phase roadmap.

## Current status

Phase 3 adds webcam landmark collection and CSV inspection. The classifier, prediction service, and desktop application are not implemented yet, so no gesture predictions are available.
