# SignBridge 🌉🤟

> **Two-Way Real-Time American Sign Language (ASL) Communication Bridge**  
> Empowering fluid, bidirectional conversations between the Deaf/Hard-of-Hearing community and non-signers using computer vision, machine learning, and skeletal sign synthesis.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue.svg)](https://www.typescriptlang.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-FF6F00.svg)](https://developers.google.com/mediapipe)
[![Tests](https://img.shields.io/badge/Pytest-91%20Passed%20(100%25)-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/Coverage-89.65%25-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/Ruff-Passed-black.svg)](https://docs.astral.sh/ruff/)
[![Types](https://img.shields.io/badge/Mypy-0%20Issues-blue.svg)](https://mypy-lang.org/)

---

## 📖 Table of Contents
- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Team Roles & Ownership](#-team-roles--ownership)
- [Machine Learning Models](#-machine-learning-models)
- [API Contract Reference](#-api-contract-reference)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [1. Backend Setup](#1-backend-setup)
  - [2. Frontend Setup](#2-frontend-setup)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Repository Structure](#-repository-structure)
- [License](#-license)

---

## 🌟 Overview

SignBridge bridges the communication gap between American Sign Language users and non-signers without requiring specialized data gloves or proprietary hardware. Using an ordinary consumer webcam, SignBridge delivers:

1. **Sign to Text (Fingerspelling & Isolated Words)**: MediaPipe hand and upper-body landmark tracking streams skeletal coordinate tensors into a trained Random Forest classifier and temporal gesture segmenter, outputting confirmed letters, words, and full English sentences.
2. **Text to Sign (Dynamic 3D Skeletal Animation)**: Translates arbitrary English words and phrases into anatomically accurate 21-joint skeletal hand and upper-body animations on an interactive HTML5 canvas, complete with timeline scrubbing, speed modulation, and letter-by-letter fingerspelling fallbacks.

---

## 🏛 System Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Browser (React 19 + TypeScript + Vite)"]
        Cam["Webcam Video Feed"] --> MP["MediaPipe Tasks Vision<br/>(Hands & Pose Landmarker)"]
        MP --> Overlay["Canvas Skeletal Overlay<br/>(21 Hand + 33 Pose Points)"]
        MP --> UI["Application Viewports<br/>(SpellPanel / WordsPanel / CollectPanel)"]
        TTS_UI["TextToSignPlayer Canvas<br/>(Skeletal Animation Renderer)"]
    end

    subgraph Backend ["Backend Platform (FastAPI + Python 3.11)"]
        API["REST & WebSocket Endpoints<br/>(/api/recognize, /api/text-to-sign, /ws/*)"]
        SM["Session Manager & TTL Cache<br/>(In-Memory Stateful Tracking)"]
        STAB["Stability & Smoothing Filter<br/>(Streak & Cooldown Logic)"]
        RF["ASL Random Forest Model<br/>(300 Trees, 63 Landmark Features)"]
        SE["Suggestion & Vocabulary Engine<br/>(Dictionary Auto-Completion)"]
        DB[(SQLite Storage<br/>signbridge.db)]
    end

    UI -- "POST /api/recognize (60ms stream)" --> API
    API --> SM --> STAB --> RF
    RF --> SE --> API
    API --> UI
    API -- "Audit & Samples" --> DB

    TTS_UI -- "POST /api/text-to-sign" --> API
    API -- "Keyframe Sequences" --> TTS_UI
```

---

## 🚀 Key Features

### 1. 🔤 Sign to Text (Spell Mode)
- **High Accuracy (95.07%)**: Classifies all 26 ASL alphabet characters (A–Z) using 63 MediaPipe 3D hand coordinates.
- **Ultra-Fast Real-Time Inference**: Throttled at **60ms (~16 fps)** for fluid typing without muscle fatigue.
- **Stability Smoothing**: Configurable auto-confirm streak (0.25s / 0.4s / 0.8s hold presets) with real-time visual progress meters.
- **Instant Manual Confirmation**: Click **`+ Add "{letter}" Now`** to immediately append a detected letter without waiting.
- **Word & Sentence Buffer**: Word chips, backspace (`⌫`), word commitment (`↵ Add Word`), clear (`✕ Clear`), and instant dictionary auto-suggestions.

### 2. 🎬 Text to Sign (Animation Mode)
- **High-Fidelity Skeletal Synthesis**: Renders smooth 21-joint skeletal hand animations on an interactive dark canvas.
- **Comprehensive Timeline Controls**: Play, pause, restart (⏮), step backwards (◀ Prev), and step forwards (Next ▶).
- **Variable Playback Speed**: Modulate animation tempo from **0.5x, 1.0x, 1.5x, to 2.0x**.
- **Lexical & Fingerspelling Fallback**: Known vocabulary words animate full gestures; unrecognized words automatically decompose into fluent letter-by-letter fingerspelled sequences.

### 3. 🌐 Word Signs (Words Mode)
- **Upper-Body + Hand Fusion**: Tracks shoulder, elbow, and wrist joints alongside hands using `pose_landmarker.task`.
- **Everyday Vocabulary**: Supports 30 target words (`hello`, `thank you`, `please`, `water`, `help`, `apple`, etc.).

### 4. 🗄 Translation History Drawer
- **Persistent Log**: Captures spoken/signed phrases with timestamps and direction (`Sign → Text` or `Text → Sign`).
- **One-Click Replay**: Re-routes any historical sentence directly into the Text-to-Sign synthesizer for demonstration.

### 5. 🛠 Developer Landmark Inspector & Dataset Tool
- **Live Landmark Telemetry**: Inspects hand joint detection, handedness confidence (`Left` vs `Right`), and pose points.
- **Ground-Truth Recording**: Directly records raw landmark tensors to SQLite via `POST /api/samples` to collect custom training data.

---

## 👥 Team Roles & Ownership

The SignBridge system was engineered collaboratively across three specialized roles:

```mermaid
graph LR
    A["Person A<br/>Backend & Integration Lead"] --- B["Person B<br/>Machine Learning Engineer"]
    B --- C["Person C<br/>Frontend & UI/UX Engineer"]
    C --- A
```

- **Person A (Backend, Infrastructure & Integration)**:
  - Architecture of FastAPI platform, session lifecycle, and stability filtering.
  - Frozen Section 12 REST API and WebSocket streaming routes (`/ws/recognize`, `/ws/recognize-words`).
  - SQLite schema, sample collection endpoints, and history audit repository.
  - Test harness: 91 unit, integration, and system tests (89.65% coverage), Ruff linting, Mypy typing.
- **Person B (Machine Learning & Data Pipelines)**:
  - Kaggle ASL dataset preparation, cleaning, and normalization.
  - Feature extraction pipeline converting 21 3D points $(x, y, z)$ into 63 normalized coordinates.
  - ASL Alphabet Model (`asl_model.joblib`): 300-tree Random Forest, 95.07% accuracy across 26 classes.
  - 30-word isolated gesture model architecture and vocabulary design (`word_vocab.json`).
- **Person C (Frontend UI/UX & MediaPipe Engine)**:
  - React 19 + TypeScript + Vite component architecture.
  - MediaPipe Vision Tasks integration (`hand_landmarker` and `pose_landmarker.task`).
  - Panel implementations: `SpellPanel`, `WordsPanel`, `CollectPanel`, `HistoryDrawer`, `DeveloperSaveControl`.
  - Skeletal canvas rendering engine with timeline scrubbing and responsive design system.

---

## 🧠 Machine Learning Models

### ASL Alphabet Model
- **File**: `backend/models/asl_model.joblib`
- **Checksum (SHA-256)**: `D1B0600C19505F85F8C4077F0868FC01BA92414A4E737DD1721F60808C7DFC11`
- **Algorithm**: Random Forest Classifier (300 estimators, Gini criterion)
- **Input Dimensions**: 63 features (21 MediaPipe hand landmarks $\times [x, y, z]$, wrist-centered & normalized)
- **Output Classes**: 26 classes (`A` through `Z`)
- **Evaluation Accuracy**: **95.0745%** on held-out test split
- **Confusion Matrix & Metrics**: See [`backend/models/confusion_matrix.png`](backend/models/confusion_matrix.png) and [`backend/models/metrics.json`](backend/models/metrics.json).

---

## 📡 API Contract Reference

All endpoints conform strictly to the frozen Section 12 interface contract:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System health check, active model status, and version |
| `GET` | `/api/languages` | Supported sign languages (`ASL`, `ISL`) |
| `GET` | `/api/vocabulary` | Supported lexicon and fingerspelling dictionary |
| `POST` | `/api/recognize` | Single-frame fingerspelling recognition with session tracking |
| `POST` | `/api/session/reset` | Resets active word buffer and filters for a session |
| `POST` | `/api/session/backspace`| Deletes last letter from the in-progress word |
| `POST` | `/api/session/commit` | Commits current spelled word into the sentence |
| `POST` | `/api/words/recognize` | Isolated word gesture sequence classification |
| `POST` | `/api/text-to-sign` | Synthesizes skeletal keyframe animation sequences from text |
| `GET` | `/api/history` | Fetches translation history log |
| `DELETE`| `/api/history` | Clears translation history log |
| `POST` | `/api/samples` | Saves labeled landmark samples for retraining |
| `WS` | `/ws/recognize` | Real-time bidirectional WebSocket stream for fingerspelling |
| `WS` | `/ws/recognize-words` | WebSocket stream for continuous word gesture recognition |

---

## 🛠 Getting Started

### Prerequisites
- **Python**: Version 3.11 or higher
- **Node.js**: Version 18.0 or higher (with `npm`)
- **Webcam**: Standard USB or integrated webcam

---

### 1. Backend Setup

```bash
# Clone the repository
git clone https://github.com/thenix09/SIGN_BRIDGE.git
cd SIGN_BRIDGE

# Create and activate a Python virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start the FastAPI backend server
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend API and interactive OpenAPI documentation will be live at:
- **API Server**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 2. Frontend Setup

In a new terminal window:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```

The web application will open at **[http://localhost:5173](http://localhost:5173)**.

---

## 🧪 Testing & Quality Assurance

SignBridge enforces a strict verification pipeline:

```bash
# 1. Run full backend test suite with coverage
python -m pytest backend/tests --cov=backend/app --cov-fail-under=80 -W ignore

# 2. Check code style and formatting
python -m ruff check .

# 3. Verify strict static type hints
python -m mypy backend/app

# 4. Verify frontend TypeScript build
cd frontend && npm run build
```

### Current Verification Metrics:
- **Unit & System Tests**: 91 passed / 91 executed (100% pass rate)
- **Code Coverage**: 89.65% (exceeds the 80% threshold)
- **Mypy Static Typing**: 0 issues across 52 source files
- **Ruff Linter**: Clean (0 warnings or errors)
- **Frontend Build**: `tsc -b && vite build` clean in 2.5s

---

## 📂 Repository Structure

```
SIGN_BRIDGE/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI routers (recognize, words, tts, history, etc.)
│   │   ├── core/            # Configuration, event bus, custom exceptions
│   │   ├── domain/          # Pydantic domain models & gesture definitions
│   │   ├── recognition/     # MediaPipe processors, stability filter, Random Forest
│   │   ├── repositories/    # SQLite session and translation history repositories
│   │   ├── services/        # SignToTextService, TextToSignService, SessionManager
│   │   └── main.py          # FastAPI application factory & lifespan
│   ├── models/              # ASL joblib model, metadata, vocabularies, metrics
│   ├── scripts/             # Performance benchmarks & dataset utilities
│   └── tests/               # Unit, integration, and system test suites
├── frontend/
│   ├── public/              # Static assets & MediaPipe task vision models
│   ├── src/
│   │   ├── components/      # UI panels (spell, words, tts, collect, camera, layout)
│   │   ├── services/        # API client, Camera service, MediaPipe service
│   │   ├── styles/          # CSS design system & dark theme tokens
│   │   ├── types/           # TypeScript API interfaces & landmark definitions
│   │   ├── App.tsx          # Master tab controller & view layout
│   │   └── main.tsx         # React DOM entrypoint
│   ├── package.json
│   └── vite.config.ts
├── docs/                    # SRS, Architecture Design, Model Cards, QA Reports
├── requirements.txt         # Production backend dependencies
├── requirements-dev.txt     # Test & linting dependencies
├── pyproject.toml           # Ruff, Mypy, and Pytest configuration
└── README.md                # Project documentation
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
