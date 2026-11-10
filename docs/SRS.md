# SignBridge — Software Requirements Specification (SRS)

**Document Version:** 1.0.0-draft
**Date:** November 9, 2026
**Author:** Person C (Frontend, Testing & Documentation)
**Project:** SignBridge — American Sign Language (ASL) Recognition & Translation System
**Timeline:** 7-Day Intensive Build Plan (November 7–14, 2026)

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) document defines the system requirements, architectural boundaries, functional specifications, non-functional constraints, and interface contracts for **SignBridge**. It serves as the single source of truth for development, integration, and verification across the three-person engineering team.

### 1.2 Project Overview
SignBridge is a web-based, real-time sign language recognition and bidirectional communication platform. The system leverages browser-based computer vision (MediaPipe) for client-side hand tracking, machine learning models for gesture classification (isolated letters and words), and an animation renderer for text-to-sign translation.

### 1.3 Intended Audience
* **Development Team:**
  * **Person A:** Backend, Platform, APIs, and Deployment.
  * **Person B:** Data, Machine Learning, Feature Extraction, and Recognition Models.
  * **Person C:** Frontend Architecture, UI/UX, Integration Testing, and Project Documentation.
* **Evaluators & Stakeholders:** Course instructors, accessibility advocates, and testing participants evaluating project deliverables.

### 1.4 Definitions and Abbreviations
* **ASL:** American Sign Language.
* **Landmark:** A normalized 3D point $(x, y, z)$ identified on a human hand by MediaPipe.
* **HandLandmarker:** The Google MediaPipe Tasks Vision pipeline extracting 21 spatial landmarks per hand.
* **Handedness:** Classification of a detected hand as either "Left" or "Right" with a confidence score.
* **Fingerspelling (Spell Mode):** Recognition of static ASL manual alphabet letters (A–Z).
* **Word Signs (Words Mode):** Recognition of dynamic sign sequences representing common words.
* **Text-to-Sign:** Translation of written English text into animated or sequenced sign demonstrations.
* **WASM:** WebAssembly, used for client-side model execution.
* **SRS:** Software Requirements Specification.
* **TBD:** To Be Determined (subject to official Build Specification v2 confirmation).

---

## 2. Overall Description

### 2.1 Product Perspective
SignBridge is an independent, self-contained web application operating across two primary tiers:
1. **Frontend Client (`frontend/`):** React 19, TypeScript, and Vite running in modern web browsers, executing MediaPipe hand landmark detection locally on the user's hardware via WebAssembly/WebGL.
2. **Backend Server (`backend/`):** Python-based service providing ML inference pipelines, sequence classification, template rendering, and WebSocket streams for real-time sign recognition, deployable via Docker and Render.

```
┌────────────────────────────────────────────────────────┐
│                   Client Browser                       │
│  ┌───────────────────────┐   ┌──────────────────────┐  │
│  │   Webcam Video Feed   │──►│   MediaPipe Vision   │  │
│  └───────────────────────┘   │  (21 Hand Landmarks) │  │
│                              └──────────┬───────────┘  │
│                                         │              │
│  ┌───────────────────────┐              │              │
│  │  Skeleton Overlay &   │◄─────────────┘              │
│  │   UI Visualizations   │                             │
│  └───────────────────────┘                             │
└────────────────┬───────────────────────────────────────┘
                 │ WebSockets / HTTP
                 ▼
┌────────────────────────────────────────────────────────┐
│                SignBridge Backend Server               │
│  ┌───────────────────────┐   ┌──────────────────────┐  │
│  │  Alphabet Recognizer  │   │   Word Recognizer    │  │
│  │  (Landmark Classifier)│   │ (Temporal Sequences) │  │
│  └───────────────────────┘   └──────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │    Text-to-Sign Engine & Lexicon Store           │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### 2.2 Product Objectives
* Provide accessible, real-time sign recognition directly from standard consumer webcams without dedicated hardware gloves or depth sensors.
* Support bidirectional communication: Sign-to-Text (recognition) and Text-to-Sign (visual demonstration).
* Maintain strict separation of concerns among platform, ML models, and presentation layers.

### 2.3 Main System Features
* **Live Camera Viewport:** Mirrored video feed with dynamic status and resolution reporting.
* **Dual-Hand Landmark Tracking:** Extraction of 21 3D landmarks for both left and right hands.
* **Skeleton Overlay:** Accurate, hardware-aligned visual overlay of joints and bones.
* **Fingerspelling Translation:** Real-time character-by-character letter recognition.
* **Word Sign Recognition:** Temporal gesture segmentation and classification for everyday vocabulary.
* **Text-to-Sign Player:** Visual playback of signs corresponding to user-provided text.
* **Team Dataset Collector (`collect.html` / Collect Tab):** Structured interface for recording landmark sequences to train and calibrate ML models.

### 2.4 Stakeholders and Their Roles
* **Person A (Backend & Platform):** Scaffolding, settings, frozen API contract, WebSocket servers, session management, deployment on Render with HTTPS.
* **Person B (Data & Machine Learning):** Kaggle dataset processing (ASL Alphabet, asl-signs), model architectures, training pipelines, ONNX export, and evaluation gates.
* **Person C (Frontend & Testing):** Frontend component hierarchy, camera lifecycle, MediaPipe WASM integration, responsive UI, integration testing, and documentation.
* **Target Users:** Deaf and Hard-of-Hearing individuals, ASL students, educators, and hearing communication partners.

### 2.5 Intended Users
* **Primary Users:** Individuals communicating via ASL seeking immediate digital transcription.
* **Secondary Users:** Non-signers learning ASL vocabulary or reading sign translations.
* **Administrative / Team Users:** Developers and testers recording calibrated sign sequences.

### 2.6 Operating Environment
* **Client Hardware:** Standard laptop or desktop computer equipped with an integrated or USB webcam (minimum $640 \times 480$ at 15 fps).
* **Client Software:** Modern evergreen browser supporting WebAssembly, WebGL, and the MediaDevices API (Google Chrome, Apple Safari, Mozilla Firefox, Microsoft Edge).
* **Server Environment:** Docker container running Python on Linux (Render cloud hosting with HTTPS).

### 2.7 Assumptions and Dependencies
* The client browser has access to camera hardware with user permission granted.
* MediaPipe WebAssembly and model assets can be served locally or cached by the browser.
* Network connectivity is available for WebSocket communication with the backend service.

### 2.8 Constraints
* **Development Timeline:** Strict 7-day delivery window with feature freeze on Day 6.
* **Ownership Isolation:** Team members modify only their allocated directories (`frontend/`, `backend/`, `docs/`).
* **Frozen API Contract:** Client-server contracts are locked on Day 1; no unilateral endpoint additions.
* **Hardware Independence:** Must run on standard consumer laptops without requiring dedicated external GPUs.

---

## 3. System Scope

### 3.1 In-Scope Functionality
* **Webcam Capture:** Camera discovery, user permission handling, lifecycle start/stop, and error recovery.
* **Hand Landmark Extraction:** Up to 2 hands tracked simultaneously, producing 21 normalized $(x, y, z)$ coordinates per hand.
* **Fingerspelling Mode (Spell):** Recognition of ASL alphabet letters.
* **Word Sign Mode (Words):** Recognition of a curated baseline vocabulary (30 words targeted, minimum 20 words per gate).
* **Text-to-Sign Demonstration:** Visual playback of sign animations/templates with speed controls.
* **Dataset Collection Interface:** Standardized capture tool for recording 50 samples per letter for model training.
* **Developer Diagnostics:** Live frame rate display, detection status indicator, and JSON landmark export.

### 3.2 Out-of-Scope (Explicitly Excluded / Stretch Features)
* **Conversation Mode:** Two-way continuous automated dialogue is classified strictly as a stretch feature and is not part of the core 7-day milestone.
* **Microphone / Speech Input:** Audio speech-to-text input is excluded from the core release.
* **Extended Vocabulary (>50 Words):** The baseline vocabulary is constrained to $\le 30$ verified signs to ensure accuracy and real-time inference.
* **Mobile Native Applications:** Only responsive web browsers are officially targeted.

### 3.3 Current Implementation Status (Day 1 Milestone)
* **Completed & Verified:**
  * Frontend skeleton and responsive layout.
  * Camera service with full error handling (`NotAllowedError`, `NotFoundError`, `NotReadableError`).
  * Live camera feed with track termination on stop and unmount.
  * Local MediaPipe HandLandmarker model (`hand_landmarker.task`) and local WASM execution.
  * Real-time dual-hand landmark detection and mirrored canvas skeleton overlay.
* **Pending Subsequent Days:**
  * Alphabet letter classification model (Person B, Days 2–3).
  * Word sequence model and ONNX integration (Person B, Days 3–4).
  * Real backend API and WebSocket routes (Person A, Days 3–4).
  * Full integration and live testing sessions (Person C, Days 4–5).

---

## 4. Functional Requirements

> [!IMPORTANT]
> The official **SignBridge Build Specification v2** defines the authoritative requirements FR-1 through FR-12. To prevent fabricated or divergent requirements, the requirements below reflect the structure and scope mandated by the project specification. Where full text from Build Specification v2 is pending team retrieval, the requirement is explicitly marked `[Pending Build Specification v2 Text]`.

### FR-1: Video Stream Acquisition & Lifecycle
* **Description:** The system shall capture live video from the user's default video input device via the browser MediaDevices API at a minimum resolution of $640 \times 480$.
* **Acceptance Condition:** The user can grant camera access, observe the live video feed, and cleanly terminate hardware capture using a dedicated "Stop Camera" control.
* **Implementation Status:** Implemented & Verified in Milestone 2 (`CameraService`, `CameraFeed`).

### FR-2: Client-side Hand Landmark Detection
* **Description:** The system shall continuously process active video frames using Google MediaPipe Hand Landmarker to extract 21 3D spatial landmarks per detected hand.
* **Acceptance Condition:** Up to two hands (left and right) are detected simultaneously in real time without causing main-thread UI freezing.
* **Implementation Status:** Implemented & Verified in Milestone 3 (`MediaPipeService`, `LandmarkCanvas`).

### FR-3: Skeleton Visualization & Mirrored Overlay
* **Description:** The system shall render visual landmarks and inter-joint skeletal connections directly over the live camera feed on a transparent canvas layer.
* **Acceptance Condition:** The overlay coordinates must accurately track the physical hands on screen, accounting for mirrored camera rendering, with distinct color coding for left and right hands.
* **Implementation Status:** Implemented & Verified in Milestone 3 (`LandmarkCanvas`).

### FR-4: Handedness Classification & Confidence Scoring
* **Description:** The system shall determine the handedness (Left or Right) and detection confidence for each detected hand and present this information to the user.
* **Acceptance Condition:** Each detected hand displays a readable status tag indicating handedness and confidence percentage.
* **Implementation Status:** Implemented & Verified in Milestone 3 (`LandmarkCanvas`).

### FR-5: Static Letter Recognition (Fingerspelling Mode)
* **Description:** `[Pending Build Specification v2 Text]`
  * *Scope:* Classification of single-frame hand landmark configurations into corresponding ASL manual alphabet letters (A–Z).
  * *Planned Owner:* Person B (model), Person A (backend route/service), Person C (UI panel).
  * *Target Milestone:* Day 2–3.

### FR-6: Temporal Gesture Segmentation
* **Description:** `[Pending Build Specification v2 Text]`
  * *Scope:* Detection of gesture initiation, stroke phase, and termination across continuous frame windows to isolate distinct sign instances.
  * *Planned Owner:* Person B (`SignSegmenter`).
  * *Target Milestone:* Day 3.

### FR-7: Word Sign Recognition (Words Mode)
* **Description:** `[Pending Build Specification v2 Text]`
  * *Scope:* Temporal sequence classification of landmark windows into recognized ASL words from the baseline vocabulary (target 30 words, minimum gate 20 words).
  * *Planned Owner:* Person B (`train_word_model.py`), Person A (`WordSignService`).
  * *Target Milestone:* Day 3–4.

### FR-8: Gesture Stability & Filtering
* **Description:** `[Pending Build Specification v2 Text]`
  * *Scope:* Debouncing and stability filtering to eliminate spurious transitions and noise between consecutive sign predictions.
  * *Planned Owner:* Person B (`StabilityFilter`), Person C (UI smoothing).
  * *Target Milestone:* Day 2–3.

### FR-9: Text-to-Sign Demonstration & Playback
* **Description:** `[Pending Build Specification v2 Text]`
  * *Scope:* Rendering visual demonstrations of ASL signs corresponding to user-entered text using landmark templates or video references.
  * *Planned Owner:* Person B (`LandmarkTemplateRenderer`), Person A (`TextToSignService`), Person C (`TextToSignPlayer`).
  * *Target Milestone:* Day 4.

### FR-10: Dataset Collection Interface (`collect.html`)
* **Description:** `[Pending Build Specification v2 Text]`
  * *Scope:* Dedicated interface enabling team members to record calibrated 50-sample sets per letter and export structured JSON files for model training.
  * *Planned Owner:* Person C (Frontend Collect page), Person B (data ingestion).
  * *Target Milestone:* Day 2 (built first for team recording session).

### FR-11: Developer Diagnostics & Parity Verification
* **Description:** `[Pending Build Specification v2 Text]`
  * *Scope:* Developer tools panel displaying measured frame rate (fps), detection latency, and a "Save last window as JSON" utility for browser-model parity checks.
  * *Planned Owner:* Person C (`frontend/`), Person B (`check_browser_parity.py`).
  * *Target Milestone:* Day 3.

### FR-12: System Health & Backend Synchronization
* **Description:** `[Pending Build Specification v2 Text]`
  * *Scope:* Continuous status monitoring between client and backend via `/api/health`, vocabulary synchronization via `/api/vocabulary`, and graceful fallback to offline/stub operation.
  * *Planned Owner:* Person A (`api/health`), Person C (`api.client.ts`).
  * *Target Milestone:* Day 1 (stubbed), Day 4 (real backend).

---

## 5. Non-Functional Requirements

### 5.1 Performance
* **NFR-1 (Camera Frame Rate):** The client video capture and landmark extraction pipeline shall maintain an operational frame rate of $\ge 15$ frames per second (targeting 30 fps) on standard consumer laptops.
* **NFR-2 (Inference Concurrency):** MediaPipe landmark detection shall utilize a non-blocking frame loop (`requestAnimationFrame`) with concurrency locks to prevent frame backlog and memory leaks.
* **NFR-3 (Startup Latency):** MediaPipe WASM and model asset initialization shall complete within 3 seconds under local execution.

### 5.2 Usability & Accessibility
* **NFR-4 (High Contrast Compliance):** UI color schemes shall adhere to WCAG 2.1 AA standards for text and interactive component contrast.
* **NFR-5 (Mirrored Interaction):** The live video preview and skeleton overlay shall be mirrored horizontally (`scaleX(-1)` visual alignment) so that physical movement corresponds naturally to screen display.
* **NFR-6 (Clear Status Communication):** All operational states (Camera Off, Requesting Permission, Live Tracking, No Hands, Error) shall be communicated via distinct visual badges and icons.

### 5.3 Reliability & Fault Tolerance
* **NFR-7 (Camera Error Recovery):** The system shall intercept and gracefully handle browser `NotAllowedError`, `NotFoundError`, and `NotReadableError` DOMExceptions with actionable resolution instructions.
* **NFR-8 (Hardware Acceleration Fallback):** If WebGL/GPU delegation fails on the client device, the MediaPipe pipeline shall automatically fallback to CPU execution without crashing.
* **NFR-9 (Resource Disposal):** All camera tracks and WebGL/WASM contexts shall be completely terminated upon user deactivation or component unmount.

### 5.4 Security & Privacy
* **NFR-10 (Client-side Privacy):** Camera video stream frames shall be processed exclusively within the local browser runtime memory for landmark extraction; raw video streams shall never be transmitted or stored on remote servers without explicit user initiation.
* **NFR-11 (Transport Security):** All external client-server communications shall use HTTPS and WSS (secure WebSockets) in deployed environments.

### 5.5 Maintainability & Modular Architecture
* **NFR-12 (Strict Decoupling):** Frontend presentation components shall remain decoupled from camera hardware drivers, MediaPipe inference logic, and network API transports.
* **NFR-13 (Frozen API Compliance):** Network communication shall conform to the frozen API contract maintained in `docs/DESIGN.md`.

### 5.6 Compatibility
* **NFR-14 (Browser Compatibility):** The application shall execute reliably on Chrome (v110+), Safari (v16+), Edge (v110+), and Firefox (v115+) on macOS, Windows, and Linux.

### 5.7 Availability
* **NFR-15 (Stub Resilience):** The frontend shall remain fully operational using local stub fixtures when the backend service is offline or in development.

---

## 6. External Interface Requirements

### 6.1 User Interface
The SignBridge interface utilizes a responsive 2-column layout:
* **Primary Column:** Camera viewport card with embedded video element, transparent canvas overlay, live status pills, and resolution indicator.
* **Secondary Column:** System controls sidebar, mode indicators, start/stop buttons, and diagnostics.
* **Navigation Bar:** Tab controls allowing instant switching between Fingerspelling (Spell), Word Signs (Words), and Dataset Collection.

### 6.2 Hardware Interface (Webcam)
* **Standard:** W3C Media Capture and Streams API (`navigator.mediaDevices.getUserMedia`).
* **Default Constraints:**
  ```json
  {
    "video": {
      "width": { "ideal": 640 },
      "height": { "ideal": 480 },
      "facingMode": "user"
    },
    "audio": false
  }
  ```

### 6.3 Software Interfaces
* **Google MediaPipe Tasks Vision (`@mediapipe/tasks-vision`):** Browser-based WebAssembly and WebGL delegate for 21-point hand landmark identification.
* **Local Asset Store:**
  * Model Asset: `/models/hand_landmarker.task`
  * WASM Runtime: `/wasm/vision_wasm_internal.wasm`

### 6.4 Communications & API Interface (Person A Contract)
The frontend communicates with Person A's backend using the following frozen endpoints:

| Endpoint | Protocol | Purpose | Client Implementation |
| :--- | :--- | :--- | :--- |
| `/api/health` | HTTP GET | Backend health check and version verification | Stubbed in `api.client.ts` |
| `/api/vocabulary` | HTTP GET | Retrieve active letter and word recognition vocabulary | Stubbed in `api.client.ts` |
| `/api/languages` | HTTP GET | Check supported sign languages status | Stubbed in `api.client.ts` |
| `/api/text-to-sign` | HTTP POST | Request sign playback sequences for text strings | Stubbed in `api.client.ts` |
| `/ws/letters` | WebSocket | Real-time streaming recognition for alphabet fingerspelling | Scheduled for Day 3 |
| `/ws/words` | WebSocket | Real-time streaming recognition for word signs | Scheduled for Day 4 |

---

## 7. Use Cases

### UC-1: Real-time Fingerspelling Recognition (Spell Mode)
* **Use Case ID:** UC-1
* **Actor:** Sign Language Communicator (Fingerspelling)
* **Preconditions:** Camera is connected, browser permissions are granted, application is in "Spell" mode.
* **Main Flow:**
  1. User positions hand in front of the camera viewport.
  2. System detects hand landmarks and renders skeleton overlay in real time.
  3. Landmark coordinates are packaged and classified against the alphabet model.
  4. Recognized letter is appended to the current active word buffer.
* **Expected Outcome:** User sees immediate visual feedback of their hand posture and the transcribed letter.

### UC-2: Isolated Word Sign Recognition (Words Mode)
* **Use Case ID:** UC-2
* **Actor:** Sign Language Communicator (Signing Words)
* **Preconditions:** Camera is active, application is in "Words" mode.
* **Main Flow:**
  1. User performs an ASL sign from the supported vocabulary within the camera viewport.
  2. System segments the temporal gesture window and extracts sequence features.
  3. Classifier evaluates sequence against the trained word model.
  4. System displays the top-1 predicted word with confidence score.
* **Expected Outcome:** The performed sign is accurately identified and displayed in the translation history.

### UC-3: Text-to-Sign Demonstration Playback
* **Use Case ID:** UC-3
* **Actor:** ASL Learner / Hearing Communicator
* **Preconditions:** Application is loaded; user accesses Text-to-Sign panel.
* **Main Flow:**
  1. User enters an English phrase or word into the input field.
  2. System checks vocabulary lexicon for available sign sequences.
  3. Player animates landmark demonstrations or reference sign templates.
  4. User adjusts playback speed or replays the demonstration.
* **Expected Outcome:** Clear visual animation demonstrating how to produce the requested sign.

### UC-4: Team Dataset Collection
* **Use Case ID:** UC-4
* **Actor:** Team Member / Calibrated Signer (Person A, B, or C)
* **Preconditions:** User navigates to `collect.html` (or Collect tab).
* **Main Flow:**
  1. User selects target sign letter or word.
  2. System initiates countdown and records landmark frames across repeated trials (50 samples per letter).
  3. User reviews sample count and saves recording as structured JSON.
* **Expected Outcome:** Validated, clean landmark sequences formatted for model training ingestion.

---

## 8. Limitations and Future Scope

### 8.1 Current Implementation Limitations
* **Day 1 Milestone:** Hand detection and skeleton rendering are operational; sign classification models and real-time backend synchronization are not yet connected.
* **Lighting and Occlusion Sensitivity:** Tracking reliability decreases under extreme backlight, low lighting, or when one hand fully occludes the other.
* **Framing Constraint:** Hands must remain within the camera viewport boundaries ($640 \times 480$) to maintain active landmark tracking.

### 8.2 Planned Features (Days 2–6)
* Static alphabet recognition integration (Day 2–3).
* Team dataset recording and model retraining (Day 3).
* Word sequence recognition and ONNX runtime integration (Day 3–4).
* Real backend API and WebSocket cutover (Day 4).
* Joint end-to-end integration and live accuracy benchmarking (Day 5).

### 8.3 Explicitly Excluded Features
* Two-way automated conversation mode.
* Continuous full-sentence grammatical ASL parsing.
* Speech-to-text / microphone input.

### 8.4 Future Enhancements
* Vocabulary expansion beyond the baseline 30 words.
* Full-body pose and facial expression landmark integration for grammatical sign nuances.
* Offline Progressive Web App (PWA) installation.

---

## 9. Verification & Traceability Matrix

| Requirement ID | Specification Source | Responsible Owner | Implementation State | Verification Method |
| :--- | :--- | :--- | :--- | :--- |
| **FR-1** | Build Specification v2 (B10) | Person C | Implemented | Manual browser test (Start/Stop camera) |
| **FR-2** | Build Specification v2 (B10) | Person C | Implemented | Live dual-hand MediaPipe tracking |
| **FR-3** | Build Specification v2 (B10) | Person C | Implemented | Mirrored canvas overlay check |
| **FR-4** | Build Specification v2 (B10) | Person C | Implemented | Handedness pill rendering |
| **FR-5** | Build Specification v2 (B2, B4) | Person B & A | Planned (Day 2–3) | Unit & integration tests |
| **FR-6** | Build Specification v2 (B2, B7) | Person B | Planned (Day 3) | Segmenter unit tests |
| **FR-7** | Build Specification v2 (B7, B8) | Person B & A | Planned (Day 3–4) | Word model benchmark |
| **FR-8** | Build Specification v2 (B4) | Person B | Planned (Day 2–3) | Stability unit tests |
| **FR-9** | Build Specification v2 (B8, B9) | Person A & B | Planned (Day 4) | Template playback test |
| **FR-10** | Build Specification v2 (B10) | Person C | Planned (Day 2) | 50-sample recording session |
| **FR-11** | Build Specification v2 (B10) | Person C | Planned (Day 3) | Parity capture JSON test |
| **FR-12** | Build Specification v2 (B1, B6) | Person A | In Progress (Stubbed) | Health check & WebSocket test |

---

*SignBridge Specification — Authored by Person C for Team Review.*
