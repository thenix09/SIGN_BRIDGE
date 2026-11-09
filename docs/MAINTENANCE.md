# SignBridge Maintenance & Operations Guide

This guide details maintenance procedures, extension workflows, versioning rules, and operational constraints for the **SignBridge** platform in accordance with the system specification.

---

## 1. Maintenance Categories & Operational Procedures

### 1.1 Corrective Maintenance (Defect Resolution)
- **Model Deserialization / Startup Errors**:
  - *Symptom*: API starts with `model_loaded: false` or returns `503 ModelNotAvailableError`.
  - *Fix*: Verify that `backend/models/asl_model.joblib` or `backend/models/word_model.onnx` exists and matches the expected features ($63$ floats for letters, $32 \times 104$ tensor for words). If running inside Docker, ensure the `COPY backend/models` instruction includes all trained artifacts.
- **WebSocket Disconnection Handling**:
  - *Symptom*: Socket disconnects unexpectedly due to transient network interruptions.
  - *Fix*: The backend preserves session state in memory up to `SIGNBRIDGE_SESSION_TTL_SECONDS` ($1800\text{ s}$ / $30\text{ min}$). Clients should reconnect and continue using the existing `session_id`, or call `POST /api/session/reset` to reinitialize.

### 1.2 Adaptive Maintenance (Environment & Upgrades)
- **Upgrading MediaPipe Vision Tasks**:
  - When updating `@mediapipe/tasks-vision` on the frontend (or updating `hand_landmarker.task` / `vision_wasm`):
    1. Verify that landmark indices remained standard ($0 = \text{wrist}$, $1..4 = \text{thumb}$, $5..8 = \text{index}$, etc.).
    2. Run `python backend/scripts/verify_preprocessing_parity.py` to ensure the normalized coordinates match the server-side `HandFrame` and `LandmarkWindow` implementations.
    3. If MediaPipe changes coordinate normalization or anchor origins, update `canonicalise_landmarks` in `backend/app/domain/hand_frame.py` and retrain the models.
- **Python Runtime Updates**:
  - Maintain compatibility across Python 3.11 and 3.12. Ensure pinned dependency versions in `requirements.txt` and `requirements-dev.txt` build wheel binaries without requiring host C++ compilers.

### 1.3 Perfective Maintenance (Enhancements & Extensions)

#### Procedure A: Adding a New Sign Language (e.g. Indian Sign Language - ISL)
1. **Define the Recognizer**:
   Create a subclass of `LandmarkRecognizer` in `backend/app/recognition/isl_recognizer.py`:
   ```python
   from backend.app.recognition.landmark_recognizer import LandmarkRecognizer

   class ISLRecognizer(LandmarkRecognizer):
       @property
       def language(self) -> str:
           return "ISL"
   ```
2. **Register with the Factory**:
   In `backend/app/recognition/isl_recognizer.py`:
   ```python
   from backend.app.recognition.factory import RecognizerFactory
   RecognizerFactory.register("ISL", ISLRecognizer)
   ```
3. **Train and Commit the Model**:
   - Collect ISL landmark samples using `POST /api/samples` or export via `GET /api/samples/export`.
   - Train using `python backend/scripts/train.py --language ISL`.
   - Save the serialized model artifact to `backend/models/isl_model.joblib`.
4. **Update Available Modes**:
   - `GET /api/languages` will automatically discover the registered recognizer and report its availability status.

#### Procedure B: Expanding the Word Vocabulary
1. **Update Vocabulary Specification**:
   Add new vocabulary entries to `backend/data/vocabulary.json`:
   ```json
   {
     "words": [
       {"label": "newword", "category": "general"}
     ],
     "aliases": {
       "new word": "newword"
     }
   }
   ```
2. **Collect Sequence Data**:
   Record multi-frame sign instances with `LandmarkFrame` (pose + hands) and store them in `backend/data/processed/word_dataset.npz`.
3. **Retrain the Word Model**:
   ```bash
   python backend/scripts/train_word_model.py
   python backend/scripts/eval_word_model.py
   python backend/scripts/export_word_model_onnx.py
   ```
4. **Evaluate Accuracy Gate**:
   Run `python backend/scripts/check_gate.py`. For vocabulary sizes up to 30 words, accuracy must be $\ge 0.75$; for 50 words, accuracy must be $\ge 0.70$.
5. **Update Lexicon for Text-to-Sign**:
   Extract medoid sequences for the new words and update `backend/models/lexicon.json`.

### 1.4 Preventive Maintenance (Code Quality & Security)
- **Continuous Integration Checks**:
  - Run `python -m ruff check .` on every pull request to ensure 0 lint errors.
  - Run `python -m mypy backend/app` with strict type annotations.
  - Run `python -m pytest --cov=backend/app --cov-fail-under=80` to preserve $\ge 80\%$ test coverage.
- **Session & Memory Garbage Collection**:
  - `SessionManager.cleanup_expired()` automatically runs on session access to evict inactive sessions older than 30 minutes, preventing memory leaks on single-worker deployments.

---

## 2. Release & Semantic Versioning Protocol

SignBridge adheres to [Semantic Versioning 2.0.0](https://semver.org/):

$$\text{MAJOR}.\text{MINOR}.\text{PATCH}$$

- **MAJOR (e.g. 2.0.0)**: Incompatible API contract changes (e.g. modifications to frozen Section 12 endpoints or breaking changes to WebSocket payload shapes). Requires updating `docs/API_CHANGELOG` in `docs/DESIGN.md`.
- **MINOR (e.g. 1.1.0)**: Backward-compatible functionality additions (e.g. adding a new language like ISL, expanding the closed vocabulary from 30 to 50 words).
- **PATCH (e.g. 1.0.1)**: Backward-compatible bug fixes, performance improvements, or dependency patches.

### Release Checklist
1. All 91 unit and integration tests pass with $\ge 80\%$ coverage.
2. `ruff` and `mypy` pass with 0 errors.
3. `check_gate.py` passes the word gate threshold.
4. Model artifacts (`templates.json`, `letters_meta.json`, `word_vocab.json`, `word_gate.json`, `vocabulary.json`) are committed.
5. Docker container builds cleanly and responds to `GET /api/health`.

---

## 3. Known Limitations & Mitigation

| Limitation | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Motion Letters (J and Z)** | Fingerspelling $J$ and $Z$ involves dynamic motion that static single-frame models cannot fully capture. | The UI displays an explicit guidance banner noting that $J$ and $Z$ have lower reliability than static alphabet letters. |
| **Lighting & Contrast** | Poor or uneven lighting can degrade MediaPipe hand and pose detection confidence. | On-screen user guidance advises even front-facing lighting and contrasting background. |
| **Free-Tier Cloud Cold Starts** | On Render free tier, idle containers spin down and take 30–60 seconds to cold start. SQLite resets on redeploy. | Warm up the service endpoint before live demonstrations; keep a local `docker compose up` instance as a redundant backup. |
| **Single-Worker In-Memory State** | Sessions are stored in process memory on the single Uvicorn worker. | Multiple workers would split session state; production scaling requires Redis-backed session storage if scaling beyond 1 worker. |
