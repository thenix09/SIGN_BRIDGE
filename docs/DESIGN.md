# SignBridge System Architecture & Design Document

This document defines the high-level architecture, design decisions, software patterns, sequence workflows, and state machines implemented in **SignBridge**.

---

## 1. System Architecture

SignBridge uses a strictly layered architecture with unidirectional dependency flow:
$$\text{Presentation (API / WebSocket)} \longrightarrow \text{Service Layer} \longrightarrow \text{Domain / Strategies} \longrightarrow \text{Repositories / Persistence}$$

```mermaid
flowchart TD
    Client["Browser Client (React + MediaPipe Tasks Vision)"] -->|JSON over WebSocket / REST| API["API Layer (FastAPI Routers, Schemas, CORS)"]
    API --> Services["Services Layer (SignToText, WordSign, TextToSign, History, Sessions)"]
    Services --> Domain["Domain Layer (HandFrame, LandmarkWindow, Gesture, SignFrame, Sample)"]
    Services --> Recognition["Recognition Subsystem (SignRecognizer, WordRecognizer)"]
    Services --> Rendering["Rendering Subsystem (SignRenderer, LexiconStore, TemplateStore)"]
    Services --> Repositories["Data Layer (Repository[T] -> SQLite)"]
    Services --> EventBus["EventBus (Observer Pattern)"]
```

---

## 2. Design Decisions & Trade-offs

| Decision | Alternatives Considered | Reason for Choice | Trade-off |
| :--- | :--- | :--- | :--- |
| **Client-side MediaPipe** | Server-side video streaming / OpenCV | Zero video transmission, privacy by design, reduced server bandwidth and compute | Client device GPU/CPU load |
| **RandomForest for Letters** | 2D CNN / MobileNet on cropped images | Deterministic landmark vector input (63 floats), fast CPU inference (<10 ms), small footprint | Cannot easily model raw pixel features |
| **SQLite with SQLAlchemy** | PostgreSQL / MongoDB | Zero-config standalone deployment, single-worker execution, portable | Concurrency limited to single process / file locks |
| **1D CNN + BiGRU for Words** | Transformer / Heavy 3D CNN | Sequence of 32 frames with 104 features runs efficiently in ONNX runtime | Moderate vocabulary capacity (30–50 words) |
| **Closed Vocabulary (30 words)**| Open vocabulary translation | High accuracy on isolated sign recognition within 7-day timeline | Not continuous sentence translation |
| **Nearest-Wrist Slot Assignment** | Handedness label classification | Immune to camera mirror confusion and user handedness flips | Needs clear pose tracking |
| **Dominant-Hand Canonicalisation** | Separate models for L/R hands | Doubles effective training data and simplifies single model | Requires dominant hand detection |
| **No Face Mesh (Pose only)** | 468-point Face Mesh | Avoids heavy compute; 3 key points (nose, mouth corners) capture facial relative height | Lacks fine facial expression grammar |
| **Single Handedness Rule (Letters)** | Automatic mirroring heuristic | Consistent server-side canonicalisation using majority dataset label | Requires client to send MediaPipe label |

---

## 3. Design Patterns Applied

| Pattern | Location | Purpose |
| :--- | :--- | :--- |
| **Abstraction** | `SignRecognizer`, `WordRecognizer`, `SignRenderer`, `Repository[T]` | Decouples high-level service logic from specific model frameworks or storage backends. |
| **Inheritance** | `LandmarkRecognizer` $\to$ `ASLRecognizer`, `ISLRecognizer`; `OnnxWordRecognizer` $\to$ `ASLWordRecognizer` | Extends base normalization and inference routines for language-specific specializations. |
| **Polymorphism** | `FakeRecognizer`, `FakeWordRecognizer`, `FakeSignRenderer` | Allows instant substitution of fakes in unit and integration test suites without disk dependencies. |
| **Encapsulation** | `HandFrame`, `LandmarkWindow`, `StabilityFilter`, `SignSegmenter`, `WordBuilder` | Domain objects enforce strict coordinate validations and immutable post-initialization state. |
| **Strategy & Factory** | `RecognizerFactory`, `RendererFactory` | Dynamically instantiates the appropriate recognition or rendering algorithm based on mode and language code. |
| **Composite** | `FallbackSignRenderer` | Combines `WordSequenceRenderer` and `LandmarkTemplateRenderer` into a unified playback pipeline. |
| **Observer** | `EventBus` | Decouples recognition services from audit persistence in `HistoryService`. |
| **Singleton** | `Settings` via `get_settings()` with `@lru_cache` | Centralized, environment-overridable configuration singleton. |
| **Repository** | `SQLiteTranslationRepository`, `SQLiteSampleRepository` | Encapsulates database queries behind `Repository[T]` abstraction. |
| **State** | `StabilityFilter` (`RecognizerState`), `SignSegmenter` (`SegmenterState`) | Stateful lifecycle transitions for multi-frame letter confirmation and gesture boundary detection. |
| **Dependency Injection** | `backend/app/api/deps.py` | FastAPI `Depends()` injects singletons and services into route handlers. |

---

## 4. Class Architecture

```mermaid
classDiagram
    class SignRecognizer {
        <<abstract>>
        +language: str
        +load() void
        +recognize(frame: HandFrame) Gesture
    }
    class WordRecognizer {
        <<abstract>>
        +language: str
        +vocabulary: tuple[str]
        +load() void
        +recognize(window: LandmarkWindow, top_k: int) list[Gesture]
    }
    class LandmarkRecognizer {
        +model_path: Path
        +threshold: float
        +recognize(frame: HandFrame) Gesture
    }
    class ASLRecognizer {
        +language: str = "ASL"
    }
    class OnnxWordRecognizer {
        +model_path: Path
        +vocab_path: Path
        +recognize(window: LandmarkWindow, top_k: int) list[Gesture]
    }
    class ASLWordRecognizer {
        +language: str = "ASL"
    }
    class SignRenderer {
        <<abstract>>
        +render(text: str) list[SignFrame]
    }
    class LandmarkTemplateRenderer {
        +template_store: TemplateStore
        +render(text: str) list[SignFrame]
    }
    class WordSequenceRenderer {
        +lexicon_store: LexiconStore
        +render(text: str) list[SignFrame]
    }
    class FallbackSignRenderer {
        +word_renderer: WordSequenceRenderer
        +template_renderer: LandmarkTemplateRenderer
        +render_with_metadata(text: str) tuple
    }

    SignRecognizer <|-- LandmarkRecognizer
    LandmarkRecognizer <|-- ASLRecognizer
    WordRecognizer <|-- OnnxWordRecognizer
    OnnxWordRecognizer <|-- ASLWordRecognizer
    SignRenderer <|-- LandmarkTemplateRenderer
    SignRenderer <|-- WordSequenceRenderer
    SignRenderer <|-- FallbackSignRenderer
```

---

## 5. Sequence Workflows

### 5.1 Sign-to-Text (Fingerspelling Pipeline)
```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Browser as Browser (MediaPipe)
    participant WS as /ws/recognize
    participant Service as SignToTextService
    participant Recognizer as ASLRecognizer
    participant Filter as StabilityFilter
    participant Builder as WordBuilder
    participant Engine as SuggestionEngine
    participant Bus as EventBus

    User->>Browser: Perform letter hand sign
    Browser->>WS: Stream frame [session_id, landmarks, handedness]
    WS->>Service: process_frame(...)
    Service->>Service: canonicalise_landmarks() -> HandFrame
    Service->>Recognizer: recognize(frame) -> Gesture
    Service->>Filter: update(gesture)
    alt 8 consecutive frames confirmed
        Filter-->>Service: (CONFIRMED, letter)
        Service->>Builder: add_letter(letter)
        Service->>Bus: publish("letter_confirmed", letter)
    else Still stabilising
        Filter-->>Service: (STABILISING, None)
    end
    Service->>Engine: suggest(current_word)
    Service-->>WS: RecognitionResult
    WS-->>Browser: JSON [label, confidence, state, current_word, suggestions]
```

### 5.2 Word Recognition Sequence Workflow
```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Browser as Browser (MediaPipe)
    participant WS as /ws/recognize-words
    participant Service as WordSignService
    participant Segmenter as SignSegmenter
    participant Model as ASLWordRecognizer
    participant Builder as WordBuilder
    participant Bus as EventBus

    User->>Browser: Perform sign gesture (pause, sign, pause)
    Browser->>WS: Stream frame [pose, hands]
    WS->>Service: process_frame(...)
    Service->>Segmenter: update(frame)
    alt Sign in motion
        Segmenter-->>Service: (SIGNING, None)
        Service-->>WS: WordResult(state="SIGNING")
    else Sign completed (idle threshold reached)
        Segmenter-->>Service: (COMPLETE, LandmarkWindow)
        Service->>Model: recognize(window, top_k=3)
        Model-->>Service: [top1, top2, top3]
        alt top1.conf >= 0.50 AND (top1 - top2) >= 0.10
            Service->>Builder: commit_word(top1.label)
            Service->>Bus: publish("word_recognised", top1.label)
            Service-->>WS: WordResult(state="COMPLETE", accepted_word=top1.label)
        else Low confidence or margin
            Service-->>WS: WordResult(state="COMPLETE", accepted_word=None, message="No confident match")
        end
    end
    WS-->>Browser: JSON [state, top_k, accepted_word, sentence, message]
```

### 5.3 Text-to-Sign Sequence Workflow
```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as POST /api/text-to-sign
    participant Service as TextToSignService
    participant Renderer as FallbackSignRenderer
    participant WordR as WordSequenceRenderer
    participant LetterR as LandmarkTemplateRenderer
    participant Bus as EventBus

    Client->>API: {"language": "ASL", "text": "Thank you cat"}
    API->>Service: convert("ASL", text)
    Service->>Renderer: render_with_metadata(text)
    Renderer->>Renderer: apply_aliases() -> ["thankyou", "cat"]
    alt Word in lexicon ("thankyou")
        Renderer->>WordR: render("thankyou")
        WordR-->>Renderer: 48 sign frames
    else Fallback word ("cat")
        Renderer->>LetterR: render("cat")
        LetterR-->>Renderer: letter frames ['C', 'A', 'T']
    end
    Renderer-->>Service: frames, segments, fallback_tokens=["cat"]
    Service->>Bus: publish("text_converted", {"text": text})
    Service-->>API: ConversionResult
    API-->>Client: JSON [frames, segments, fallback_tokens]
```

---

## 6. State Machine Diagrams

### 6.1 StabilityFilter State Lifecycle
```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> DETECTING: Gesture detected above threshold
    DETECTING --> STABILISING: Same gesture observed again
    DETECTING --> IDLE: No gesture or different label
    STABILISING --> STABILISING: Consecutive frames < 8
    STABILISING --> CONFIRMED: Consecutive frames == 8
    STABILISING --> DETECTING: Gesture changed
    CONFIRMED --> IDLE: Cooldown elapsed (10 frames)
```

### 6.2 SignSegmenter State Lifecycle
```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> SIGNING: Hand present & motion >= 0.01
    IDLE --> IDLE: Hand absent or stationary
    SIGNING --> SIGNING: Continuous motion
    SIGNING --> COMPLETE: (idle_count >= 8 OR len >= 90) AND len >= 12
    SIGNING --> IDLE: (idle_count >= 8) AND len < 12 (discarded)
    COMPLETE --> IDLE: Window emitted exactly once
```

---

## 7. API Changelog

| Date | Author | Version | Notes / Changes |
| :--- | :--- | :--- | :--- |
| **2026-11-07** | Person A | 1.0.0 | Frozen API contract established (Section 12). Scaffold and sample response fixtures committed. |
| **2026-11-09** | Person A | 1.1.0 | Added real-time fingerspelling endpoints (`POST /api/recognize`, `WS /ws/recognize`, `POST /api/session/*`, `GET /api/languages`). |
| **2026-11-09** | Person A | 1.2.0 | Implemented isolated word recognition endpoints (`POST /api/recognize-words`, `WS /ws/recognize-words`, `GET /api/vocabulary`). |
| **2026-11-09** | Person A | 1.3.0 | Implemented text-to-sign rendering endpoint (`POST /api/text-to-sign`). |
| **2026-11-09** | Person A | 1.4.0 | Added audit history endpoints (`GET /api/history`, `DELETE /api/history`) and training sample collection (`POST /api/samples`, `GET /api/samples/stats`, `GET /api/samples/export`). |
