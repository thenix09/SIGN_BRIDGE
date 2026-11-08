# SignBridge Model Card & Dataset Documentation

**Document Owner:** Person B (Data & Machine Learning)  
**Specification Reference:** Build Specification v2 (§2, §14, §19)  
**Last Updated:** November 7, 2026  

---

## 1. Overview & System Purpose

SignBridge is a lightweight, privacy-preserving two-way American Sign Language (ASL) educational bridge designed for real-time browser interaction:
1. **Sign to Text (Spell Mode):** Recognizes static ASL fingerspelled letters (A–Z) from single-hand MediaPipe landmarks.
2. **Sign to Text (Words Mode):** Recognizes isolated ASL signs from a curated closed vocabulary of 30 common words using upper-body landmark sequences.
3. **Text to Sign:** Synthesizes sign animations using dictionary playback of recorded sign movements for vocabulary words and fingerspelled fallbacks for out-of-vocabulary terms.

---

## 2. Dataset Sources, Licences & Attribution

### 2.1 Words Mode: Kaggle Isolated Sign Language Recognition (`asl-signs`)
- **Dataset Source:** Google "Isolated Sign Language Recognition" Competition ([Kaggle `asl-signs`](https://www.kaggle.com/competitions/asl-signs)), derived from the PopSign ASL project.
- **Underlying Corpus:** PopSign ASL, developed by Georgia Institute of Technology and Google.
- **Licence:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Access Date:** November 7, 2026.
- **Attribution Notice:**
  > PopSign ASL Dataset © 2022 Georgia Institute of Technology and Google LLC. Licensed under CC BY 4.0.
- **Redistribution Restrictions:**
  - Raw Parquet landmark sequence files from Kaggle must not be redistributed, republished, or committed to public version control.
  - The repository enforces this via `.gitignore` (`backend/data/raw/`).
  - Only derivative, aggregated artifacts (preprocessed compact tensors `data/word_dataset.npz`, trained model weights `models/word_model.onnx`, and representative playback medoids `models/lexicon.json`) may be distributed.
- **Competition Rule Compliance:**
  - Used strictly for non-commercial educational and research evaluation.
  - Selected subset: 11,316 sequences representing 30 distinct vocabulary words from 21 signers.

### 2.2 Spell Mode: ASL Alphabet Dataset & Team Live Recordings
- **Primary Source:** Kaggle "ASL Alphabet" dataset (~87,000 synthetic and captured images for static hand shapes A–Z).
- **Access Date:** November 7, 2026.
- **Domain Adaptation / Team Recordings:**
  - 50 live samples per letter recorded per team member using `collect.html` (Day 3 joint recording session).
  - Used as an uncorrupted live hold-out set to bridge the synthetic-to-webcam domain gap.

---

## 3. Privacy by Design & Landmark Constraints

1. **Zero Video / Image Storage or Transmission:**
   - Raw video frames never leave the client's web browser.
   - Feature extraction (hands and pose) is performed entirely on-device in JavaScript using MediaPipe Tasks Vision.
   - Only numeric landmark coordinates ($x, y, z$) are transmitted over WebSocket or REST.
2. **Facial Privacy & Upper-Body Landmark Selection:**
   - The raw Kaggle `asl-signs` dataset contains 468 facial mesh coordinates per frame.
   - **In accordance with Build Specification v2 (§2, §7), all 468 facial mesh points are discarded during preprocessing.**
   - Skeleton playback has no facial expressions (an acknowledged ASL limitation).
   - Only the **9 specified upper-body pose landmarks** (`POSE_INDICES = (0, 9, 10, 11, 12, 13, 14, 15, 16)`) and **both hands** (21 landmarks each) are preserved:
     - **Head / Face anchors (3 pose points):** Nose (index 0), left mouth corner (index 9), right mouth corner (index 10).
     - **Torso & Arms (6 pose points):** Left shoulder (index 11), right shoulder (index 12), left elbow (index 13), right elbow (index 14), left wrist (index 15), right wrist (index 16).
     - **Hands (42 points total):** Left hand (21 landmarks, indices 0..20) and Right hand (21 landmarks, indices 0..20).
   - **Feature Vector:** Each frame is encoded into exactly **104 values**:
     - 18 pose coordinates ($9 \text{ points} \times [x, y]$)
     - 42 left hand coordinates ($21 \text{ points} \times [x, y]$)
     - 42 right hand coordinates ($21 \text{ points} \times [x, y]$)
     - 2 hand presence binary flags ($1.0$ if detected, $0.0$ if missing/zero-filled)
     - Coordinate dimensions use $x$ and $y$ only; $z$ depth is discarded as noisy.

---

## 4. Intended Use & Honesty Disclaimers

### 4.1 Intended Use
- Educational demonstration of machine learning applied to assistive accessibility technology.
- Learning tool for students and developers practicing basic isolated ASL vocabulary and fingerspelling.

### 4.2 Out-of-Scope & Prohibited Use Cases
- **Not a Translation System:** Words mode recognizes isolated signs from a closed vocabulary of 30 words. It has no continuous sentence translation model, temporal grammar parser, or facial expression grammar.
- **Medical & Legal Prohibition:** SignBridge **MUST NOT** be deployed in healthcare, legal, emergency, safety-critical, or high-stakes communication environments.
- **No Replacement for Certified Interpreters:** This project is an assistive prototype and does not replace qualified, certified Deaf or ASL interpreters.
- **Motion Letter Limitations (J & Z):** ASL fingerspelling for letters J and Z requires dynamic motion. Spell mode evaluates only static hand frames; recognition for J and Z is therefore marked as low-reliability.
- **Text-to-Sign Playback Disclaimer:** Text-to-sign playback is dictionary playback of recorded kinematic movements, not natural language generation.

---

## 5. Model Architecture & Pipeline Specifications

### 5.1 Spell Mode (Letters)
- **Input:** Single `HandFrame` of 21 landmarks ($x, y, z$).
- **Features:** 63 normalized floats (wrist at origin, normalized by maximum distance from wrist).
- **Canonicalization:** Letter handedness canonicalization ($x = 1 - x$) based on dataset majority label in `letters_meta.json`.
- **Model:** `RandomForestClassifier(n_estimators=300)`.
- **Inference Latency:** Target $<10\text{ ms}$ on CPU.

### 5.2 Words Mode (Isolated Signs)
- **Input:** `LandmarkWindow` containing a sequence of `LandmarkFrame`s.
- **Preprocessing:**
  - Leading/trailing idle hand frames trimmed.
  - Coordinate normalization: Shoulder midpoint as origin; shoulder width as unit scale.
  - Slot assignment: Left vs. right hand determined by distance to pose wrists (indices 15 & 16), immune to handedness label flips.
  - Dominant-hand mirroring: Computed via cumulative wrist kinetic energy ($x \to -x$ if left-dominant).
  - Temporal resampling: Uniform linear interpolation to fixed length $T = 32$ frames ($104$ features/frame).
- **Architecture:** 1D Convolutional feature extractor + Bidirectional GRU (hidden size 128) + Linear classification head ($<2\text{M}$ parameters).
- **Runtime:** Exported to ONNX (`word_model.onnx`) executed via `onnxruntime` with dynamic batching.
- **Gate Requirement:** $\ge 0.75$ top-1 accuracy on held-out signers (`SIGNBRIDGE_WORD_GATE`).

---

## 6. Vocabulary & Selection Criteria

The 30 isolated sign words selected for the closed vocabulary satisfy the strict spec requirements:
- Every word has $\ge 150$ sequences in the dataset (actual minimum: 312 sequences).
- Every word is signed by $\ge 15$ distinct signers (actual: all 21 participants sign each word).

| Category | Words |
| :--- | :--- |
| **Greetings** | `hello`, `bye`, `thankyou`, `please` |
| **Questions** | `who`, `where`, `why`, `if`, `any` |
| **Family** | `mom`, `dad`, `brother`, `grandma` |
| **Food & Drink** | `apple`, `milk`, `water`, `pizza` |
| **Actions** | `jump`, `read`, `sleep`, `dance`, `drink` |
| **Feelings** | `happy`, `sad`, `sick` |
| **Objects** | `book`, `chair`, `car`, `shoe`, `pen` |

---

## 7. Model Evaluation & Pruning Log

*(To be populated following initial training and validation runs)*

| Iteration | Vocabulary Size | Held-out Top-1 | Held-out Top-3 | Pruned Words | Rationale |
| :---: | :---: | :---: | :---: | :---: | :--- |
| Baseline | 30 words | TBD | TBD | None | Initial 30-word training set |
