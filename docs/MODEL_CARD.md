# SignBridge Model Card & Dataset Documentation

**Document Owner:** Person B (Data & Machine Learning)  
**Specification Reference:** Build Specification v2 (§2, §14, §19)  
**Last Updated:** November 9, 2026

---

## 1. Overview & System Purpose

SignBridge is a lightweight, privacy-preserving two-way American Sign Language (ASL) educational bridge designed for real-time browser interaction:
1. **Sign to Text (Spell Mode):** Recognizes static ASL fingerspelled letters (A-Z) from single-hand MediaPipe landmarks.
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
  - The approved preprocessed tensors, manifest, and training checkpoints are also local-only and must not be committed to this repository.
  - `.gitignore` excludes `backend/data/raw/`, `backend/data/processed/`, and model checkpoint files.
- **Competition Rule Compliance:**
  - Used strictly for non-commercial educational and research evaluation.
  - `backend/data/selected_parquet_paths.txt` and `backend/data/selected_sequences.csv` contain the same 11,316 unique candidate paths across 30 words and 21 signers. These are candidates, not all records used by the trained model.
  - `backend/data/processed/word_manifest.csv` selects 6,000 unique candidates (200 per word): 140 train, 30 validation, and 30 test records per word. It uses signer-disjoint splits of 4,200/900/900.
  - The manifest was verified as a subset of the 11,316 candidate paths. Its ordered paths, labels, splits, sequence IDs, and signer IDs match the saved arrays in `word_dataset.npz`. The remaining 5,316 paths are additional candidates, not required examples for this saved training split.
  - `build_word_manifest.py` includes verified on-disk training candidates before sampling to its 140-per-word quota. If all 11,316 candidates were downloaded, the training pool would exceed that quota for every word and the builder would reject it. A full-candidate download is therefore not needed for the current model or its fixed 6,000-record manifest. To reprocess the saved manifest, retain that manifest and make its 6,000 referenced Parquet files available; do not rebuild it from a different set of locally available files and assume the result is identical.
  - At the latest local audit, all 6,000 manifest Parquet paths were present on disk (zero missing). The local raw directory contained 6,250 Parquet files. The complete raw set is not necessary to run inference or retrain from the already processed `word_dataset.npz`. All files referenced by the saved manifest are currently available locally, allowing the NPZ to be regenerated from that manifest.

### 2.2 Spell Mode: ASL Alphabet Dataset & Team Live Recordings
- **Primary Source:** Kaggle [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), approximately 87,000 200x200 images in A-Z, `del`, `nothing`, and `space` classes. The letter classifier uses A-Z only; the other labels are excluded.
- **Dataset Listing Licence:** GPL-2, as specified by the Kaggle dataset listing. This is separate from the word dataset's CC BY 4.0 licence above; do not apply the word-corpus attribution or licence to alphabet images.
- **Access Date:** November 7, 2026.
- **Data handling:** Raw images and derived feature CSVs are local-only and must not be committed. The static-pose classifier does not make J/Z movement recognition reliable.
- **Domain Adaptation / Team Recordings:** The specification calls for 50 live samples per letter from each of three team members. These recordings are intended for person-level holdout evaluation and subsequent training; do not report them as collected or evaluated until results are available.

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
- **Architecture:** Two 1D convolution blocks (128 channels each; convolution, batch normalization, ReLU, dropout 0.15), followed by a two-layer bidirectional GRU (128 hidden units per direction, dropout 0.3 between layers), temporal mean/max pooling, and a 512-to-128-to-30 linear head with dropout 0.3. The verified checkpoint has **653,982 trainable parameters**.
- **Input/output:** float32 tensor `(batch, 32, 104)` to 30 class logits. The ONNX graph has a symbolic batch dimension.
- **Runtime:** `backend.app.recognition.onnx_word_model.ONNXWordRecognizer` loads `word_model.onnx` and `word_vocab.json`; `recognize(LandmarkWindow, top_k=3)` returns `Gesture` candidates ranked by confidence.
- **Gate Requirement:** $\ge 0.75$ top-1 accuracy on held-out signers (`SIGNBRIDGE_WORD_GATE`).

---

## 6. Vocabulary & Selection Criteria

The closed vocabulary is alphabetically ordered. This exact order defines the output class indices in the checkpoint, vocabulary JSON, processed metadata, and ONNX metadata:

| Class ID | Label | Category |
| ---: | --- | --- |
| 0 | `any` | Questions |
| 1 | `apple` | Food & Drink |
| 2 | `book` | Objects |
| 3 | `brother` | Family |
| 4 | `bye` | Greetings |
| 5 | `car` | Objects |
| 6 | `chair` | Objects |
| 7 | `dad` | Family |
| 8 | `dance` | Actions |
| 9 | `drink` | Actions |
| 10 | `grandma` | Family |
| 11 | `happy` | Feelings |
| 12 | `hello` | Greetings |
| 13 | `if` | Questions |
| 14 | `jump` | Actions |
| 15 | `milk` | Food & Drink |
| 16 | `mom` | Family |
| 17 | `pen` | Objects |
| 18 | `pizza` | Food & Drink |
| 19 | `please` | Greetings |
| 20 | `read` | Actions |
| 21 | `sad` | Feelings |
| 22 | `shoe` | Objects |
| 23 | `sick` | Feelings |
| 24 | `sleep` | Actions |
| 25 | `thankyou` | Greetings |
| 26 | `water` | Food & Drink |
| 27 | `where` | Questions |
| 28 | `who` | Questions |
| 29 | `why` | Questions |

`label_to_id` maps each label to the ID above; `id_to_label` is the inverse using stringified IDs as JSON keys. The saved vocabulary and processed metadata contain these maps. The checkpoint stores the ordered vocabulary but does not store explicit mapping fields.

The source candidate CSV contains 11,316 sequences across 21 signers. The fixed manifest selects 200 records per word from all 21 signers. Candidate availability by signer split is 7,275 train-signer candidates, 1,677 validation-signer candidates, and 2,364 test-signer candidates; the manifest selects 140/30/30 per word respectively. Every word has 21 signers represented in the candidate file. The held-out split is signer-independent, not a random sample of the training signers.

---

## 7. Model Evaluation & Pruning Log

The 1D CNN + BiGRU checkpoint was selected using validation data only. The held-out test split was reserved for final reporting and was not used for model selection or tuning. Results below are recorded in the local `backend/models/word_eval_metrics.json`; the test metrics and confusion matrix were independently recomputed from the saved checkpoint and processed test split and matched:

| Metric | Result |
| :--- | ---: |
| Trainable parameters | 653,982 |
| Best epoch | 28 |
| Best validation loss | 1.1007 |
| Best validation accuracy | 85.78% |
| Training-split loss at selected checkpoint (evaluation mode) | 0.8012 |
| Held-out test loss | 1.2461 |
| Held-out test top-1 accuracy | 80.44% |
| Macro precision | 81.45% |
| Macro recall | 80.44% |
| Macro F1 | 80.23% |
| Model-card accuracy gate (>= 75%) | Passed |

Each test class has 30 examples. `chair` is the only word below 60% per-word test accuracy: precision 86.67%, recall/accuracy 43.33%, and F1 57.78%. It is a weak class despite the overall gate passing. The full per-word precision, recall, F1, accuracy, support, and 30-by-30 confusion matrix are in the local evaluation JSON. Train and validation losses/accuracy and per-class results are evidence in the saved evaluation artifact; the reported overall test values match fresh inference.

### Training configuration provenance

The checkpoint verifies the architecture, parameter count, selected epoch (28), validation metrics, and ordered vocabulary. It does **not** contain the complete run configuration, seed, optimizer state provenance, or a reproducible training history. The saved evaluation JSON also omits training history and elapsed duration. Consequently, the exact historical values for epoch limit, batch size, learning rate, weight decay, label smoothing, patience, seed, and hardware are **unverified**. Do not treat the current script defaults as confirmed settings for this checkpoint.

The current training script’s callable defaults are 50 epochs, batch size 32, learning rate `1e-3`, weight decay `1e-4`, label smoothing `0.1`, patience 10, and seed 42. These are code defaults for a new run, not proof of the settings that produced the saved artifact. The script’s command-line interface exposes epochs, batch size, learning rate, patience, and seed; weight decay and label smoothing are currently function defaults.

The selected dataset is class-balanced at 140/30/30 records per class across train/validation/test, so this split has no class-count imbalance. The source population and signing behavior may still be unrepresentative. signer-disjoint evaluation reduces signer leakage between splits; it does not establish generalization to all ASL signers, environments, or cameras. The notably weak `chair` result and isolated-word scope are material limitations.

## 8. Inference, Artifacts, and Reproduction

### 8.1 Run ONNX inference

From the repository root, install project runtime dependencies and provision the ONNX file and vocabulary as described below. The recognizer consumes a `LandmarkWindow`; each frame has exactly nine `(x, y)` pose points in `POSE_INDICES` order and either 21 `(x, y)` points per detected hand or `None`. The window applies the same normalization, resampling, and dominant-hand canonicalization as the offline preprocessing path.

```python
from backend.app.domain.sequence import LandmarkFrame, LandmarkWindow
from backend.app.recognition.onnx_word_model import ONNXWordRecognizer

# Replace these example coordinates with detections for every frame in a sign.
pose = tuple((0.5, 0.5) for _ in range(9))
hand = tuple((0.5, 0.5) for _ in range(21))
frames = [
    LandmarkFrame(pose=pose, left_hand=hand, right_hand=None)
    for _ in range(32)
]
window = LandmarkWindow(frames)

recognizer = ONNXWordRecognizer()
recognizer.load()
for candidate in recognizer.recognize(window, top_k=3):
    print(candidate.label, candidate.confidence)
```

The example coordinates only demonstrate the expected structure and are not meaningful signs. In application code, pass ordered live or recorded landmarks. Confidence is the class softmax probability over all 30 classes; returned `Gesture` values are sorted highest first. A truncated top-k list need not sum to 1.

The default model paths are `backend/models/word_model.onnx` and `backend/models/word_vocab.json`. They can be overridden with the existing settings environment variables:

```powershell
$env:SIGNBRIDGE_WORD_MODEL_PATH = "C:\models\word_model.onnx"
$env:SIGNBRIDGE_WORD_VOCAB_PATH = "C:\models\word_vocab.json"
```

### 8.2 Artifact provisioning

`word_model.pt` and `word_model.onnx` are ignored by Git and are not distributed in a clean clone. Serving requires the ONNX model and the matching `word_vocab.json`; the PyTorch checkpoint is needed for evaluation, continued training, or ONNX re-export, not normal ONNX inference. Provision binary artifacts through the team’s approved artifact storage or release process, verify their checksums and vocabulary/model version pairing, and place them at the configured paths. Do not force-add model binaries to Git. `word_vocab.json` is tracked, but it must match the provisioned ONNX output class order.

#### Canonical local artifacts and integrity

The following inventory records the artifacts present during this verification. The two model binaries and processed data are Git-ignored; the vocabulary and evaluation metrics are tracked. Sizes are bytes. SHA-256 values are for exact-file integrity checks, not signatures or proof of provenance.

| File | Format / purpose | Size | SHA-256 |
| --- | --- | ---: | --- |
| `backend/models/word_model.pt` | PyTorch checkpoint; needed for checkpoint loading and re-export | 7,884,487 | `1c9640febfd0e0e8e3dcfd9bbf1113ad124b035aad342047231809dd84a90f23` |
| `backend/models/word_model.onnx` | ONNX model; required for ONNX inference | 2,619,318 | `fdb233bc107df129408a78cb455c1b9a0e5b5d5a7864fbd4d8814f111bb5ad83` |
| `backend/models/word_vocab.json` | JSON class labels and ID maps; required alongside ONNX | 1,560 | `f524c7e654537ebaf1cee3fe2dfd337d8ee24d171ac4a3899a2d2d5df9d079b0` |
| `backend/models/word_eval_metrics.json` | JSON evaluation results; not an inference dependency | 15,800 | `4fa0332e6530597e92c185cbd4a8460832ef5bac6756dde0bef654763ad9da29` |
| `backend/data/processed/meta.json` | JSON processed-data metadata; optional verifier ordering cross-check | 2,561 | `ed354eff4672690309133e2b5c5d5d79bfd8d8ba486911c700a40f3aae3fa672` |
| `backend/data/processed/word_dataset.npz` | NumPy compressed tensors; optional verifier input and required for re-export | 34,541,034 | `98380076ef1c32e7d3f1e4fc4f81e82d4a3bfd8bd1e070af885b9515f1ca4b83` |
| `backend/data/processed/word_manifest.csv` | CSV manifest; needed to reproduce preprocessing from raw files | 445,288 | `13ab367f9f6dcab86d37b1226a5651f5432e4564abc00d26484313957a7e44c6` |

No team-approved artifact host, release, or transfer location is currently specified here. Obtain the matching model pair from the responsible project owner through a team-approved transfer mechanism; the location and access procedure remain a team decision. Copy them without renaming to `backend/models/word_model.pt` and `backend/models/word_model.onnx`. Keep the tracked vocabulary at `backend/models/word_vocab.json`, and verify that it is the paired version. Do not commit the ignored binaries or select an external hosting service without team approval.

In PowerShell, check copied files against the hashes above:

```powershell
Get-FileHash backend\models\word_model.pt -Algorithm SHA256
Get-FileHash backend\models\word_model.onnx -Algorithm SHA256
Get-FileHash backend\models\word_vocab.json -Algorithm SHA256
```

With the project's development dependencies installed, run the artifact verifier from the repository root. It checks the checkpoint, 30-class vocabulary and mappings, ONNX graph and CPU session, finite inference, normalized softmax, and PyTorch/ONNX parity (maximum absolute logit difference at most `1e-4`):

```powershell
python backend\scripts\verify_word_model_artifacts.py
```

The verifier requires PyTorch, ONNX, ONNX Runtime, and NumPy (included by `requirements-dev.txt`). It uses a test-split tensor from `word_dataset.npz` when available. If processed tensors are absent, it reports and uses deterministic synthetic input; this does not verify the real-data preprocessing path. A present but malformed processed dataset causes verification to fail. The `--checkpoint`, `--onnx`, `--vocabulary`, `--sample-data`, and `--metadata` options allow checking files in another location without copying them into the repository.

### 8.3 Reproduce a new training run and ONNX export

Training requires the validated local processed files `backend/data/processed/word_dataset.npz` and `meta.json`; raw data download is not part of this procedure. The checkpoint, vocabulary, and evaluation JSON are written to `--models-dir`, so use a separate output directory to avoid replacing existing artifacts.

```powershell
python backend/scripts/train_word_model.py `
  --dataset backend/data/processed/word_dataset.npz `
  --meta backend/data/processed/meta.json `
  --models-dir backend/data/processed/reproduction-model `
  --epochs 50 --batch-size 32 --lr 0.001 --patience 10 --seed 42

python backend/scripts/export_word_model_onnx.py `
  --checkpoint backend/data/processed/reproduction-model/word_model.pt `
  --vocabulary backend/data/processed/reproduction-model/word_vocab.json `
  --dataset backend/data/processed/word_dataset.npz `
  --meta backend/data/processed/meta.json `
  --output backend/data/processed/reproduction-model/word_model.onnx `
  --validation-samples 50
```

This is an explicit example using current CLI values and code defaults for a **new** run; it does not reproduce or establish the unknown historical training configuration. The exporter checks ONNX validity, dynamic shapes, and PyTorch/ONNX parity on deterministic validation samples. To regenerate tensors from raw Parquet, use the existing fixed `word_manifest.csv` and ensure every referenced file is available before running `backend/scripts/preprocess_words.py`; the raw files are ignored/local-only. Rebuilding the manifest from candidate paths can select a different subset because its training selection depends on which candidate Parquet files are already on disk.
