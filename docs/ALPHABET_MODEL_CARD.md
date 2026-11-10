# SignBridge ASL Alphabet Model Card

## Summary

This model classifies one detected hand pose as one of the 26 static alphabet labels `A` through `Z`. It is a prototype for educational demonstration, not a continuous sign-language translator or a substitute for a qualified interpreter.

## Model and input contract

- **Estimator:** scikit-learn `RandomForestClassifier`, 300 trees.
- **Input:** one float feature vector of length 63.
- **Output:** one label in `A`-`Z`; `predict_proba()` returns 26 values in the model's `classes_` order.
- **Class order:** uppercase alphabetical order, `A`=0 through `Z`=25. `backend/models/letters_meta.json` records the labels and both index mappings.

### Landmark preprocessing

MediaPipe detects 21 hand landmarks. Before feature extraction, landmarks are canonicalized to right-handed orientation: for a detected hand whose handedness differs from the metadata's `Right` canonical hand, each x coordinate is mirrored (`x = 1 - x`). `HandFrame.from_raw()` then validates and stores the 21 XYZ points. `HandFrame.to_feature_vector()` subtracts the wrist (landmark 0) from every point, scales all coordinates by the greatest wrist-to-landmark Euclidean distance (using 1 for a near-zero scale), and flattens the 21 XYZ triples in landmark order. This yields exactly 63 values.

## Dataset and evaluation

The alphabet pipeline uses the Kaggle ASL Alphabet image dataset. The model's labels are the 26 letters; `del`, `nothing`, and `space` are not model classes. MediaPipe landmark detections are extracted into normalized 63-feature records.

The reported evaluation is **95.074530% accuracy on 1,543 held-out samples**; the full dataset contains 8,008 samples (6,465 training and 1,543 evaluation). The described split uses `GroupShuffleSplit`, where groups are adjacent blocks of 100 original dataset images (`img0`, `img1`, etc.). These groups are image blocks, **not signer identities**, so the split does not establish signer-independent generalization. This number is a reported held-out result under that evaluation methodology; it is not a guarantee of accuracy on real-world users, camera setups, lighting, hand poses, or deployment conditions. No independently measured real-world or signer-held-out performance is claimed here.

## Limitations and intended use

- This is single-frame static-pose classification. ASL `J` and `Z` require motion and are not reliably represented by a static pose classifier.
- The label set is closed; the model has no unknown-sign class or language grammar.
- The image-block split may share signers across partitions and must not be interpreted as a signer-independent test.
- Random Forest probabilities are model scores, not calibrated measures of real-world correctness.
- Do not use for medical, legal, emergency, safety-critical, or other high-stakes communication.

## Artifacts and provisioning

The canonical files are:

| File | Purpose | Size | SHA-256 |
| --- | --- | ---: | --- |
| `backend/models/asl_model.joblib` | Trained Random Forest used for inference | 108,305,401 bytes | `d1b0600c19505f85f8c4077f0868fc01ba92414a4e737dd1721f60808c7dfc11` |
| `backend/models/letters_meta.json` | Labels, class-index maps, canonical handedness, and feature contract | 1,342 bytes | `fcfc11401216662c24b0894bcd2e9828d3fee5aabc8bd2da67145644913d47fb` |

`asl_model.joblib` is Git-ignored and is not present in a clean clone. Obtain the matching binary and metadata from the project owner using a team-approved transfer mechanism, then place them at the paths above. No particular artifact host or distribution service is selected by this document. Do not commit the model binary or substitute a different model without updating the checksum and obtaining project approval. `letters_meta.json` is tracked; it must retain the matching A-Z order and preprocessing metadata.

Check the copied model checksum in PowerShell:

```powershell
Get-FileHash backend\models\asl_model.joblib -Algorithm SHA256
```

The expected model digest is shown above. A different digest means the file is not the verified artifact; do not treat it as valid.

## Verification

From the repository root, install the project's runtime dependencies and run:

```powershell
python backend\scripts\verify_alphabet_model_artifacts.py
```

The verifier enforces the pinned SHA-256, loads the model with joblib, checks estimator type/count, input feature count, fitted tree count, class order and metadata, then runs a deterministic zero-feature smoke input through `predict_proba()` and `predict()`. It checks that all 26 probabilities are finite, within `[0, 1]`, and normalized, and that the prediction is an A-Z label consistent with the maximum probability. It exits nonzero and prints `FAIL` if any check fails.

## Runtime dependencies and integration

The alphabet model runtime requires `joblib`, `scikit-learn`, and `numpy` from `requirements.txt`. At inference, pass the canonicalized 21-point `HandFrame` to the registered letter recognizer; the configured model path defaults to `backend/models/asl_model.joblib`. The runtime model artifact and its metadata must be provisioned before live inference is available.
