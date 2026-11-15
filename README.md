# NeuroAlert — EEG-Based Epileptic Seizure Detection

NeuroAlert is a research prototype for detecting potential epileptic seizure activity in electroencephalogram (EEG) recordings using deep learning.

The project combines two CNN–BiGRU models trained with different loss functions and evaluates their ensemble on an independent test set.

> **Research use only.** NeuroAlert is not clinically validated and must not be used for diagnosis, emergency response, or patient monitoring.

## Project Overview

The system processes multichannel EEG recordings in non-overlapping 10-second windows. It applies signal preprocessing, performs inference using two trained neural networks, and combines their outputs using fixed-reference rank averaging.

### Models

- CNN–BiGRU trained with binary cross-entropy (BCE).
- CNN–BiGRU trained with focal loss.
- Ensemble weighting: 50% BCE and 50% Focal.
- Decision threshold: 0.99, selected using validation data.

The ensemble scores are rank-based scores, not calibrated probabilities of seizure.

## EEG Preprocessing

- Input: EDF recordings containing the required 17 EEG channels.
- Original sampling frequency: 256 Hz.
- Target sampling frequency: 128 Hz.
- Window duration: 10 seconds.
- Band-pass filter: 0.5–40 Hz, fourth-order Butterworth.
- Zero-phase filtering applied per window.
- Normalization using training-only channel statistics.
- Normalized values clipped to the range [-8, 8].

The inference script expects recordings with the required channel names and sampling frequency.

## Independent Test Results

The final evaluation used four held-out CHB-MIT subjects: chb02, chb09, chb14, and chb22.

| Metric | Result |
|---|---:|
| Test windows | 9,199 |
| Seizure windows | 57 |
| Precision | 75.00% |
| Recall (window level) | 57.89% |
| F1-score | 65.35% |
| Specificity | 99.88% |
| Accuracy | 99.62% |
| PR-AUC | 0.6467 |
| ROC-AUC | 0.8886 |
| True positives | 33 |
| False positives | 11 |
| False negatives | 24 |
| True negatives | 9,131 |

### Event-Level Evaluation

| Metric | Result |
|---|---:|
| Total seizure events | 12 |
| Detected events | 7 |
| Missed events | 5 |
| Event sensitivity | 58.33% |
| False alarm episodes | 9 |
| False alarms per negative hour | 0.3544 |

These results indicate important limitations. The test set is highly imbalanced, making accuracy alone an insufficient measure of performance. Five of twelve seizure events were missed, and performance varied substantially across patients. The system is not reliable enough for clinical deployment.

The independent test set was not used for threshold tuning or model retraining.

## Repository Structure

```text
NeuroAlert-Seizure-Detection/
├── predict.py
├── models/
│   ├── cnn_bigru_bce_best.pt
│   └── cnn_bigru_focal_best.pt
├── cache/
│   └── normalization_stats.json
├── results/
│   └── ensemble_v1/
│       ├── ensemble_metadata.json
│       └── validation_score_references.npz
└── README.md
```

## Installation

Python 3.10 or newer is recommended. Install the required Python packages:

```bash
python -m pip install numpy pandas torch mne matplotlib scipy
```

## Running Inference

After cloning the repository and preparing a compatible EDF recording, run:

```bash
python predict.py --edf /path/to/recording.edf --output output_run
```

For a limited demonstration:

```bash
python predict.py --edf /path/to/recording.edf --output output_run --max-windows 10
```

The output directory must not already exist.

Expected outputs:

- `window_predictions.csv`
- `detected_episodes.csv`
- `prediction_report.json`
- `prediction_timeline.png`

**Reproducibility status:** Model checkpoint loading and configuration consistency have been checked. Complete end-to-end inference on a compatible EDF recording has not yet been verified in this repository.

## Dataset

The project uses EEG data from the CHB-MIT Scalp EEG Database, available through PhysioNet:

https://physionet.org/content/chbmit/

The original EEG recordings are not included in this repository. Users should obtain data from the original source and follow its usage and citation requirements.

## Limitations and Future Work

- Limited independent test population.
- Uneven performance across held-out patients.
- Missed seizure events and remaining false alarms.
- Strict requirements for EEG channels and sampling frequency.
- No prospective clinical validation.
- No demonstrated real-time monitoring performance.

Future work may investigate broader patient-level validation, improved event sensitivity, calibration, and robustness across recording conditions.

## Disclaimer

NeuroAlert is an experimental machine-learning research project. Its outputs are not medical diagnoses and must not be used to make clinical decisions or provide real-time patient safety alerts.
