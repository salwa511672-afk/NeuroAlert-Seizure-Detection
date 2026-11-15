
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
import mne
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, resample_poly
from torch import nn


ROOT = Path("/content/drive/MyDrive/NeuroAlert_Detection")


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class SeizureCNNBiGRU(nn.Module):

    def __init__(self, channels=17, hidden_size=48):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(channels, 32, 9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        self.temporal_pool = nn.AdaptiveAvgPool1d(40)

        self.bigru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, 64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.temporal_pool(x).transpose(1, 2)

        sequence, _ = self.bigru(x)

        features = torch.cat(
            [
                sequence.mean(dim=1),
                sequence.max(dim=1).values
            ],
            dim=1
        )

        return self.classifier(features).squeeze(1)


# ============================================================
# FIXED ENSEMBLE INFERENCE
# ============================================================

class NeuroAlertPredictor:

    def __init__(self, root=ROOT, device=None):

        self.root = Path(root)
        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        )

        cache = self.root / "cache"
        models = self.root / "models"
        ensemble = self.root / "results/ensemble_v1"

        with open(
            cache / "normalization_stats.json",
            encoding="utf-8"
        ) as f:
            norm = json.load(f)

        with open(
            ensemble / "ensemble_metadata.json",
            encoding="utf-8"
        ) as f:
            meta = json.load(f)

        self.channels = list(norm["channels"])
        self.mean = np.asarray(
            norm["mean"], dtype=np.float32
        )[:, None]

        self.std = np.asarray(
            norm["std"], dtype=np.float32
        )[:, None]

        self.fs = int(norm["source_fs"])
        self.target_fs = int(norm["target_fs"])
        self.seconds = int(norm["window_seconds"])

        self.clip_min, self.clip_max = (
            norm["normalization_clip"]
        )

        self.threshold = float(
            meta["selected_threshold"]
        )

        self.weight_bce = float(
            meta["weights"]["BCE"]
        )

        self.weight_focal = float(
            meta["weights"]["Focal"]
        )

        assert len(self.channels) == 17
        assert self.fs == 256
        assert self.target_fs == 128
        assert self.seconds == 10
        assert norm["computed_from"] == "train_only"
        assert norm["filter_mode"] == "zero_phase_per_window"

        assert self.threshold == 0.99
        assert self.weight_bce == 0.5
        assert self.weight_focal == 0.5

        assert np.isfinite(self.mean).all()
        assert np.isfinite(self.std).all()
        assert (self.std > 1e-6).all()

        self.sos = butter(
            int(norm["filter_order"]),
            norm["bandpass_hz"],
            fs=self.fs,
            btype="bandpass",
            output="sos"
        )

        with np.load(
            ensemble / "validation_score_references.npz"
        ) as refs:

            self.bce_reference = np.asarray(
                refs["bce_sorted"],
                dtype=np.float64
            ).copy()

            self.focal_reference = np.asarray(
                refs["focal_sorted"],
                dtype=np.float64
            ).copy()

            reference_threshold = float(
                refs["selected_threshold"][0]
            )

        assert reference_threshold == self.threshold

        assert len(self.bce_reference) == 9951
        assert len(self.focal_reference) == 9951

        self.bce = self._load_model(
            models / "cnn_bigru_v1/best_model.pt"
        )

        self.focal = self._load_model(
            models / "cnn_bigru_focal_v1/best_model.pt"
        )

    def _load_model(self, path):

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        assert checkpoint["channels"] == self.channels
        assert checkpoint["input_shape"] == [17, 1280]

        model = SeizureCNNBiGRU(
            channels=17,
            hidden_size=48
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        model = model.to(self.device)
        model.eval()

        return model

    @staticmethod
    def _reference_percentile(scores, reference):

        n = len(reference)

        left = np.searchsorted(
            reference,
            scores,
            side="left"
        )

        right = np.searchsorted(
            reference,
            scores,
            side="right"
        )

        return np.clip(
            (left + right + 1) / (2.0 * n),
            0,
            1
        )

    def _preprocess(self, raw_volts):

        # Match the previously audited preprocessing:
        # volts -> microvolts
        # zero-phase Butterworth per 10-second window
        # polyphase resampling
        # float16 intermediate cache
        # training-only normalization

        x = np.asarray(
            raw_volts,
            dtype=np.float64
        ) * 1e6

        if x.shape != (17, 2560):
            raise ValueError(
                f"Unexpected EEG window shape: {x.shape}"
            )

        x = sosfiltfilt(
            self.sos,
            x,
            axis=-1
        )

        x = resample_poly(
            x,
            up=self.target_fs,
            down=self.fs,
            axis=-1
        )

        x = x.astype(np.float16).astype(np.float32)

        x = (x - self.mean) / self.std

        np.clip(
            x,
            self.clip_min,
            self.clip_max,
            out=x
        )

        if not np.isfinite(x).all():
            raise RuntimeError(
                "EEG preprocessing produced invalid values"
            )

        return np.ascontiguousarray(
            x, dtype=np.float32
        )

    @torch.inference_mode()
    def _predict_batch(self, windows):

        batch = torch.from_numpy(
            np.stack(windows)
        ).to(self.device)

        score_bce = torch.sigmoid(
            self.bce(batch)
        ).cpu().numpy()

        score_focal = torch.sigmoid(
            self.focal(batch)
        ).cpu().numpy()

        return score_bce, score_focal

    def predict_edf(
        self,
        edf_path,
        batch_size=64,
        max_windows=None
    ):

        edf_path = Path(edf_path)

        if not edf_path.is_file():
            raise FileNotFoundError(edf_path)

        raw = mne.io.read_raw_edf(
            str(edf_path),
            preload=False,
            verbose="ERROR"
        )

        rows = []

        try:

            sfreq = float(raw.info["sfreq"])

            if not np.isclose(sfreq, self.fs):
                raise ValueError(
                    f"Expected {self.fs} Hz, got {sfreq} Hz"
                )

            picks = []

            for channel in self.channels:

                matches = [
                    i for i, name in enumerate(raw.ch_names)
                    if name.strip().upper() == channel.upper()
                ]

                if len(matches) != 1:
                    raise ValueError(
                        f"Channel {channel}: "
                        f"expected one match, found {len(matches)}"
                    )

                picks.append(matches[0])

            window_samples = self.fs * self.seconds

            total_windows = (
                raw.n_times // window_samples
            )

            if max_windows is not None:
                if max_windows < 1:
                    raise ValueError(
                        "max_windows must be >= 1"
                    )
                total_windows = min(
                    total_windows,
                    max_windows
                )

            if total_windows == 0:
                raise ValueError(
                    "EDF is shorter than 10 seconds"
                )

            for batch_start in range(
                0, total_windows, batch_size
            ):

                batch_end = min(
                    total_windows,
                    batch_start + batch_size
                )

                windows = []

                for index in range(
                    batch_start, batch_end
                ):

                    start = index * window_samples
                    stop = start + window_samples

                    raw_window = raw.get_data(
                        picks=picks,
                        start=start,
                        stop=stop
                    )

                    windows.append(
                        self._preprocess(raw_window)
                    )

                bce_scores, focal_scores = (
                    self._predict_batch(windows)
                )

                bce_ranks = self._reference_percentile(
                    bce_scores,
                    self.bce_reference
                )

                focal_ranks = self._reference_percentile(
                    focal_scores,
                    self.focal_reference
                )

                ensemble_scores = (
                    self.weight_bce * bce_ranks
                    + self.weight_focal * focal_ranks
                )

                for j, index in enumerate(
                    range(batch_start, batch_end)
                ):

                    score = float(
                        ensemble_scores[j]
                    )

                    rows.append({
                        "file": edf_path.name,
                        "start_sec": int(
                            index * self.seconds
                        ),
                        "end_sec": int(
                            (index + 1) * self.seconds
                        ),
                        "score_bce": float(bce_scores[j]),
                        "score_focal": float(focal_scores[j]),
                        "ensemble_score": score,
                        "prediction": int(
                            score >= self.threshold
                        )
                    })

        finally:
            raw.close()

        return pd.DataFrame(rows)

    def summarize_episodes(self, predictions):

        positive = predictions[
            predictions["prediction"] == 1
        ].sort_values("start_sec")

        episodes = []

        for row in positive.itertuples(index=False):

            start = int(row.start_sec)
            end = int(row.end_sec)

            if (
                episodes
                and start == episodes[-1]["end_sec"]
            ):
                episodes[-1]["end_sec"] = end
                episodes[-1]["windows"] += 1
                episodes[-1]["max_score"] = max(
                    episodes[-1]["max_score"],
                    float(row.ensemble_score)
                )
            else:
                episodes.append({
                    "start_sec": start,
                    "end_sec": end,
                    "windows": 1,
                    "max_score": float(row.ensemble_score)
                })

        result = pd.DataFrame(
            episodes,
            columns=[
                "start_sec",
                "end_sec",
                "windows",
                "max_score"
            ]
        )

        if len(result):
            result["duration_sec"] = (
                result["end_sec"]
                - result["start_sec"]
            )

        return result

    def save_results(
        self,
        predictions,
        edf_path,
        output_dir
    ):

        output_dir = Path(output_dir)

        if output_dir.exists():
            raise FileExistsError(
                f"Output already exists: {output_dir}"
            )

        episodes = self.summarize_episodes(
            predictions
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=False
        )

        predictions.to_csv(
            output_dir / "window_predictions.csv",
            index=False
        )

        episodes.to_csv(
            output_dir / "detected_episodes.csv",
            index=False
        )

        report = {
            "file": Path(edf_path).name,
            "windows_analyzed": len(predictions),
            "positive_windows": int(
                predictions["prediction"].sum()
            ),
            "potential_episodes": int(len(episodes)),
            "threshold": self.threshold,
            "ensemble_weights": {
                "BCE": self.weight_bce,
                "Focal": self.weight_focal
            },
            "window_seconds": self.seconds,
            "clinical_use": False,
            "disclaimer": (
                "Research demonstration only. "
                "Not for diagnosis or medical monitoring."
            )
        }

        with open(
            output_dir / "prediction_report.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False
            )

        fig, ax = plt.subplots(
            figsize=(14, 5)
        )

        midpoints = (
            predictions["start_sec"]
            + predictions["end_sec"]
        ) / 2 / 60

        ax.plot(
            midpoints,
            predictions["ensemble_score"],
            linewidth=1.5,
            color="#2563EB",
            label="Ensemble score"
        )

        ax.axhline(
            self.threshold,
            color="#DC2626",
            linestyle="--",
            label=f"Threshold = {self.threshold}"
        )

        for episode in episodes.itertuples(index=False):

            ax.axvspan(
                episode.start_sec / 60,
                episode.end_sec / 60,
                alpha=0.18,
                color="#DC2626"
            )

        ax.set_title(
            "NeuroAlert - EEG Potential Seizure Windows"
        )

        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Ensemble rank score")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.legend()

        fig.tight_layout()

        fig.savefig(
            output_dir / "prediction_timeline.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        return report


def main():

    parser = argparse.ArgumentParser(
        description="NeuroAlert research EEG inference"
    )

    parser.add_argument(
        "--edf",
        required=True,
        help="Path to EDF recording"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="New output directory"
    )

    parser.add_argument(
        "--max-windows",
        type=int,
        default=None
    )

    args = parser.parse_args()

    predictor = NeuroAlertPredictor()

    predictions = predictor.predict_edf(
        args.edf,
        max_windows=args.max_windows
    )

    report = predictor.save_results(
        predictions,
        args.edf,
        args.output
    )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        )
    )

    print("PREDICTION COMPLETED")


if __name__ == "__main__":
    main()
