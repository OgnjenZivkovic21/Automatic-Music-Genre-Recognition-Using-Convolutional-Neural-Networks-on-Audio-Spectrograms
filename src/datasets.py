"""PyTorch Dataset koji cita manifest CSV, secka svaku pesmu na kratke
isecke i vraca mel-spektrogram + labelu zanra."""
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

from features import load_audio, extract_mel_spectrogram


class GTZANSpectrogramDataset(Dataset):
    def __init__(self, manifest_csv, genres, sample_rate=22050,
                 segment_duration=3, n_mels=128):
        self.genres = genres
        self.genre_to_idx = {g: i for i, g in enumerate(genres)}
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.n_mels = n_mels

        with open(manifest_csv) as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        y, sr = load_audio(row["path"], sample_rate=self.sample_rate)

        seg_len = self.segment_duration * sr
        if len(y) > seg_len:
            start = np.random.randint(0, len(y) - seg_len)
            y = y[start:start + seg_len]
        else:
            y = np.pad(y, (0, max(0, seg_len - len(y))))

        log_mel = extract_mel_spectrogram(y, sr, n_mels=self.n_mels)
        label = self.genre_to_idx[row["genre"]]

        x = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        return x, label
