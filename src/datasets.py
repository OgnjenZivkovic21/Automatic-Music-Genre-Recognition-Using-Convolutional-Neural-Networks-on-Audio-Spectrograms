"""PyTorch Dataset koji cita manifest CSV, secka svaku pesmu na kratke
isecke i vraca mel-spektrogram + labelu zanra.

Kesiranje: mel-spektrogram CELE pesme se racuna samo jednom po fajlu (prvi put
kad se zatrazi) i cuva u memoriji (self._spec_cache). Svaki naredni zahtev za
istu pesmu samo isece nasumican deo iz vec izracunatog spektrograma - bez
ponovnog citanja audio fajla sa diska i bez ponovnog racunanja FFT-a. Kes
zivi samo dok skripta radi (nestaje kad se proces zavrsi)."""
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

from features import (
    load_audio,
    extract_mel_spectrogram,
    spec_augment,
    DEFAULT_HOP_LENGTH,
)


class GTZANSpectrogramDataset(Dataset):
    def __init__(self, manifest_csv, genres, sample_rate=22050,
                 segment_duration=3, n_mels=128, augment=False,
                 samples_per_track=1):
        """augment=True: primeni SpecAugment na spektrogram (koristiti SAMO
        za trening). samples_per_track: koliko nasumicnih isecaka po pesmi
        po epohi - efektivno mnozi velicinu trening skupa bez novih audio
        fajlova (koristiti >1 SAMO za trening, val/test ostaju na 1)."""
        self.genres = genres
        self.genre_to_idx = {g: i for i, g in enumerate(genres)}
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.n_mels = n_mels
        self.augment = augment
        self.samples_per_track = samples_per_track
        self._spec_cache = {}

        with open(manifest_csv) as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows) * self.samples_per_track

    def _get_full_spectrogram(self, path):
        """Vraca mel-spektrogram CELE pesme (128 x ceo broj frejmova),
        iz kesa ako je vec izracunat, inace ga racuna i kesira."""
        cached = self._spec_cache.get(path)
        if cached is not None:
            return cached

        y, sr = load_audio(path, sample_rate=self.sample_rate)
        full_log_mel = extract_mel_spectrogram(y, sr, n_mels=self.n_mels)
        self._spec_cache[path] = full_log_mel
        return full_log_mel

    def get_eval_crops(self, idx, n_crops=5):
        """Vraca N ravnomerno rasporedjenih isecaka CELE pesme (bez
        nasumicnosti, bez SpecAugment-a) - koristi se za test-time averaging
        u evaluate.py: napravi se predikcija za svaki isecak, pa se softmax
        verovatnoce usrednje umesto da se ceo rezultat oslanja na samo jedan
        (nasumican) isecak. idx ovde ide 0..len(self.rows)-1 (BEZ mnozenja
        sa samples_per_track - to je samo za trening)."""
        row = self.rows[idx]
        full_log_mel = self._get_full_spectrogram(row["path"])

        frames_per_seg = int(round(
            self.segment_duration * self.sample_rate / DEFAULT_HOP_LENGTH
        ))
        n_frames_total = full_log_mel.shape[1]

        if n_frames_total > frames_per_seg:
            offsets = np.linspace(0, n_frames_total - frames_per_seg, n_crops).astype(int)
        else:
            offsets = [0] * n_crops

        crops = []
        for f0 in offsets:
            if n_frames_total > frames_per_seg:
                log_mel = full_log_mel[:, f0:f0 + frames_per_seg].copy()
            else:
                pad_width = frames_per_seg - n_frames_total
                log_mel = np.pad(
                    full_log_mel, ((0, 0), (0, pad_width)),
                    mode="constant", constant_values=full_log_mel.min(),
                )
            crops.append(log_mel)

        # (n_crops, 1, n_mels, frames_per_seg)
        x = torch.tensor(np.stack(crops), dtype=torch.float32).unsqueeze(1)
        label = self.genre_to_idx[row["genre"]]
        return x, label

    def __getitem__(self, idx):
        row = self.rows[idx % len(self.rows)]
        full_log_mel = self._get_full_spectrogram(row["path"])

        frames_per_seg = int(round(
            self.segment_duration * self.sample_rate / DEFAULT_HOP_LENGTH
        ))
        n_frames_total = full_log_mel.shape[1]

        if n_frames_total > frames_per_seg:
            f0 = np.random.randint(0, n_frames_total - frames_per_seg)
            log_mel = full_log_mel[:, f0:f0 + frames_per_seg].copy()
        else:
            pad_width = frames_per_seg - n_frames_total
            log_mel = np.pad(
                full_log_mel, ((0, 0), (0, pad_width)),
                mode="constant", constant_values=full_log_mel.min(),
            )

        if self.augment:
            log_mel = spec_augment(log_mel)
        label = self.genre_to_idx[row["genre"]]

        x = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        return x, label
