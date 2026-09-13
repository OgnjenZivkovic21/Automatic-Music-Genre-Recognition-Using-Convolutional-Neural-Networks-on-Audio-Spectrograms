"""Izvlacenje feature-a: mel-spektrogram (za CNN) i klasicni rucno
izvuceni feature-i (MFCC/spektralne statistike, za SVM/RandomForest baseline).
"""
import numpy as np
import librosa


def load_audio(path, sample_rate=22050, duration=None):
    y, sr = librosa.load(path, sr=sample_rate, duration=duration)
    return y, sr


def extract_mel_spectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    """Vraca log-mel spektrogram (n_mels x vreme), ulaz za CNN."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    return librosa.power_to_db(mel, ref=np.max)


def extract_classical_features(y, sr, n_mfcc=20):
    """Vraca fiksni vektor feature-a (mean + std nekoliko deskriptora)
    za klasicni ML baseline (SVM / RandomForest)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    def stats(arr):
        return [np.mean(arr), np.std(arr)]

    feats = []
    for arr in (mfcc, chroma, spec_centroid, spec_rolloff, zcr):
        for row in np.atleast_2d(arr):
            feats.extend(stats(row))
    feats.append(float(tempo))
    return np.array(feats, dtype=np.float32)
