"""Pravi manifest (putanja, zanr, track_id) iz GTZAN foldera i deli ga
na train/val/test PO PESMI (ne po iseccima) da bi se izbeglo curenje
podataka (data leakage) - ista pesma se nikad ne nalazi u dva razlicita splita.

Ocekivana struktura (GTZAN, nakon preuzimanja i raspakivanja):
  data/raw/gtzan/genres_original/<zanr>/<zanr>.000xx.wav

Napomena: GTZAN sadrzi jedan poznato osteceni fajl (jazz.00054.wav - pogresan
WAV header). build_manifest ga automatski preskace uz upozorenje; ovo je
poznato ogranicenje dataset-a i vredi ga pomenuti u radu.
"""
import csv
import wave
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import load_config


def is_readable_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() > 0
    except Exception:
        return False


def build_manifest(raw_dir: Path, genres: list) -> list:
    manifest = []
    skipped = []
    for genre in genres:
        genre_dir = raw_dir / genre
        if not genre_dir.exists():
            print(f"[upozorenje] fali folder za zanr: {genre_dir}")
            continue
        for wav_path in sorted(genre_dir.glob("*.wav")):
            if not is_readable_wav(wav_path):
                skipped.append(str(wav_path))
                continue
            manifest.append({
                "path": str(wav_path),
                "genre": genre,
                "track_id": wav_path.stem,
            })
    if skipped:
        print(f"[upozorenje] preskoceno {len(skipped)} ostecenih/necitljivih fajlova:")
        for p in skipped:
            print(f"  - {p}")
    return manifest


def split_manifest(manifest: list, test_size: float, val_size: float, seed: int):
    train_val, test = train_test_split(
        manifest, test_size=test_size,
        stratify=[m["genre"] for m in manifest], random_state=seed
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val,
        stratify=[m["genre"] for m in train_val], random_state=seed
    )
    return train, val, test


def write_csv(rows: list, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "genre", "track_id"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    manifest = build_manifest(raw_dir, cfg["data"]["genres"])
    print(f"Pronadjeno {len(manifest)} validnih audio fajlova.")

    if not manifest:
        print("Nema fajlova - da li si preuzeo i raspakovao GTZAN u data/raw/gtzan/genres_original/?")
    else:
        train, val, test = split_manifest(
            manifest,
            test_size=cfg["split"]["test_size"],
            val_size=cfg["split"]["val_size"],
            seed=cfg["seed"],
        )
        processed_dir = Path(cfg["data"]["processed_dir"])
        write_csv(train, processed_dir / "train_manifest.csv")
        write_csv(val, processed_dir / "val_manifest.csv")
        write_csv(test, processed_dir / "test_manifest.csv")
        print(f"train={len(train)} val={len(val)} test={len(test)}")
