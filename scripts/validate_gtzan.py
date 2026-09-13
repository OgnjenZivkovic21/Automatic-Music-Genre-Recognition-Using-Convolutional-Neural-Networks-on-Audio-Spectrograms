import wave
from pathlib import Path

DEST_ROOT = Path("data/raw/gtzan/genres_original")

def validate(min_duration=5.0):
    bad = []
    total = 0
    for wav_path in sorted(DEST_ROOT.rglob("*.wav")):
        total += 1
        try:
            with wave.open(str(wav_path), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                duration = frames / rate if rate else 0
                if duration < min_duration:
                    bad.append((str(wav_path), f"trajanje samo {duration:.2f}s"))
        except Exception as e:
            bad.append((str(wav_path), f"greska pri citanju: {e}"))

    print(f"Provereno {total} fajlova.")
    if bad:
        print(f"Sumnjivih/ostecenih fajlova: {len(bad)}")
        for path, reason in bad:
            print(f"  - {path}: {reason}")
    else:
        print("Svi fajlovi izgledaju ispravno.")

if __name__ == "__main__":
    validate()
