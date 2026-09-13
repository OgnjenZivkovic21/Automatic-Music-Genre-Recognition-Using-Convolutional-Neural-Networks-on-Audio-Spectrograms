"""Jednokratno: izvlaci GTZAN audio iz preuzetog Kaggle zip-a
(data.zip, sadrzi 'Data/genres_original/...') u data/raw/gtzan/genres_original/.
Radi u vremenski ogranicenim "talasima" (da ne pregazi timeout shell poziva) -
preskace vec izvucene fajlove ispravne velicine, pa je bezbedno vise puta
pozvati dok se ne izvuku svi."""
import zipfile
import time
import sys
from pathlib import Path

ZIP_PATH = Path("data.zip")
PREFIX = "Data/genres_original/"
DEST_ROOT = Path("data/raw/gtzan/genres_original")
TIME_BUDGET_SEC = 100


def extract():
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    count, skipped = 0, 0
    with zipfile.ZipFile(ZIP_PATH) as z:
        infos = [i for i in z.infolist() if not i.is_dir() and i.filename.startswith(PREFIX)]
        total = len(infos)
        for info in infos:
            rel = info.filename[len(PREFIX):]
            dest = DEST_ROOT / rel
            if dest.exists() and dest.stat().st_size == info.file_size:
                skipped += 1
                continue
            if time.time() - start > TIME_BUDGET_SEC:
                print(f"[stao zbog vremena] izvuceno {count} u ovom pozivu, "
                      f"preskoceno {skipped} vec gotovih, ukupno u zipu {total}", flush=True)
                sys.exit(2)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            count += 1
    print(f"[gotovo] izvuceno {count} novih, preskoceno {skipped} vec postojecih, ukupno {total}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    extract()
