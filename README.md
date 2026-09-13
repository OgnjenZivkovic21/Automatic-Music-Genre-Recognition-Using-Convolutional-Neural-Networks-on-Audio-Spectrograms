# Prepoznavanje zanra muzike primenom masinskog ucenja

Diplomski rad - poredjenje klasicnog masinskog ucenja (SVM / Random Forest na
rucno izvucenim audio feature-ima) i dubokog ucenja (CNN na mel-spektrogramima)
za automatsku klasifikaciju muzickih zanrova.

## Struktura projekta

```
data/
  raw/gtzan/genres_original/<zanr>/<zanr>.000xx.wav   <- ovde ide GTZAN (preuzimas rucno, vidi ispod)
  raw/fma_small/                                       <- opciono, kasnije, za proveru generalizacije
  processed/                                           <- manifest CSV-ovi, keširani feature-i (generise se)
src/
  config.py            - ucitava configs/config.yaml
  features.py           - mel-spektrogram (za CNN) i klasicni feature-i (za SVM/RF)
  data_prep.py           - pravi train/val/test split (po pesmi, ne po isecku - bez curenja podataka)
  datasets.py             - PyTorch Dataset za spektrograme
  models/cnn.py            - CNN arhitektura
  models/baseline.py        - SVM / RandomForest pipeline
  train_baseline.py          - trening klasicnog ML baseline-a
  train_cnn.py                 - trening CNN-a
  evaluate.py                   - evaluacija sacuvanog CNN modela na test skupu
configs/config.yaml    - svi hiperparametri i putanje na jednom mestu
models_saved/            - sacuvani checkpoint-i (generise se)
```

## Preuzimanje GTZAN dataset-a (radi se rucno - vidi objasnjenje u chatu)

1. Preuzmi GTZAN (npr. sa Kaggle-a: "GTZAN Dataset - Music Genre Classification",
   autor andradaolteanu) - zip fajl od ~1.2GB.
2. Raspakuj ga tako da dobijes strukturu:
   `data/raw/gtzan/genres_original/blues/blues.00000.wav`, ... (10 foldera, po zanru).
3. Proveri da folder odgovara putanji iz `configs/config.yaml` (`data.raw_dir`).

## Pokretanje

```bash
python3 -m venv .venv
source .venv/bin/activate        # na Windows-u: .venv\Scripts\activate
pip install -r requirements.txt

cd src
python3 data_prep.py             # pravi train/val/test manifest CSV-ove
python3 train_baseline.py        # SVM + RandomForest baseline
python3 train_cnn.py             # CNN na mel-spektrogramima
python3 evaluate.py              # evaluacija CNN-a na test skupu
```
