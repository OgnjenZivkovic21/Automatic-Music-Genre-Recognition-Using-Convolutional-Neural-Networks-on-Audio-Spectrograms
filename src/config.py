"""Ucitava configs/config.yaml u obican dict."""
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
MODELS_SAVED_DIR = PROJECT_ROOT / "models_saved"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Putanje u config.yaml su relativne prema korenu projekta (PROJECT_ROOT),
    # ne prema folderu iz kog se skripta pokrece - ovde ih pretvaramo u
    # apsolutne da bi skripte radile identicno bez obzira odakle se pokrecu.
    cfg["data"]["raw_dir"] = str(PROJECT_ROOT / cfg["data"]["raw_dir"])
    cfg["data"]["processed_dir"] = str(PROJECT_ROOT / cfg["data"]["processed_dir"])

    return cfg


if __name__ == "__main__":
    print(load_config())
