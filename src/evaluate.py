"""Evaluira sacuvani CNN checkpoint na test splitu."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from config import load_config, MODELS_SAVED_DIR, RESULTS_DIR
from datasets import GTZANSpectrogramDataset
from models.cnn import GenreCNN


def run():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    genres = cfg["data"]["genres"]
    processed_dir = cfg["data"]["processed_dir"]

    test_ds = GTZANSpectrogramDataset(
        f"{processed_dir}/test_manifest.csv", genres,
        sample_rate=cfg["data"]["sample_rate"],
        segment_duration=cfg["data"]["segment_duration"],
        n_mels=cfg["data"]["n_mels"],
        deterministic=True,
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"])

    model = GenreCNN(n_genres=len(genres), n_mels=cfg["data"]["n_mels"]).to(device)
    model.load_state_dict(torch.load(MODELS_SAVED_DIR / "genre_cnn_best.pt", map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu()
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    print(classification_report(y_true, y_pred, target_names=genres))
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=genres, yticklabels=genres)
    plt.xlabel("Predvidjeni zanr")
    plt.ylabel("Stvarni zanr")
    plt.title("CNN - confusion matrica (test skup)")
    plt.tight_layout()
    cm_path = RESULTS_DIR / "cnn_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrica sacuvana: {cm_path}")


if __name__ == "__main__":
    run()
