"""Evaluira sacuvani CNN checkpoint na test splitu."""
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from config import load_config
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
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"])

    model = GenreCNN(n_genres=len(genres), n_mels=cfg["data"]["n_mels"]).to(device)
    model.load_state_dict(torch.load("../models_saved/genre_cnn.pt", map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu()
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    print(classification_report(y_true, y_pred, target_names=genres))
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    run()
