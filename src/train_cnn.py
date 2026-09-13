"""Trenira CNN na mel-spektrogramima."""
import os
import torch
from torch.utils.data import DataLoader

from config import load_config
from utils import set_seed
from datasets import GTZANSpectrogramDataset
from models.cnn import GenreCNN


def run():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    genres = cfg["data"]["genres"]
    processed_dir = cfg["data"]["processed_dir"]

    train_ds = GTZANSpectrogramDataset(
        f"{processed_dir}/train_manifest.csv", genres,
        sample_rate=cfg["data"]["sample_rate"],
        segment_duration=cfg["data"]["segment_duration"],
        n_mels=cfg["data"]["n_mels"],
    )
    val_ds = GTZANSpectrogramDataset(
        f"{processed_dir}/val_manifest.csv", genres,
        sample_rate=cfg["data"]["sample_rate"],
        segment_duration=cfg["data"]["segment_duration"],
        n_mels=cfg["data"]["n_mels"],
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    model = GenreCNN(n_genres=len(genres), n_mels=cfg["data"]["n_mels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        print(f"epoch {epoch+1}/{cfg['train']['epochs']} "
              f"train_loss={total_loss/len(train_loader):.4f} "
              f"val_acc={correct/total:.4f}")

    os.makedirs("../models_saved", exist_ok=True)
    torch.save(model.state_dict(), "../models_saved/genre_cnn.pt")


if __name__ == "__main__":
    run()
