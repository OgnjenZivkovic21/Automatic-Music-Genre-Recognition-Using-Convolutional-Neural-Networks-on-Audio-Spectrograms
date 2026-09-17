"""Trenira CNN na mel-spektrogramima."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import load_config, MODELS_SAVED_DIR, RESULTS_DIR
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
    # Learning rate scheduler: na svakih 15 epoha prepolovi lr (npr. 0.001 -> 0.0005 -> 0.00025)
    # cilj: da se trening "smiri" pred kraj umesto da lr ostane fiksan kroz svih 40 epoha
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    MODELS_SAVED_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    train_losses = []
    val_accs = []

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

        val_acc = correct / total
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch+1}/{cfg['train']['epochs']} "
              f"train_loss={train_loss:.4f} "
              f"val_acc={val_acc:.4f} "
              f"lr={current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODELS_SAVED_DIR / "genre_cnn_best.pt")
            print(f"  -> novi najbolji model sacuvan (val_acc={val_acc:.4f})")

        scheduler.step()

    torch.save(model.state_dict(), MODELS_SAVED_DIR / "genre_cnn_last.pt")
    print(f"Trening zavrsen. Najbolji val_acc={best_val_acc:.4f} "
          f"(genre_cnn_best.pt), poslednja epoha sacuvana kao genre_cnn_last.pt")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(epochs_range, train_losses, marker="o")
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("CNN trening - loss i validaciona tacnost po epohama")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, val_accs, marker="o", color="tab:orange")
    axes[1].set_ylabel("Validaciona tacnost")
    axes[1].set_xlabel("Epoha")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "cnn_training_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Graf sacuvan: {plot_path}")


if __name__ == "__main__":
    run()
