"""Evaluira sacuvani CNN checkpoint na test splitu.

Test-time averaging: za svaku test pesmu se uzima N_CROPS = 5 ravnomerno
rasporedjenih isecaka cele pesme (bez nasumicnosti), za svaki se izracuna
softmax verovatnoca po zanru, pa se te verovatnoce usrednje - konacna
predikcija je zanr sa najvecom usrednjenom verovatnocom. Ovo je stabilnije
od stare verzije koja je pravila predikciju na osnovu samo JEDNOG nasumicnog
3-sekundnog isecka po pesmi."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import load_config, MODELS_SAVED_DIR, RESULTS_DIR
from datasets import GTZANSpectrogramDataset
from models.cnn import GenreCNN

N_CROPS = 5


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

    model = GenreCNN(n_genres=len(genres), n_mels=cfg["data"]["n_mels"]).to(device)
    model.load_state_dict(torch.load(MODELS_SAVED_DIR / "genre_cnn_best.pt", map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for idx in range(len(test_ds.rows)):
            x, label = test_ds.get_eval_crops(idx, n_crops=N_CROPS)
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1)
            avg_probs = probs.mean(dim=0)
            pred = avg_probs.argmax().item()
            y_true.append(label)
            y_pred.append(pred)

    print(f"(test-time averaging, {N_CROPS} isecaka po pesmi)")
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
