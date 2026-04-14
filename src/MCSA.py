from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _to_str_list(items) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in list(items)]


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str], Path]:
    src_npz = Path(__file__).with_name("motor_current_data.npz")
    cwd_npz = Path.cwd() / "motor_current_data.npz"

    print(f"Current working directory: {Path.cwd()}")
    print(f"Exists in src/: {src_npz.exists()} | Exists in cwd/: {cwd_npz.exists()}")

    if src_npz.exists():
        chosen = src_npz
    elif cwd_npz.exists():
        chosen = cwd_npz
    else:
        raise FileNotFoundError("Could not find motor_current_data.npz in src/ or cwd")

    print(f"Loading dataset from: {chosen.resolve()}")
    data = np.load(chosen, allow_pickle=True)

    sequences = data["sequences"] if "sequences" in data else data["X"]
    labels = data["labels"] if "labels" in data else data["y"]
    class_names = _to_str_list(data["class_names"])

    return sequences.astype(np.float32), labels.astype(np.int64), class_names, chosen


def select_representative_indices(
    sequences: np.ndarray, labels: np.ndarray, class_ids: np.ndarray
) -> list[int]:
    selected = []
    for class_id in class_ids:
        idxs = np.where(labels == class_id)[0]
        x_c = sequences[idxs]
        rms = np.sqrt(np.mean(x_c**2, axis=1))
        ptp = np.ptp(x_c, axis=1)
        score = np.abs(rms - np.median(rms)) / (np.std(rms) + 1e-8)
        score += np.abs(ptp - np.median(ptp)) / (np.std(ptp) + 1e-8)
        selected.append(int(idxs[np.argmin(score)]))
    return selected


def plot_waveforms_same_scale(
    sequences: np.ndarray, labels: np.ndarray, class_names: list[str]
) -> None:
    class_ids = np.unique(labels)
    idxs = select_representative_indices(sequences, labels, class_ids)
    t = np.arange(sequences.shape[1])

    plt.figure(figsize=(10, 4.5))
    for idx, class_id in zip(idxs, class_ids):
        plt.plot(t, sequences[idx], linewidth=1.8, label=class_names[int(class_id)])
    plt.title("One representative waveform per class (same scale)")
    plt.xlabel("Time step")
    plt.ylabel("Current (a.u.)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    y = sequences[idxs]
    ymin, ymax = y.min(), y.max()
    pad = 0.1 * (ymax - ymin + 1e-8)
    plt.ylim(ymin - pad, ymax + pad)
    plt.tight_layout()
    plt.show()


def reshape_for_models(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lstm_transformer_input = sequences[:, :, np.newaxis]  # (batch, 128, 1)
    cnn1d_input = sequences[:, np.newaxis, :]  # (batch, 1, 128)
    return lstm_transformer_input, cnn1d_input


def stratified_split_indices(
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        n_test = n - n_train - n_val

        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train : n_train + n_val])
        test_idx.append(idx[n_train + n_val : n_train + n_val + n_test])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


class MotorLSTM(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_size: int = 32, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        final_hidden = h_n[-1]
        return self.fc(final_hidden)


class MotorCNN1D(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pos_embed[:, :seq_len, :]


class MotorTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 32,
        max_len: int = 128,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional = LearnedPositionalEncoding(max_len=max_len, d_model=d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.positional(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def first_layer_attention_weights(
        self, x: torch.Tensor, average_attn_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = self.positional(x)
        first_layer = self.encoder.layers[0]
        attn_out, attn_weights = first_layer.self_attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=average_attn_weights,
        )
        return attn_out, attn_weights


def build_loaders(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    train_ds = TensorDataset(X_t[train_idx], y_t[train_idx])
    val_ds = TensorDataset(X_t[val_idx], y_t[val_idx])
    test_ds = TensorDataset(X_t[test_idx], y_t[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        if is_train:
            loss.backward()
            optimizer.step()

        bsz = yb.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy_from_logits(logits, yb) * bsz
        total_n += bsz

    return total_loss / total_n, total_acc / total_n


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    model_name: str,
) -> dict[str, list[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    print(f"\nTraining {model_name} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[{model_name}] epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
            )

    return history


def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            ys.append(yb.numpy())
            ps.append(preds)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return y_true, y_pred


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def summarize_test_results(
    cms: dict[str, np.ndarray], class_names: list[str]
) -> None:
    print("\n=== Test Accuracy Summary ===")

    per_class_by_model: dict[str, np.ndarray] = {}
    overall_by_model: dict[str, float] = {}

    for model_name, cm in cms.items():
        cm = cm.astype(np.float64)
        total = cm.sum()
        overall = float(np.trace(cm) / (total + 1e-8))
        per_class = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
        overall_by_model[model_name] = overall
        per_class_by_model[model_name] = per_class

        print(f"{model_name}: overall test accuracy = {overall:.4f}")
        for i, cname in enumerate(class_names):
            print(f"  class {i} ({cname}) accuracy = {per_class[i]:.4f}")

    print("\nBest architecture by class:")
    model_names = list(cms.keys())
    stacked = np.vstack([per_class_by_model[m] for m in model_names])
    best_idx = np.argmax(stacked, axis=0)
    for i, cname in enumerate(class_names):
        m = model_names[int(best_idx[i])]
        print(f"  {cname}: {m} ({stacked[int(best_idx[i]), i]:.4f})")


def plot_training_histories(histories: dict[str, dict[str, list[float]]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    model_names = list(histories.keys())

    for j, name in enumerate(model_names):
        h = histories[name]
        axes[0, j].plot(h["train_loss"], label="train")
        axes[0, j].plot(h["val_loss"], label="val")
        axes[0, j].set_title(f"{name} loss")
        axes[0, j].grid(True, alpha=0.3)
        if j == 0:
            axes[0, j].set_ylabel("Cross-entropy")
            axes[0, j].legend()

        axes[1, j].plot(h["train_acc"], label="train")
        axes[1, j].plot(h["val_acc"], label="val")
        axes[1, j].set_title(f"{name} accuracy")
        axes[1, j].grid(True, alpha=0.3)
        axes[1, j].set_xlabel("Epoch")
        if j == 0:
            axes[1, j].set_ylabel("Accuracy")

    fig.suptitle("Training curves (80 epochs)")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(cms: dict[str, np.ndarray], class_names: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    model_names = list(cms.keys())

    for ax, name in zip(axes, model_names):
        cm = cms[name].astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-8
        cm_norm = cm / row_sums
        im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"{name} test confusion")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=20)
        ax.set_yticklabels(class_names)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

    fig.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.9, label="Row-normalized fraction"
    )
    plt.tight_layout()
    plt.show()


def plot_transformer_attention_heatmaps(
    model: MotorTransformerEncoder,
    X_seq: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    device: torch.device,
) -> None:
    model.eval()

    def _first_index_of_class(class_id: int) -> int:
        idx = np.where(y == class_id)[0]
        if idx.size == 0:
            raise ValueError(f"No samples found for class_id={class_id}")
        return int(idx[0])

    idx_bearing = _first_index_of_class(1)
    idx_winding = _first_index_of_class(2)
    idxs = [idx_bearing, idx_winding]
    titles = [
        f"Attention: {class_names[1]} (sample {idx_bearing})",
        f"Attention: {class_names[2]} (sample {idx_winding})",
    ]

    maps = []
    with torch.no_grad():
        for idx in idxs:
            x = torch.from_numpy(X_seq[idx : idx + 1]).to(device)
            _, attn_w = model.first_layer_attention_weights(
                x, average_attn_weights=False
            )
            attn_map = attn_w[0].mean(dim=0).cpu().numpy()
            maps.append(attn_map)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for ax, m, title in zip(axes, maps, titles):
        im = ax.imshow(m, aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Key time step")
        ax.set_ylabel("Query time step")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, label="Attention weight")
    plt.tight_layout()
    plt.show()


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")

    sequences, labels, class_names, _ = load_dataset()
    print(f"Dataset shapes -> sequences: {sequences.shape}, labels: {labels.shape}")

    plot_waveforms_same_scale(sequences, labels, class_names)

    X_seq, X_cnn = reshape_for_models(sequences)
    print(f"LSTM/Transformer input shape: {X_seq.shape}")
    print(f"1D CNN input shape: {X_cnn.shape}")

    train_idx, val_idx, test_idx = stratified_split_indices(
        labels, 0.70, 0.15, 0.15, seed=42
    )
    print(
        f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
    )

    epochs = 80
    lr = 1e-3

    lstm_train, lstm_val, lstm_test = build_loaders(
        X_seq, labels, train_idx, val_idx, test_idx, batch_size=64
    )
    lstm_model = MotorLSTM(input_dim=1, hidden_size=32, num_classes=3).to(device)
    lstm_hist = train_model(
        lstm_model,
        lstm_train,
        lstm_val,
        epochs=epochs,
        lr=lr,
        device=device,
        model_name="LSTM",
    )

    cnn_train, cnn_val, cnn_test = build_loaders(
        X_cnn, labels, train_idx, val_idx, test_idx, batch_size=64
    )
    cnn_model = MotorCNN1D(num_classes=3).to(device)
    cnn_hist = train_model(
        cnn_model,
        cnn_train,
        cnn_val,
        epochs=epochs,
        lr=lr,
        device=device,
        model_name="CNN1D",
    )

    tr_train, tr_val, tr_test = build_loaders(
        X_seq, labels, train_idx, val_idx, test_idx, batch_size=64
    )
    transformer_model = MotorTransformerEncoder(
        input_dim=1, d_model=32, max_len=X_seq.shape[1], num_classes=3
    ).to(device)
    tr_hist = train_model(
        transformer_model,
        tr_train,
        tr_val,
        epochs=epochs,
        lr=lr,
        device=device,
        model_name="Transformer",
    )

    histories = {
        "LSTM": lstm_hist,
        "CNN1D": cnn_hist,
        "Transformer": tr_hist,
    }
    plot_training_histories(histories)

    y_true_l, y_pred_l = predict_all(lstm_model, lstm_test, device)
    y_true_c, y_pred_c = predict_all(cnn_model, cnn_test, device)
    y_true_t, y_pred_t = predict_all(transformer_model, tr_test, device)

    cms = {
        "LSTM": confusion_matrix(y_true_l, y_pred_l, num_classes=3),
        "CNN1D": confusion_matrix(y_true_c, y_pred_c, num_classes=3),
        "Transformer": confusion_matrix(y_true_t, y_pred_t, num_classes=3),
    }
    summarize_test_results(cms, class_names)
    plot_confusion_matrices(cms, class_names)

    plot_transformer_attention_heatmaps(
        transformer_model,
        X_seq[test_idx],
        labels[test_idx],
        class_names,
        device,
    )


if __name__ == "__main__":
    main()
