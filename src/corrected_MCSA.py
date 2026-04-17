import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

data = np.load("motor_current_data.npz")
sequences = data["sequences"]  # (600, 128), float32
labels = data["labels"]  # (600,), int64 in {0, 1, 2}
class_names = list(data["class_names"])
print(f"Sequences: {sequences.shape}")
print(f"Classes: {class_names}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3))
for c in range(3):
    idx = np.where(labels == c)[0][0]
    axes[c].plot(sequences[idx], color="#4c78a8", lw=0.8)
    axes[c].set_title(class_names[c], fontsize=11)
    axes[c].set_xlabel("Sample")
    axes[c].set_ylim(-1.8, 1.8)
    if c == 0:
        axes[c].set_ylabel("Current (A)")
    axes[c].spines["top"].set_visible(False)
    axes[c].spines["right"].set_visible(False)
fig.suptitle("Example waveforms (anomalies are subtle)", fontsize=12)
fig.tight_layout()
plt.show()

rng = np.random.default_rng(0)
idx = rng.permutation(len(sequences))
n_test = int(0.15 * len(sequences))
n_val = int(0.15 * len(sequences))
X_test = sequences[idx[:n_test]]
y_test = labels[idx[:n_test]]
X_val = sequences[idx[n_test : n_test + n_val]]
y_val = labels[idx[n_test : n_test + n_val]]
X_train = sequences[idx[n_test + n_val :]]
y_train = labels[idx[n_test + n_val :]]
# For RNN/transformer: (batch, 128, 1)
X_train_seq = torch.tensor(X_train).unsqueeze(-1)
X_val_seq = torch.tensor(X_val).unsqueeze(-1)
X_test_seq = torch.tensor(X_test).unsqueeze(-1)
# For 1D CNN: (batch, 1, 128) — channel first
X_train_cnn = torch.tensor(X_train).unsqueeze(1)
X_val_cnn = torch.tensor(X_val).unsqueeze(1)
X_test_cnn = torch.tensor(X_test).unsqueeze(1)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)
train_seq_loader = DataLoader(
    TensorDataset(X_train_seq, y_train_t), batch_size=32, shuffle=True
)
train_cnn_loader = DataLoader(
    TensorDataset(X_train_cnn, y_train_t), batch_size=32, shuffle=True
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")


class MotorLSTM(nn.Module):
    def __init__(self, hidden_size=32, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, 128, 32)
        h_last = out[:, -1, :]  # final hidden state
        return self.fc(h_last)


class MotorCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global average pooling -> (batch, 64, 1)
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x).squeeze(-1)  # (batch, 64)
        return self.fc(x)


class MotorTransformer(nn.Module):
    def __init__(
        self,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        seq_len=128,
        n_classes=3,
    ):
        super().__init__()
        # Project 1D input to d_model dimensions
        self.input_proj = nn.Linear(1, d_model)
        # Learned positional encoding (one vector per time step)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classification head
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (batch, 128, 1)
        x = self.input_proj(x)  # (batch, 128, 32)
        x = x + self.pos_encoding  # add positional encoding
        x = self.encoder(x)  # (batch, 128, 32) — self-attention
        x = x.mean(dim=1)  # mean-pool over time -> (batch, 32)
        return self.fc(x)


def train_model(model, train_loader, X_val, y_val, epochs=80, lr=1e-3):
    """Train and return (train_losses, val_losses, val_accs)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, val_accs = [], [], []
    # Epoch 0: before training
    model.eval()
    with torch.no_grad():
        vp = model(X_val)
        vl = criterion(vp, y_val).item()
        va = (vp.argmax(dim=1) == y_val).float().mean().item()
    train_losses.append(vl)
    val_losses.append(vl)
    val_accs.append(va)
    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for X_batch, y_batch in train_loader:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            val_acc = (val_pred.argmax(dim=1) == y_val).float().mean().item()
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return train_losses, val_losses, val_accs


import time

results = {}
for name, ModelClass, loader, X_v, epochs in [
    ("LSTM", MotorLSTM, train_seq_loader, X_val_seq, 50),
    ("1D CNN", MotorCNN, train_cnn_loader, X_val_cnn, 50),
    ("Transformer", MotorTransformer, train_seq_loader, X_val_seq, 50),
]:
    torch.manual_seed(0)
    model = ModelClass()
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    tr, vl, va = train_model(model, loader, X_v, y_val_t, epochs=epochs)
    elapsed = time.time() - t0
    results[name] = {
        "model": model,
        "train": tr,
        "val": vl,
        "acc": va,
        "params": n_params,
        "time": elapsed,
    }
    print(
        f"{name:12s}: {n_params:>6,} params, val acc = {va[-1]:.1%}, "
        f"time = {elapsed:.1f}s"
    )

print(f"\n{'Model':12s}  {'Params':>8s}  {'Test Acc':>8s}")
print("-" * 35)
for name, r in results.items():
    model = r["model"]
    model.eval()
    X_te = X_test_cnn if name == "1D CNN" else X_test_seq
    with torch.no_grad():
        test_pred = model(X_te).argmax(dim=1)
        test_acc = (test_pred == y_test_t).float().mean().item()
    r["test_acc"] = test_acc
    print(f"{name:12s}  {r['params']:>8,}  {test_acc:>8.1%}")

print(f"\n{'Model':12s}  {'Healthy':>8s}  {'Bearing':>8s}  {'Winding':>8s}")
print("-" * 45)
for name, r in results.items():
    model = r["model"]
    model.eval()
    X_te = X_test_cnn if name == "1D CNN" else X_test_seq
    with torch.no_grad():
        preds = model(X_te).argmax(dim=1).numpy()
    for c in range(3):
        mask = y_test == c
        acc = (preds[mask] == y_test[mask]).mean()
        print(f"  {class_names[c]:>13s}: {acc:.1%}", end="")
    print(f"  [{name}]")
