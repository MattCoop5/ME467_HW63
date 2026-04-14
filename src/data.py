import numpy as np


def generate_motor_data(n_per_class=400, seq_len=128, seed=42):
    """Generate synthetic motor current sequences.
    Three classes:
    - Healthy: clean sinusoid + noise, optional load variation
    - Bearing wear: sinusoid + subtle high-frequency ripple + noise
    - Winding fault: slightly asymmetric sinusoid + noise
    The anomalies are intentionally small relative to the noise,
    making visual inspection unreliable.
    """
    rng = np.random.default_rng(seed)
    n_cycles = 4  # ~4 electrical cycles in 128 samples
    t = np.linspace(0, 2 * np.pi * n_cycles, seq_len)
    sequences = []
    labels = []
    class_names = ["healthy", "bearing_wear", "winding_fault"]
    for _ in range(n_per_class):
        # --- Healthy ---
        A = rng.uniform(0.8, 1.2)
        # Optional load variation (smooth amplitude envelope)
        load_env = 1.0 + rng.uniform(-0.15, 0.15) * np.sin(t * rng.uniform(0.05, 0.2))
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0, rng.uniform(0.05, 0.08), seq_len)
        current = A * load_env * np.sin(t + phase) + noise
        sequences.append(current.astype(np.float32))
        labels.append(0)
    for _ in range(n_per_class):
        # --- Bearing wear: high-frequency ripple ---
        A = rng.uniform(0.8, 1.2)
        load_env = 1.0 + rng.uniform(-0.15, 0.15) * np.sin(t * rng.uniform(0.05, 0.2))
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0, rng.uniform(0.05, 0.08), seq_len)
        # Ripple: high frequency, small amplitude (smaller than noise)
        ripple_freq = rng.uniform(15, 25)  # much higher than the 4-cycle base
        ripple_amp = rng.uniform(0.10, 0.20)
        ripple = ripple_amp * np.sin(ripple_freq * t + rng.uniform(0, 2 * np.pi))
        current = A * load_env * np.sin(t + phase) + ripple + noise
        sequences.append(current.astype(np.float32))
        labels.append(1)
    for _ in range(n_per_class):
        # --- Winding fault: asymmetric peaks ---
        A = rng.uniform(0.8, 1.2)
        load_env = 1.0 + rng.uniform(-0.15, 0.15) * np.sin(t * rng.uniform(0.05, 0.2))
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0, rng.uniform(0.05, 0.08), seq_len)
        # Asymmetry: add a small rectified component
        asymmetry = rng.uniform(0.15, 0.30)
        base = np.sin(t + phase)
        current = A * load_env * base + asymmetry * np.maximum(0, base) + noise
        sequences.append(current.astype(np.float32))
        labels.append(2)
    sequences = np.array(sequences)  # (600, 128)
    labels = np.array(labels, dtype=np.int64)
    # Shuffle
    idx = rng.permutation(len(sequences))
    sequences = sequences[idx]
    labels = labels[idx]
    return sequences, labels, class_names


sequences, labels, class_names = generate_motor_data(n_per_class=400, seed=42)
print(f"Sequences: {sequences.shape}")
print(f"Labels: {labels.shape}, classes: {class_names}")
print(f"Class distribution: {[int((labels == i).sum()) for i in range(3)]}")
# Save
np.savez(
    "motor_current_data.npz",
    sequences=sequences,
    labels=labels,
    class_names=class_names,
)
print("Saved motor_current_data.npz")
