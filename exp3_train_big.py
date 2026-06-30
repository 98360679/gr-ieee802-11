#!/usr/bin/env python3
"""
exp3_train_big.py — rule out capacity for the 0.52 all-replay separability.
Trains a ~10x larger CNN from scratch, many epochs, on the cached enrollment windows
(leakage-safe run-3 holdout already baked into the cache). If held-out acc stays ~0.5,
the fingerprints are genuinely ambiguous under controlled (same-content replay)
conditions, not a capacity limit.
"""
import os, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from exp3_finetune import augment, per_device_eval, show, SCRATCH
from exp3_fp_model import NUM_CLASSES, DEVICE_NAMES, n_params

CACHE = os.path.join(SCRATCH, "ftwin_enrollment.npz")   # built by the prior 6/29 all-replay run


class BigCNN(nn.Module):
    def __init__(self, nc=NUM_CLASSES):
        super().__init__()
        def b(ci, co, k):
            return nn.Sequential(nn.Conv1d(ci, co, k, padding=k // 2),
                                 nn.BatchNorm1d(co), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.features = nn.Sequential(
            b(2, 64, 7), b(64, 128, 5), b(128, 128, 5),
            b(128, 256, 3), b(256, 256, 3), b(256, 512, 3))   # 1024 -> 16
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, nc))

    def forward(self, x):
        return self.head(self.features(x))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(CACHE)
    Xtr, ytr, Xva, yva, vfid = z['Xtr'], z['ytr'], z['Xva'], z['yva'], z['vfid']
    print(f"cache {CACHE}\n  train {Xtr.shape}  val {Xva.shape}")
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=256, shuffle=True)
    Xva_t = torch.from_numpy(Xva)
    m = BigCNN(NUM_CLASSES).to(dev)
    print(f"  BigCNN params: {n_params(m):,}  (vs ~180k small)")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    EP = 80
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = 0.0; best_state = None
    for ep in range(1, EP + 1):
        m.train(); tot = nb = 0
        for xb, yb in dl:
            xb = augment(xb.to(dev)); yb = yb.to(dev)
            opt.zero_grad(); loss = lossf(m(xb), yb); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        w, f, _, _ = per_device_eval(m, dev, Xva_t, yva, vfid, NUM_CLASSES)
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:2d}  loss {tot/nb:.4f}  val_win {w:.4f}  val_frame {f:.4f}")
        if w > best:
            best = w; best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    w, f, per, C = per_device_eval(m, dev, Xva_t, yva, vfid, NUM_CLASSES)
    show("BigCNN held-out (run 3):", per, NUM_CLASSES, f)
    print(f"\n  BEST val_win {w:.4f}  val_frame {f:.4f}  (small model was 0.52)")
    print("\n  confusion (rows=true, cols=pred):")
    names = [DEVICE_NAMES[i] for i in range(NUM_CLASSES)]
    print("  true\\pred " + "".join(f"{n:>9}" for n in names))
    for i in range(NUM_CLASSES):
        print(f"  {names[i]:>9}" + "".join(f"{C[i,j]:>9}" for j in range(NUM_CLASSES)) + f"  n={C[i].sum()}")
    torch.save(best_state, "fingerprint_bigcnn_replay_all6.pt")
    print("\n  saved -> fingerprint_bigcnn_replay_all6.pt")


if __name__ == '__main__':
    main()
