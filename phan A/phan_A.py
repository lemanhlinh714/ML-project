import numpy as np
import pandas as pd


# =========================================================
# ĐƯỜNG DẪN FILE
# =========================================================

lp4_path = "phan A/data/lp4.data"
lp5_path = "phan A/data/lp5.data"


# =========================================================
# HÀM ĐỌC DỮ LIỆU
# =========================================================

def load_robot_data(file_path):

    samples = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    n = len(lines)

    while i < n:

        # bỏ dòng trống
        if lines[i] == "":
            i += 1
            continue

        # class label
        label = lines[i]
        labels.append(label)

        i += 1

        sample = []

        # đọc 15 dòng tiếp theo
        while i < n and len(sample) < 15:

            if lines[i] == "":
                i += 1
                continue

            values = list(map(int, lines[i].split()))

            # mỗi dòng phải có 6 giá trị
            if len(values) == 6:
                sample.append(values)

            i += 1

        # chỉ giữ sample đủ 15 dòng
        if len(sample) == 15:
            samples.append(np.array(sample))
        else:
            labels.pop()

    return samples, labels


# =========================================================
# VECTOR 90 CHIỀU
# =========================================================

def flatten_features(sample):
    """
    (15,6) -> vector 90 chiều
    """
    return sample.flatten()


# =========================================================
# LOAD LP4 + LP5
# =========================================================

samples_lp4, labels_lp4 = load_robot_data(lp4_path)
samples_lp5, labels_lp5 = load_robot_data(lp5_path)

print("LP4 samples:", len(samples_lp4))
print("LP5 samples:", len(samples_lp5))

# ghép dữ liệu
all_samples = samples_lp4 + samples_lp5
all_labels = labels_lp4 + labels_lp5

print("Total samples:", len(all_samples))


# =========================================================
# TẠO FEATURE VECTOR
# =========================================================

X = np.array([flatten_features(s) for s in all_samples])
y = np.array(all_labels)

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# =========================================================
# TẠO DATAFRAME
# =========================================================

signals = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

columns = []

for t in range(1, 16):
    for s in signals:
        columns.append(f"{s}{t}")

df = pd.DataFrame(X, columns=columns)

# thêm label
df["label"] = y


# =========================================================
# LƯU CSV
# =========================================================

output_file = "robot_flat_90.csv"

df.to_csv(output_file, index=False)

print("\nSaved:", output_file)


# =========================================================
# IN THỬ DỮ LIỆU
# =========================================================

print("\n===== SAMPLE DATA =====")
print(df.head())


# =========================================================
# ROBOT CLASSIFICATION PIPELINE
# SAVE FULL REPORT OUTPUT (FIGURES + TABLES)
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =========================================================
# CREATE OUTPUT FOLDER
# =========================================================

os.makedirs("results", exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("robot_flat_90.csv")

X = df.drop(columns=["label"]).values
y = df["label"].values

y_encoded, class_names = pd.factorize(y)

# =========================================================
# SPLIT DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

# =========================================================
# DECISION TREE (3 CONFIGS)
# =========================================================

dt_configs = [
    {"max_depth": 3, "criterion": "gini", "min_samples_leaf": 1},
    {"max_depth": 5, "criterion": "gini", "min_samples_leaf": 5},
    {"max_depth": 10, "criterion": "entropy", "min_samples_leaf": 2},
]

dt_results = []

for cfg in dt_configs:
    model = DecisionTreeClassifier(
        max_depth=cfg["max_depth"],
        criterion=cfg["criterion"],
        min_samples_leaf=cfg["min_samples_leaf"],
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    dt_results.append({
        **cfg,
        "accuracy": acc,
        "pred": preds
    })

dt_df = pd.DataFrame(dt_results)

# SAVE TABLE
dt_df.drop(columns=["pred"]).to_csv("results/decision_tree_results.csv", index=False)

# =========================================================
# ANN MODEL
# =========================================================

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=32,
    shuffle=True
)

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_ann(hidden_dim, lr, dropout, epochs=20):

    model = ANN(X_train.shape[1], hidden_dim, dropout, len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = torch.argmax(model(X_test_t), dim=1)
        acc = (preds == y_test_t).float().mean().item()

    return acc, preds.numpy()

# =========================================================
# ANN CONFIGS
# =========================================================

ann_configs = [
    {"hidden": 64, "lr": 0.01, "dropout": 0.1},
    {"hidden": 128, "lr": 0.001, "dropout": 0.3},
    {"hidden": 256, "lr": 0.005, "dropout": 0.5},
]

ann_results = []

for cfg in ann_configs:
    acc, preds = train_ann(cfg["hidden"], cfg["lr"], cfg["dropout"])

    ann_results.append({
        **cfg,
        "accuracy": acc,
        "pred": preds
    })

ann_df = pd.DataFrame(ann_results)

ann_df.drop(columns=["pred"]).to_csv("results/ann_results.csv", index=False)

# =========================================================
# FINAL TABLE
# =========================================================

summary = pd.concat([
    dt_df.drop(columns=["pred"]),
    ann_df.drop(columns=["pred"])
], ignore_index=True)

summary.to_csv("results/summary.csv", index=False)

# =========================================================
# 1. CONFUSION MATRIX (BEST ANN)
# =========================================================

best_idx = ann_df["accuracy"].idxmax()
best_preds = ann_df.loc[best_idx, "pred"]

cm = confusion_matrix(y_test, best_preds)

plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - Best ANN")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("True")

plt.savefig("results/confusion_matrix.png")
plt.close()

# =========================================================
# 2. DROPOUT vs ACCURACY
# =========================================================

dropouts = [c["dropout"] for c in ann_configs]
accs = [r["accuracy"] for r in ann_results]

plt.figure()
plt.plot(dropouts, accs, marker="o")
plt.title("Dropout vs Accuracy")
plt.xlabel("Dropout")
plt.ylabel("Accuracy")

plt.savefig("results/dropout_vs_accuracy.png")
plt.close()

# =========================================================
# DONE
# =========================================================

print("\nALL RESULTS SAVED TO /results")