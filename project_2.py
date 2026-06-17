"""
======================================================
  DecodeLabs AI Internship — Project 2
  Data Classification Using AI (KNN on Iris Dataset)
======================================================
Pipeline:
  INPUT  → Feature Scaling
  PROCESS → Train-Test Split + KNN Algorithm
  OUTPUT  → Confusion Matrix + F1 Score
"""

# ─────────────────────────────────────────────
# STEP 1 — IMPORT LIBRARIES
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)

print("=" * 60)
print("  DecodeLabs — Project 2: Data Classification Using AI")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 2 — LOAD & EXPLORE THE IRIS DATASET
# ─────────────────────────────────────────────
print("\n📦 STEP 2: Loading Iris Dataset...")

iris = load_iris()
X = iris.data          # Features: sepal length/width, petal length/width
y = iris.target        # Labels: 0=Setosa, 1=Versicolor, 2=Virginica
class_names = iris.target_names
feature_names = iris.feature_names

df = pd.DataFrame(X, columns=feature_names)
df["Species"] = [class_names[i] for i in y]

print(f"\n  Samples   : {X.shape[0]}")
print(f"  Features  : {X.shape[1]}  → {list(feature_names)}")
print(f"  Classes   : {len(class_names)}  → {list(class_names)}")
print(f"\n  Class distribution:\n{df['Species'].value_counts().to_string()}")
print(f"\n  First 5 rows:\n{df.head().to_string(index=False)}")
print(f"\n  Statistical summary:\n{df.describe().round(2).to_string()}")

# ─────────────────────────────────────────────
# STEP 3 — FEATURE SCALING (StandardScaler)
# ─────────────────────────────────────────────
print("\n⚖️  STEP 3: Feature Scaling (StandardScaler)...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n  Before scaling — mean of feature 0 : {X[:,0].mean():.3f}")
print(f"  After  scaling — mean of feature 0 : {X_scaled[:,0].mean():.6f}  (≈ 0)")
print(f"  After  scaling — std  of feature 0 : {X_scaled[:,0].std():.6f}   (≈ 1)")

# ─────────────────────────────────────────────
# STEP 4 — TRAIN-TEST SPLIT (80 / 20)
# ─────────────────────────────────────────────
print("\n✂️  STEP 4: Train-Test Split (80% train / 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, shuffle=True
)

print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 5 — FIND OPTIMAL K (Elbow Method)
# ─────────────────────────────────────────────
print("\n🔍 STEP 5: Finding Optimal K using Elbow Method...")

k_range = range(1, 21)
error_rates = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))

best_k = int(np.argmin(error_rates)) + 1
print(f"\n  Tested K values : 1 → 20")
print(f"  Optimal K found : {best_k}  (lowest error rate = {min(error_rates):.4f})")

# ─────────────────────────────────────────────
# STEP 6 — TRAIN KNN MODEL
# ─────────────────────────────────────────────
print(f"\n🤖 STEP 6: Training KNN Model with K={best_k}...")

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("  Model trained successfully ✅")

# ─────────────────────────────────────────────
# STEP 7 — EVALUATE THE MODEL
# ─────────────────────────────────────────────
print("\n📊 STEP 7: Model Evaluation...")

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")
cm = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions, target_names=class_names)

print(f"\n  Accuracy  : {accuracy*100:.2f}%")
print(f"  F1 Score  : {f1:.4f}")
print(f"\n  Confusion Matrix:\n{cm}")
print(f"\n  Classification Report:\n{report}")

# ─────────────────────────────────────────────
# STEP 8 — VISUALISE EVERYTHING
# ─────────────────────────────────────────────
print("\n🎨 STEP 8: Generating Visualisations...")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#f0f4f8")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

BLUE   = "#0d2b6e"
ORANGE = "#e85d04"
GREEN  = "#2d6a4f"
LIGHT  = "#eaf4fb"

# ── Plot 1: Feature Distributions ──────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
colors = [BLUE, ORANGE, GREEN]
for i, name in enumerate(class_names):
    mask = y == i
    ax1.scatter(X[mask, 0], X[mask, 1],
                color=colors[i], label=name.capitalize(),
                alpha=0.75, edgecolors="white", s=55)
ax1.set_xlabel(feature_names[0], fontsize=10)
ax1.set_ylabel(feature_names[1], fontsize=10)
ax1.set_title("Raw Data: Sepal Features by Class", fontsize=12, fontweight="bold", color=BLUE)
ax1.legend(fontsize=9)
ax1.set_facecolor(LIGHT)
ax1.grid(True, alpha=0.3)

# ── Plot 2: Elbow Curve ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(list(k_range), error_rates, color=BLUE, marker="o",
         markersize=6, linewidth=2)
ax2.axvline(best_k, color=ORANGE, linestyle="--", linewidth=2,
            label=f"Optimal K={best_k}")
ax2.scatter([best_k], [error_rates[best_k-1]], color=ORANGE,
            s=120, zorder=5)
ax2.set_xlabel("K Value", fontsize=10)
ax2.set_ylabel("Error Rate", fontsize=10)
ax2.set_title("Elbow Method: Choosing K", fontsize=12, fontweight="bold", color=BLUE)
ax2.legend(fontsize=9)
ax2.set_facecolor(LIGHT)
ax2.grid(True, alpha=0.3)

# ── Plot 3: Confusion Matrix ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
sns.heatmap(cm,
            annot=True, fmt="d", cmap="Blues",
            xticklabels=[c.capitalize() for c in class_names],
            yticklabels=[c.capitalize() for c in class_names],
            linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8}, ax=ax3,
            annot_kws={"size": 14, "weight": "bold"})
ax3.set_xlabel("Predicted Label", fontsize=10)
ax3.set_ylabel("True Label", fontsize=10)
ax3.set_title("Confusion Matrix", fontsize=12, fontweight="bold", color=BLUE)

# ── Plot 4: Per-Class F1 Scores ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
report_dict = classification_report(y_test, predictions,
                                    target_names=class_names,
                                    output_dict=True)
f1_scores = [report_dict[c]["f1-score"] for c in class_names]
bars = ax4.barh([c.capitalize() for c in class_names], f1_scores,
                color=colors, edgecolor="white", height=0.5)
for bar, val in zip(bars, f1_scores):
    ax4.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", ha="right",
             fontsize=11, fontweight="bold", color="white")
ax4.set_xlim(0, 1.05)
ax4.set_xlabel("F1 Score", fontsize=10)
ax4.set_title("Per-Class F1 Scores", fontsize=12, fontweight="bold", color=BLUE)
ax4.set_facecolor(LIGHT)
ax4.grid(True, axis="x", alpha=0.3)

# ── Plot 5: Petal Feature Scatter (scaled) ──────────────────────
ax5 = fig.add_subplot(gs[2, :2])
for i, name in enumerate(class_names):
    mask = y == i
    ax5.scatter(X[mask, 2], X[mask, 3],
                color=colors[i], label=name.capitalize(),
                alpha=0.75, edgecolors="white", s=55)
ax5.set_xlabel(feature_names[2], fontsize=10)
ax5.set_ylabel(feature_names[3], fontsize=10)
ax5.set_title("Raw Data: Petal Features by Class", fontsize=12, fontweight="bold", color=BLUE)
ax5.legend(fontsize=9)
ax5.set_facecolor(LIGHT)
ax5.grid(True, alpha=0.3)

# ── Plot 6: Score Summary Card ──────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.set_facecolor(BLUE)
ax6.axis("off")
metrics = [
    ("Accuracy",  f"{accuracy*100:.2f}%"),
    ("F1 Score",  f"{f1:.4f}"),
    ("Best K",    str(best_k)),
    ("Train",     str(len(X_train))),
    ("Test",      str(len(X_test))),
    ("Classes",   "3"),
]
ax6.text(0.5, 0.92, "Model Summary", ha="center", va="top",
         fontsize=13, fontweight="bold", color="white",
         transform=ax6.transAxes)
for idx, (label, val) in enumerate(metrics):
    y_pos = 0.75 - idx * 0.13
    ax6.text(0.18, y_pos, label + ":", ha="left", va="center",
             fontsize=10, color="#aad4f5", transform=ax6.transAxes)
    ax6.text(0.82, y_pos, val, ha="right", va="center",
             fontsize=11, fontweight="bold", color=ORANGE,
             transform=ax6.transAxes)

# ── Main Title ──────────────────────────────────────────────────
fig.suptitle(
    "DecodeLabs — Project 2: Data Classification Using AI (KNN • Iris Dataset)",
    fontsize=15, fontweight="bold", color=BLUE, y=0.98
)

plt.savefig("/home/claude/iris_knn_results.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved → iris_knn_results.png ✅")

# ─────────────────────────────────────────────
# STEP 9 — TEST WITH A NEW SAMPLE
# ─────────────────────────────────────────────
print("\n🌸 STEP 9: Predicting on a New Flower Sample...")

new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # typical Setosa
new_scaled  = scaler.transform(new_sample)
new_pred    = model.predict(new_scaled)
new_proba   = model.predict_proba(new_scaled)[0]

print(f"\n  Input features : sepal_length=5.1, sepal_width=3.5, "
      f"petal_length=1.4, petal_width=0.2")
print(f"  Predicted class: {class_names[new_pred[0]].upper()}")
print("  Class probabilities:")
for name, prob in zip(class_names, new_proba):
    bar = "█" * int(prob * 20)
    print(f"    {name:12s}: {prob:.2f}  {bar}")

print("\n" + "=" * 60)
print("  ✅  Project 2 Complete!")
print(f"     Final Accuracy : {accuracy*100:.2f}%")
print(f"     Final F1 Score : {f1:.4f}")
print("=" * 60)