import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

# ── Load & Prepare ──────────────────────────────────────────────────────────
#
# The 3-class target (Low/Medium/High) is not cleanly separable from these
# features: the Low<->Medium boundary overlaps heavily, capping any model at
# ~60% accuracy. We instead predict a binary "high burnout risk" flag, which
# is both more learnable and more useful as an early-intervention signal.

df = pd.read_csv("data.csv")

# Features the EDA linked to burnout. AI_reliance_ratio captures the EDA's
# headline finding: burnout tracks AI *reliance*, not raw exposure.
df["AI_reliance_ratio"] = df["Weekly_GenAI_Hours"] / (
    df["Weekly_GenAI_Hours"] + df["Traditional_Study_Hours"] + 1e-6
)

features = [
    "Weekly_GenAI_Hours",
    "Traditional_Study_Hours",
    "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams",
    "AI_reliance_ratio",
]

X = df[features].copy()
y = (df["Burnout_Risk_Level"] == "High").astype(int)  # 1 = high burnout risk

# ── Split ───────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

class_names = ["Not High", "High"]
pos_rate = y_train.mean()
scale_pos_weight = (1 - pos_rate) / pos_rate  # ~3.0, to counter class imbalance

print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"High-burnout rate: {y.mean():.1%}  (baseline accuracy = {1 - y.mean():.1%})\n")

# ── Random Forest (baseline) ───────────────────────────────────────────────

rf = RandomForestClassifier(
    n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("=" * 60)
print("RANDOM FOREST")
print("=" * 60)
print(classification_report(y_test, rf_preds, target_names=class_names))
print(f"ROC AUC: {roc_auc_score(y_test, rf_proba):.3f}\n")

# ── XGBoost ─────────────────────────────────────────────────────────────────

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

print("=" * 60)
print("XGBOOST")
print("=" * 60)
print(classification_report(y_test, xgb_preds, target_names=class_names))
print(f"ROC AUC: {roc_auc_score(y_test, xgb_proba):.3f}")

# ── Confusion Matrices ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, rf_preds, display_labels=class_names, ax=axes[0], cmap="Blues"
)
axes[0].set_title("Random Forest")

ConfusionMatrixDisplay.from_predictions(
    y_test, xgb_preds, display_labels=class_names, ax=axes[1], cmap="Blues"
)
axes[1].set_title("XGBoost")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

# ── Feature Importance (XGBoost) ────────────────────────────────────────────

importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(8, 5))
importances.plot(kind="barh")
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

print("\nFeature importances:")
for feat, imp in importances.iloc[::-1].items():
    print(f"  {feat:30s} {imp:.4f}")
