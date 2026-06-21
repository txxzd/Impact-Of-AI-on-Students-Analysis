import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# ── Load & Prepare ──────────────────────────────────────────────────────────

df = pd.read_csv("data.csv")

target = "Burnout_Risk_Level"

features = [
    "Weekly_GenAI_Hours",
    "Traditional_Study_Hours",
    "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams",
]

X = df[features].copy()
y = df[target].copy()

# Encode target with explicit ordering
target_order = ["Low", "Medium", "High"]
le = LabelEncoder()
le.classes_ = np.array(target_order)
y = le.transform(y)

# ── Split ───────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"Class distribution (test): {dict(zip(target_order, np.bincount(y_test)))}\n")

# ── Random Forest (baseline) ───────────────────────────────────────────────

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("=" * 60)
print("RANDOM FOREST")
print("=" * 60)
print(classification_report(y_test, rf_preds, target_names=target_order))

# ── XGBoost ─────────────────────────────────────────────────────────────────

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss",
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

print("=" * 60)
print("XGBOOST")
print("=" * 60)
print(classification_report(y_test, xgb_preds, target_names=target_order))

# ── Confusion Matrices ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, rf_preds, display_labels=target_order, ax=axes[0], cmap="Blues"
)
axes[0].set_title("Random Forest")

ConfusionMatrixDisplay.from_predictions(
    y_test, xgb_preds, display_labels=target_order, ax=axes[1], cmap="Blues"
)
axes[1].set_title("XGBoost")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

# ── Feature Importance (XGBoost) ────────────────────────────────────────────

importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(8, 6))
importances.plot(kind="barh")
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

print("\nFeature importances:")
for feat, imp in importances.iloc[::-1].items():
    print(f"  {feat:30s} {imp:.4f}")
