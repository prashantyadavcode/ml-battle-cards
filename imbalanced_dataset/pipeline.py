from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.datasets import make_classification
import numpy as np

# --- Generate imbalanced dataset ---
X, y = make_classification(
    n_samples=10000,
    n_features=10,
    weights=[0.95, 0.05],
    random_state=42
)

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Class distribution:", Counter(y))

# --- Pipeline ---
pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# --- Stratified CV on X_train only ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1')  # ✅ X_train, y_train
print(f"CV F1: {scores.mean():.3f} ± {scores.std():.3f}")

# --- Fit on full training set ---
pipe.fit(X_train, y_train)
y_prob = pipe.predict_proba(X_test)[:, 1]

# --- Threshold tuning ---
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_thresh = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold: {best_thresh:.2f}")

# --- Final prediction ---
y_pred_tuned = (y_prob >= best_thresh).astype(int)
print(classification_report(y_test, y_pred_tuned))