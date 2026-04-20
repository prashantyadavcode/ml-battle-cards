import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples = 1000, n_features = 10, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Underfitting: too simple

underfit_model = LogisticRegression()
underfit_model.fit(X_train, y_train)

print('Underfit -> Train: ', accuracy_score(y_train, underfit_model.predict(X_train)))
print('Underfit -> Test: ', accuracy_score(y_test, underfit_model.predict(X_test)))

# Overfitting: too complex, no regularization

overfit_model = RandomForestClassifier(n_estimators=500, max_depth = None, random_state=42)
overfit_model.fit(X_train, y_train)
print('Overfit -> Train: ', accuracy_score(y_train, overfit_model.predict(X_train)))
print('Overfit -> Test: ', accuracy_score(y_test, overfit_model.predict(X_test)))

# Balanced: regularization + limited depth

balanced_model = RandomForestClassifier(n_estimators=100, max_depth = 5, random_state=42)
balanced_model.fit(X_train, y_train)
print('Balanced Model -> Train: ', accuracy_score(y_train, balanced_model.predict(X_train)))
print('Balanced Model -> Test: ', accuracy_score(y_test, balanced_model.predict(X_test)))


# Learning Curve : best visual diagnostic tool

def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv = 5, scoring = 'accuracy',
        train_sizes = np.linspace(0.1, 1.0, 10)
    )
    plt.plot(train_sizes, train_scores.mean(axis = 1), label = 'Train')
    plt.plot(train_sizes, val_scores.mean(axis = 1), label = 'Validation')
    plt.title(title)
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

plt.figure(figsize = (15, 4))
plt.subplot(1, 3, 1)
plot_learning_curve(underfit_model, X, y, 'Underfitting')
plt.subplot(1, 3, 2)
plot_learning_curve(overfit_model, X, y, 'Overfitting')
plt.subplot(1, 3, 3)
plot_learning_curve(balanced_model, X, y, 'Balanced')
plt.tight_layout()
plt.savefig('learning_curves.png')
print('Plot saved as learning_curves.png')