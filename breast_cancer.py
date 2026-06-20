import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Convert labels to ±1
y = np.where(y == 0, -1, 1)
y = y.reshape(-1, 1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Random Kitchen Sink Mapping
D = 100

np.random.seed(2545)

R = 2 * np.random.randn(X_train.shape[1], D)

train_map = X_train @ R
test_map = X_test @ R

train_RKS = np.hstack([
    np.cos(train_map),
    np.sin(train_map)
])

test_RKS = np.hstack([
    np.cos(test_map),
    np.sin(test_map)
])

# PINV Training
W = np.linalg.pinv(train_RKS) @ y_train

# Prediction
pred = np.sign(test_RKS @ W)

# Accuracy
acc = accuracy_score(y_test, pred)

print("Breast Cancer Accuracy =", acc * 100)