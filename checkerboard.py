import numpy as np
from sklearn.metrics import accuracy_score

# ==========================================
# Step 1: Checkerboard Data
# ==========================================

np.random.seed(10)

N = 1000

X = np.random.uniform(-4, 4, (N, 2))

labels = np.zeros(N)

for i in range(N):

    x = int(np.floor(X[i, 0]))
    y = int(np.floor(X[i, 1]))

    if (x + y) % 2 == 0:
        labels[i] = 1
    else:
        labels[i] = -1

d = labels.reshape(-1, 1)

# ==========================================
# Step 2: RKS Mapping
# ==========================================

dim = 70

np.random.seed(2545)

R = 2 * np.random.randn(2, dim)

M_data = X @ R

data_RKS = np.hstack([
    np.cos(M_data),
    np.sin(M_data)
])

# ==========================================
# Step 3: PINV Classification
# ==========================================

w = np.linalg.pinv(data_RKS) @ d

z = np.sign(data_RKS @ w)

acc = accuracy_score(d, z) * 100

print("Checkerboard Accuracy =", acc)