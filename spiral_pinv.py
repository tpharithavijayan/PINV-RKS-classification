import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# ==========================================
# Step 1: Generate Spiral Dataset
# ==========================================

nd = 100

j = np.arange(1, nd + 1)

angle = j * np.pi / 16
radius = 6.5 * (104 - j) / 104

x = radius * np.sin(angle)
y = radius * np.cos(angle)

x1 = x.reshape(-1, 1)
y1 = y.reshape(-1, 1)

x2 = -x1
y2 = -y1

A = np.vstack([
    np.hstack([x1, y1]),
    np.hstack([x2, y2])
])

# Labels
d = np.ones((nd, 1))
d = np.vstack([d, -d])

# ==========================================
# Step 2: Random Kitchen Sink Mapping
# ==========================================

dim = 70

np.random.seed(2545)

R = 2 * np.random.randn(2, dim)

M_data = A @ R

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

print("Accuracy =", acc)
plt.scatter(A[:,0], A[:,1], c=d.flatten())
plt.title("Two Spiral Dataset")
plt.savefig("spiral_dataset.png")
plt.show()
