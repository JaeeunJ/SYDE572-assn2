import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("housing.data", header=None, sep=r'\s+')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test = X[:100], X[100:]
y_train, y_test = y[:100], y[100:]

mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)
std_X[std_X == 0] = 1e-8

X_train_norm = (X_train - mean_X) / std_X
X_test_norm = (X_test - mean_X) / std_X

dists = [np.linalg.norm(X_train_norm[i] - X_train_norm[j]) 
         for i in range(100) for j in range(i + 1, 100)]
sigma = np.mean(dists)

np.random.seed(42)
perm = np.random.permutation(len(X_train_norm))

def rbf_design_matrix(X_data, centers, sigma):
    N = X_data.shape[0]
    K = centers.shape[0]
    Phi = np.zeros((N, K))
    for k in range(K):
        dist_sq = np.sum((X_data - centers[k]) ** 2, axis=1)
        Phi[:, k] = np.exp(-dist_sq / (2 * (sigma ** 2)))
    return np.column_stack([np.ones(N), Phi])

K_values = list(range(5, 100, 10))
train_errors = []
test_errors = []

for K in K_values:
    chosen_centers = X_train_norm[perm[:K]]
    
    Phi_tr = rbf_design_matrix(X_train_norm, chosen_centers, sigma)
    Phi_te = rbf_design_matrix(X_test_norm, chosen_centers, sigma)
    
    w = np.linalg.pinv(Phi_tr) @ y_train
    
    rmse_tr = np.sqrt(np.mean((Phi_tr @ w - y_train) ** 2))
    rmse_te = np.sqrt(np.mean((Phi_te @ w - y_test) ** 2))
    
    train_errors.append(rmse_tr)
    test_errors.append(rmse_te)

plt.figure(figsize=(8, 5))
plt.plot(K_values, train_errors, marker='o', color='blue', label='Training Error')
plt.plot(K_values, test_errors, marker='s', color='orange', label='Testing Error')
plt.xlabel('Number of Basis Functions (K)')
plt.ylabel('RMS Error')
plt.title('Part 2E: RMS Error vs. Number of Gaussian Basis Functions')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.tight_layout()
plt.savefig('part2e_plot.svg')
plt.show()