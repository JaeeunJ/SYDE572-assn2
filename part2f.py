import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("housing.data", header=None, sep=r'\s+')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train = X[:100]
y_train = y[:100]

mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)
std_X[std_X == 0] = 1e-8
X_train_norm = (X_train - mean_X) / std_X

dists = [np.linalg.norm(X_train_norm[i] - X_train_norm[j]) 
         for i in range(100) for j in range(i + 1, 100)]
sigma = np.mean(dists)

np.random.seed(42)
perm = np.random.permutation(len(X_train_norm))
centers_90 = X_train_norm[perm[:90]]

def rbf_design_matrix(X_data, centers, sigma):
    N = X_data.shape[0]
    K = centers.shape[0]
    Phi = np.zeros((N, K))
    for k in range(K):
        dist_sq = np.sum((X_data - centers[k]) ** 2, axis=1)
        Phi[:, k] = np.exp(-dist_sq / (2 * (sigma ** 2)))
    return np.column_stack([np.ones(N), Phi])

Phi_full = rbf_design_matrix(X_train_norm, centers_90, sigma)

N = len(X_train_norm)
indices = np.random.permutation(N)
folds = np.array_split(indices, 10)

lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
avg_val_errors = []

for lam in lambdas:
    val_errors = []
    for k in range(10):
        val_idx = folds[k]
        tr_idx = np.setdiff1d(indices, val_idx)
        
        Phi_tr, Phi_val = Phi_full[tr_idx], Phi_full[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        
        I = np.eye(Phi_full.shape[1])
        I[0, 0] = 0.0
        
        w = np.linalg.pinv(Phi_tr.T @ Phi_tr + lam * I) @ Phi_tr.T @ y_tr
        
        pred_val = Phi_val @ w
        rmse = np.sqrt(np.mean((pred_val - y_val)**2))
        val_errors.append(rmse)
        
    avg_val_errors.append(np.mean(val_errors))

best_lambda = lambdas[np.argmin(avg_val_errors)]
print(f"Optimal Regularizing Constant: lambda = {best_lambda}")

plt.figure(figsize=(8, 5))
plt.semilogx(lambdas[1:], avg_val_errors[1:], marker='o', color='blue', label=r'Regularized ($\lambda > 0$)')
plt.axhline(y=avg_val_errors[0], color='red', linestyle='--', label=rf'Unregularized Baseline ($\lambda = 0$): {avg_val_errors[0]:.2f}')
plt.xlabel(r'$\lambda$ (Log Scale)')
plt.ylabel('Average Validation RMS Error')
plt.title(r'Part 2F: 10-Fold CV RMS Error vs. $\lambda$ (90 Gaussian Basis Functions)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.tight_layout()
plt.savefig('part2f_plot.svg')
plt.show()