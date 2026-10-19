import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data
data = pd.read_csv("housing.data", header=None, sep=r'\s+')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Extract 2nd feature (ZN) using first 100 points
X_feat = X[:100, 1]
y_train = y[:100]

# Normalize based on these 100 points
mean_X = np.mean(X_feat)
std_X = np.std(X_feat)
if std_X == 0:
    std_X = 1e-8
X_norm = (X_feat - mean_X) / std_X

# Monomial basis degree 8: [1, x, x^2, ..., x^8]
Phi = np.column_stack([X_norm**d for d in range(9)])

# 10-Fold Cross-Validation without sklearn
np.random.seed(42)
N = len(X_norm)
indices = np.random.permutation(N)
folds = np.array_split(indices, 10)

lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
avg_val_errors = []

for lam in lambdas:
    val_errors = []
    for k in range(10):
        val_idx = folds[k]
        train_idx = np.setdiff1d(indices, val_idx)
        
        Phi_tr, Phi_val = Phi[train_idx], Phi[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Regularization matrix (do not penalize bias w0)
        I = np.eye(Phi.shape[1])
        I[0, 0] = 0.0
        
        # w = (Phi^T Phi + lam * I)^(-1) Phi^T y
        w = np.linalg.pinv(Phi_tr.T @ Phi_tr + lam * I) @ Phi_tr.T @ y_tr
        
        pred_val = Phi_val @ w
        rmse = np.sqrt(np.mean((pred_val - y_val)**2))
        val_errors.append(rmse)
        
    avg_val_errors.append(np.mean(val_errors))

best_idx = np.argmin(avg_val_errors)
best_lambda = lambdas[best_idx]

# Save plot
plt.figure(figsize=(8, 5))
plt.semilogx(lambdas[1:], avg_val_errors[1:], marker='o', color='blue', label=r'Regularized ($\lambda > 0$)')
plt.axhline(y=avg_val_errors[0], color='red', linestyle='--', label=rf'Unregularized ($\lambda = 0$): {avg_val_errors[0]:.2f}')
plt.xlabel(r'$\lambda$ (Log Scale)')
plt.ylabel('Average Validation RMS Error')
plt.title(r'Part 1D: 10-Fold CV RMS Error vs. $\lambda$ (Degree 8 Polynomial)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.tight_layout()
plt.savefig('part1d_plot.svg')

print(f"Best lambda: {best_lambda}")
for l, err in zip(lambdas, avg_val_errors):
    print(f"lam={l}: {err:.4f}")