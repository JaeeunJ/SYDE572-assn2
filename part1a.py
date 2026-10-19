import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("housing.data", header=None, sep='\s+')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test = X[:100], X[100:]
y_train, y_test = y[:100], y[100:]

mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)
std_X[std_X == 0] = 1e-8

X_train_norm = (X_train - mean_X) / std_X
X_test_norm = (X_test - mean_X) / std_X

def create_monomial_basis(X_data, degree):
    N, D = X_data.shape
    Phi = [np.ones(N)]
    for d in range(1, degree + 1):
        for j in range(D):
            Phi.append(X_data[:, j] ** d)
    return np.column_stack(Phi)

train_errors = []
test_errors = []
degrees = list(range(1, 8))

for deg in degrees:
    Phi_train = create_monomial_basis(X_train_norm, deg)
    Phi_test = create_monomial_basis(X_test_norm, deg)
    
    w = np.linalg.pinv(Phi_train) @ y_train
    
    y_train_pred = Phi_train @ w
    y_test_pred = Phi_test @ w
    
    rmse_train = np.sqrt(np.mean((y_train_pred - y_train)**2))
    rmse_test = np.sqrt(np.mean((y_test_pred - y_test)**2))
    
    train_errors.append(rmse_train)
    test_errors.append(rmse_test)

plt.figure(figsize=(8, 5))
plt.plot(degrees, train_errors, marker='o', label='Training Error')
plt.plot(degrees, test_errors, marker='s', label='Testing Error')
plt.xlabel('Polynomial Degree')
plt.ylabel('RMS Error')
plt.title('RMS Error vs. Polynomial Degree (Monomial Basis)')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('part1a_plot.svg')
plt.show()