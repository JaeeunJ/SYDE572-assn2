import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("housing.data", header=None, sep='\s+')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_feature = X[:, 1]
X_train = X_feature[:100]
y_train = y[:100]
X_test = X_feature[100:]
y_test = y[100:]

mean_X = np.mean(X_train)
std_X = np.std(X_train)
if std_X == 0:
    std_X = 1e-8

X_train_norm = (X_train - mean_X) / std_X
X_test_norm = (X_test - mean_X) / std_X

def create_monomial_basis_1d(x_data, degree):
    N = len(x_data)
    Phi = [np.ones(N)]
    for d in range(1, degree + 1):
        Phi.append(x_data ** d)
    return np.column_stack(Phi)

# Helper to plot and save
def polynomial_regression_1d_vis(degree, filename):
    Phi_train = create_monomial_basis_1d(X_train_norm, degree)
    
    # Fit model (no regularization)
    w = np.linalg.pinv(Phi_train) @ y_train
    
    # Generate points for smooth curve plotting
    x_plot = np.linspace(min(X_train_norm.min(), X_test_norm.min()) - 0.5, 
                         max(X_train_norm.max(), X_test_norm.max()) + 0.5, 200)
    Phi_plot = create_monomial_basis_1d(x_plot, degree)
    y_plot = Phi_plot @ w
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train_norm, y_train, color='blue', alpha=0.6, label='Train Data')
    plt.scatter(X_test_norm, y_test, color='orange', alpha=0.4, label='Test Data', marker='x')
    plt.plot(x_plot, y_plot, color='red', linewidth=2, label=f'Degree {degree} Fit')
    
    plt.title(f'1D Polynomial Regression (Degree {degree}) on Feature 2 (ZN)')
    plt.xlabel('Normalized Feature 2 (ZN)')
    plt.ylabel('Median House Value (MEDV)')
    
    # Limit y-axis for extreme overfits so data remains visible
    plt.ylim(0, 60)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Generate the 3 interesting plots
polynomial_regression_1d_vis(1, 'part1c_deg1.svg')
polynomial_regression_1d_vis(3, 'part1c_deg3.svg')
polynomial_regression_1d_vis(9, 'part1c_deg9.svg')
print("Complete")