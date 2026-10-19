import pandas as pd
import numpy as np

data = pd.read_csv("housing.data", header=None, sep='\s+')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train = X[:100]
y_train = y[:100]

mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)
std_X[std_X == 0] = 1e-8

X_train_norm = (X_train - mean_X) / std_X

Phi_train_deg1 = np.column_stack([np.ones(len(X_train_norm)), X_train_norm])

w_deg1 = np.linalg.pinv(Phi_train_deg1) @ y_train

w_RM = w_deg1[6]
w_DIS = w_deg1[8]

print(f"Weight 6 (RM - rooms): {w_RM:.2f}")
print(f"Weight 8 (DIS - distance): {w_DIS:.2f}")