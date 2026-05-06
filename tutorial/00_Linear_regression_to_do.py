r = np.corrcoef(x, y)[0, 1]
print("\nPearson correlation coefficient:")
print("r =", r)


for i, w1 in enumerate(w1_vals):
    for j, w0 in enumerate(w0_vals):
        y_pred = w1 * x + w0
        E[i, j] = np.sum((y - y_pred)**2)


x_line = np.linspace(min(x), max(x), 100)
y_line = best_w1 * x_line + best_w0


Phi = np.column_stack((x, np.ones(len(x))))
print("\nFisher Observation Matrix Φ:")
print(Phi)

A = Phi.T @ Phi
det_A = np.linalg.det(A)

Phi_pseudo = np.linalg.inv(A) @ Phi.T

W = Phi_pseudo @ y
w1, w0 = W

print("\nEstimated parameters:")
print("w1 =", w1)
print("w0 =", w0)

x_line = np.linspace(min(x), max(x), 100)
y_line = w1 * x_line + w0

W_lstsq, residuals, rank, s = np.linalg.lstsq(Phi, y)