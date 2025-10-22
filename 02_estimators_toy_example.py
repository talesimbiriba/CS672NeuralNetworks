import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n = 3               # small dataset to expose overfitting
sigma = 0.25           # noise std
M = 9                  # polynomial degree (try 3 vs 9)

# ---- Data generation ----
def true_function(x):  # ground truth curve
    return np.sin(2*np.pi*x)

x = np.sort(rng.uniform(0, 1, size=n))
y = true_function(x) + rng.normal(0, sigma, size=n)

# ---- Design matrix ----
def design_matrix(x, M):
    x = np.asarray(x)
    Phi = np.vstack([x**m for m in range(M+1)]).T
    return Phi

Phi = design_matrix(x, M)

# ---- MLE (OLS) ----
XtX = Phi.T @ Phi
Xty = Phi.T @ y
w_mle = np.linalg.solve(XtX, Xty)

# ---- MAP (ridge) ----
s = 10.0              # prior std of weights; smaller => stronger shrinkage
lam = sigma**2 / (s**2)
w_map = np.linalg.solve(XtX + lam*np.eye(M+1), Xty)

# ---- Full Bayes posterior ----
Sigma = np.linalg.inv((Phi.T @ Phi) / sigma**2 + np.eye(M+1) / s**2)
mu = Sigma @ (Phi.T @ y) / sigma**2

# ---- Predictions on a dense grid ----
xx = np.linspace(0, 1, 400)
Phi_x = design_matrix(xx, M)

y_mle = Phi_x @ w_mle
y_map = Phi_x @ w_map
y_bayes_mean = Phi_x @ mu
# predictive variance for Bayes (includes noise)
pred_var = sigma**2 + np.sum(Phi_x @ Sigma * Phi_x, axis=1)
pred_std = np.sqrt(pred_var)

# ---- Metrics on a test set ----
xte = np.sort(rng.uniform(0, 1, size=500))
yte = true_function(xte) + rng.normal(0, sigma, size=500)
Phi_te = design_matrix(xte, M)
def rmse(y_true, y_hat): return np.sqrt(np.mean((y_true - y_hat)**2))
rmse_mle = rmse(yte, Phi_te @ w_mle)
rmse_map = rmse(yte, Phi_te @ w_map)
rmse_bayes = rmse(yte, Phi_te @ mu)

# Negative log predictive density for Bayes
from numpy import log, pi
mu_te = Phi_te @ mu
var_te = sigma**2 + np.sum(Phi_te @ Sigma * Phi_te, axis=1)
nlpd_bayes = 0.5*np.mean(np.log(2*np.pi*var_te) + (yte - mu_te)**2 / var_te)

print(f"RMSE  MLE  : {rmse_mle:.3f}")
print(f"RMSE  MAP  : {rmse_map:.3f}")
print(f"RMSE  Bayes: {rmse_bayes:.3f}")
print(f"NLPD Bayes : {nlpd_bayes:.3f}")

# ---- Plot ----
plt.figure(figsize=(8,5))
plt.scatter(x, y, s=35, label="train data")
plt.plot(xx, true_function(xx), lw=2, label="true function")
plt.plot(xx, y_bayes_mean, lw=2, label="Bayes mean")
plt.fill_between(xx, y_bayes_mean - 2*pred_std, y_bayes_mean + 2*pred_std,
                 alpha=0.2, label="Bayes ±2σ")
plt.plot(xx, y_mle,'k--',lw=1.5, label="MLE (OLS)")
plt.plot(xx, y_map, 'r', lw=0.5, color='k', ls='--', label="MAP (ridge)")

plt.legend()
plt.title(f"Polynomial degree M={M}, n={n}, σ={sigma}, s={s}")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout(); plt.show()
