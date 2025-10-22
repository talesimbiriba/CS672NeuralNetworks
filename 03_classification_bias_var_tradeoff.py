import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons

# -----------------------
# Reproducibility
# -----------------------
rng = np.random.default_rng(42)

# -----------------------
# Ground-truth data model
# -----------------------
# We define a known probability p*(x) = sigmoid(f(x)),
# then draw labels y ~ Bernoulli(p*(x)).
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def f_star(X):
    # non-linear function in R^2 -> R
    # tweak coefficients to make decision boundary curved but not too hard
    x1, x2 = X[:, 0], X[:, 1]
    return 3*np.sin(0.8*x1) + 2.0*(x2) - 0.6*x1*x2

def p_star(X):
    return sigmoid(f_star(X))

# def sample_dataset(n, low=-2.0, high=2.0, rng=rng):
#     X = rng.uniform(low, high, size=(n, 2))
#     p = p_star(X)
#     y = rng.binomial(1, p)
#     return X, y, p
def sample_dataset(n, low=-3.0, high=3.0, rng=rng):
    return sample_moon_dataset(n, noise=0.2, rng=rng)

def sample_moon_dataset(n, noise=0.0, rng=rng):
    X, y = make_moons(n_samples=n, noise=noise, random_state=rng.integers(1e9))
    # Estimate p*(x) using a KDE on the generated data
    from sklearn.neighbors import KernelDensity
    kde0 = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(X[y==0])
    kde1 = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(X[y==1])
    log_dens0 = kde0.score_samples(X)
    log_dens1 = kde1.score_samples(X)
    dens0 = np.exp(log_dens0)
    dens1 = np.exp(log_dens1)
    p = dens1 / (dens0 + dens1)
    return X, y, p

# -----------------------
# Model family: Polynomial Logistic Regression
# -----------------------
def make_model(degree, C=1.0, max_iter=200):
    # Pipeline: PolynomialFeatures -> Standardize -> LogisticRegression
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter))
    ])

# -----------------------
# Bias-Variance estimation (Brier score decomposition)
# -----------------------
def estimate_bias_variance(degree, n_train=200, n_reps=50, X_test=None, p_test=None, rng=rng):
    """
    For a fixed degree: repeatedly sample train sets, fit model, predict probabilities on X_test.
    Returns dict with bias2, var, noise, total (all averaged over test points).
    """
    models = []
    preds = []

    for _ in range(n_reps):
        Xtr, ytr, _ = sample_dataset(n_train, rng=rng)
        model = make_model(degree)
        model.fit(Xtr, ytr)
        models.append(model)
        preds.append(model.predict_proba(X_test)[:, 1])

    preds = np.vstack(preds)  # shape (n_reps, n_test)
    mean_pred = preds.mean(axis=0)  # E[\hat p]
    var_pred = preds.var(axis=0, ddof=0)  # Var[\hat p]
    bias2 = (mean_pred - p_test)**2
    noise = p_test * (1 - p_test)
    total = bias2 + var_pred + noise

    out = {
        "bias2": bias2.mean(),
        "variance": var_pred.mean(),
        "noise": noise.mean(),
        "total": total.mean(),
        "preds": preds,           # for optional inspection
        "mean_pred": mean_pred    # for optional inspection
    }
    return out

# -----------------------
# Evaluation settings
# -----------------------
n_test = 4000
X_test, y_test, p_test = sample_dataset(n_test, rng=rng)
degrees = [1, 2, 3, 4, 5, 6, 7, 8]  # model complexity knob
# degrees = [3,9]
n_reps = 60
n_train = 200

# Also track acc: we will train a model on each rep and measure train/test accuracy
def evaluate_accuracy_over_reps(degree, n_train=200, n_reps=30, X_test=None, y_test=None, rng=rng):
    train_accs, test_accs = [], []
    for _ in range(n_reps):
        Xtr, ytr, _ = sample_dataset(n_train, rng=rng)
        model = make_model(degree)
        model.fit(Xtr, ytr)
        train_accs.append(accuracy_score(ytr, model.predict(Xtr)))
        test_accs.append(accuracy_score(y_test, model.predict(X_test)))
    return np.mean(train_accs), np.std(train_accs), np.mean(test_accs), np.std(test_accs)

# -----------------------
# Run bias-variance study across degrees
# -----------------------
bv_curve = []
acc_curve = []

print("Estimating bias-variance and accuracy across model complexities...")
for d in degrees:
    bv = estimate_bias_variance(d, n_train=n_train, n_reps=n_reps, X_test=X_test, p_test=p_test, rng=rng)
    tr_m, tr_s, te_m, te_s = evaluate_accuracy_over_reps(d, n_train=n_train, n_reps=30, X_test=X_test, y_test=y_test, rng=rng)
    bv_curve.append((d, bv["bias2"], bv["variance"], bv["noise"], bv["total"]))
    acc_curve.append((d, tr_m, tr_s, te_m, te_s))

# Convert to arrays for plotting
bv_curve = np.array(bv_curve, dtype=float)
acc_curve = np.array(acc_curve, dtype=float)

# -----------------------
# Plots: Bias^2, Variance, Noise, Total (Brier score)
# -----------------------
plt.figure(figsize=(8,5))
plt.plot(bv_curve[:,0], bv_curve[:,1], marker='o', label='Bias$^2$')
plt.plot(bv_curve[:,0], bv_curve[:,2], marker='o', label='Variance')
plt.plot(bv_curve[:,0], bv_curve[:,3], marker='o', label='Noise (irreducible)')
plt.plot(bv_curve[:,0], bv_curve[:,4], marker='o', label='Total (Brier score)')
plt.xlabel('Polynomial degree (model complexity)')
plt.ylabel('Score')
plt.title('Bias–Variance Decomposition (Brier score)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# -----------------------
# Plots: Train/Test accuracy vs complexity
# -----------------------
plt.figure(figsize=(8,5))
plt.plot(acc_curve[:,0], acc_curve[:,1], marker='o', label='Train acc (mean of reps)')
plt.plot(acc_curve[:,0], acc_curve[:,3], marker='o', label='Test acc (mean of reps)')
plt.fill_between(acc_curve[:,0],
                 acc_curve[:,3]-acc_curve[:,4],
                 acc_curve[:,3]+acc_curve[:,4],
                 alpha=0.15, label='Test acc ± 1 std')
plt.xlabel('Polynomial degree (model complexity)')
plt.ylabel('Accuracy')
plt.title('Train/Test Accuracy vs Model Complexity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# -----------------------
# Visualizing decision boundaries & model variability
# -----------------------
def plot_decision_boundary(ax, model, bounds=(-3,3), ngrid=250):
    x1 = np.linspace(bounds[0], bounds[1], ngrid)
    x2 = np.linspace(bounds[0], bounds[1], ngrid)
    XX1, XX2 = np.meshgrid(x1, x2)
    grid = np.c_[XX1.ravel(), XX2.ravel()]
    proba = model.predict_proba(grid)[:,1].reshape(XX1.shape)
    cs = ax.contour(XX1, XX2, proba, levels=[0.5], linewidths=2)
    return cs

# Prepare one dataset to visualize learned boundaries variability
X_vis, y_vis, _ = sample_dataset(200, rng=rng)

def scatter_data(ax, X, y, s=15, alpha=0.8):
    ax.scatter(X[y==0,0], X[y==0,1], s=s, alpha=alpha, label='Class 0')
    ax.scatter(X[y==1,0], X[y==1,1], s=s, alpha=alpha, label='Class 1')

# Choose three complexities: low, medium, high
choices = [1, 4, 8]
fig, axes = plt.subplots(1, 3, figsize=(15,4.5), sharex=True, sharey=True)
for ax, d in zip(axes, choices):
    ax.set_title(f"Degree = {d} (decision boundary variability)")
    scatter_data(ax, X_vis, y_vis)
    # draw multiple boundaries from different resamples to show variance
    for _ in range(8):
        Xtr, ytr, _ = sample_dataset(n_train, rng=rng)
        model = make_model(d)
        model.fit(Xtr, ytr)
        try:
            plot_decision_boundary(ax, model)
        except Exception:
            pass
    ax.set_xlim([-3,3]); ax.set_ylim([-3,3])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

plt.tight_layout()

# -----------------------
# Show true probability as a sanity check (optional)
# -----------------------
fig, ax = plt.subplots(1, 1, figsize=(6,5))
grid_n = 250
g1 = np.linspace(-3, 3, grid_n)
g2 = np.linspace(-3, 3, grid_n)
G1, G2 = np.meshgrid(g1, g2)
G = np.c_[G1.ravel(), G2.ravel()]
P = p_star(G).reshape(G1.shape)
im = ax.imshow(P, extent=[-3,3,-3,3], origin='lower', aspect='auto')
ax.set_title("True probability $p^*(x)$")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()

plt.show()
