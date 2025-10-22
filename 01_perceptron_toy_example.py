""""
Implementation of Rosenblatt's Perceptron algorithm on a toy 2D dataset.
Plots the decision boundary and weight vector after each update (mistake).
Author: Tales Imbiriba
Date: 2025-08-10
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- Config ----------------
eta = 1.0          # learning rate
max_epochs = 20    # safety cap
SAVE_FRAMES = False  # set True to also save PNGs
out_dir = "perceptron_frames"

# ---------------- Data (6 points, separable) ----------------
# Labels in {-1, +1}
X = np.array([
    [0.2, 1.0],
    [0.8, 1.2],
    [0.4, 0.7],
    [1.6, 0.3],
    [2.0, 0.6],
    [1.4, 0.2],
], dtype=float)

y = np.array([+1, +1, +1, -1, -1, -1], dtype=int)

# ---------------- Setup ----------------
if SAVE_FRAMES:
    os.makedirs(out_dir, exist_ok=True)

# Augment features to absorb bias: x' = [x, 1], w' = [w, b]
X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
w = np.zeros(3)  # [w1, w2, b]

# Augment with bias and take the max Euclidean norm
R = np.linalg.norm(np.hstack([X, np.ones((X.shape[0], 1))]), axis=1).max()
print((R/0.1)**2)


# Fixed axes limits for consistent visuals
# pad = 0.4
pad = 3.0
# x1_min, x1_max = X[:, 0].min() - pad, X[:, 0].max() + pad
# x2_min, x2_max = X[:, 1].min() - pad, X[:, 1].max() + pad
x1_min, x1_max = -pad, pad
x2_min, x2_max = -pad, pad

def plot_state(w_vec, i_curr, update_idx, update=None):
    """Plot all points, highlight current point, draw decision boundary & weight vector."""
    w1, w2, b = w_vec
    if update is not None:
        update1, update2, updateb = update

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot points by class
    X_pos = X[y == +1]
    X_neg = X[y == -1]
    ax.scatter(X_pos[:, 0], X_pos[:, 1], marker="o", label="+1")
    ax.scatter(X_neg[:, 0], X_neg[:, 1], marker="x", label="-1")

    # Highlight current point
    ax.scatter(
        X[i_curr, 0], X[i_curr, 1],
        s=140, facecolors="none", edgecolors="black", linewidths=2, label="current"
    )

    # Highlight next point
    i_next = (i_curr + 1) % X.shape[0]
    ax.scatter(
        X[i_next, 0], X[i_next, 1],
        s=140, facecolors="none", edgecolors="blue", linewidths=2, label="next"
    )

    # Decision boundary: w1*x1 + w2*x2 + b = 0
    if abs(w2) > 1e-10:
        xs = np.array([x1_min, x1_max])
        ys = -(w1 * xs + b) / w2
        ax.plot(xs, ys, linewidth=2, label="boundary")
    else:
        # Vertical boundary when w2 ~ 0: w1*x1 + b = 0 -> x1 = -b/w1
        if abs(w1) > 1e-10:
            xv = -b / w1
            ax.plot([xv, xv], [x2_min, x2_max], linewidth=2, label="boundary")

    # Weight vector arrow (from origin)
    ax.quiver(
        0, 0, w1, w2,
        angles='xy', scale_units='xy', scale=1,
        width=0.006, label="w"
    )
    if update is not None:
        ax.quiver(
            0, 0, update1, update2,
            angles='xy', scale_units='xy', scale=1,
            width=0.006, label="w update", color="red"
        )

        ax.quiver(
            0, 0, w1-update1, w2-update2,
            angles='xy', scale_units='xy', scale=1,
            width=0.006, label="previous w", color="green"
        )


    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Perceptron update {update_idx}\n"
                 f"w = [{w1:.2f}, {w2:.2f}], b = {b:.2f}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if SAVE_FRAMES:
        fig.savefig(os.path.join(out_dir, f"frame_{update_idx:03d}.png"), dpi=140, bbox_inches="tight")

    plt.show()
    plt.close(fig)

# ---------------- Training (plot on every mistake) ----------------
update_count = 0
converged = False

for epoch in range(max_epochs):
    errors = 0
    for i in range(X.shape[0]):
        xi_aug = X_aug[i]
        yi = y[i]
        score = float(np.dot(w, xi_aug))
        pred = 1 if score >= 0 else -1
        update = np.zeros_like(xi_aug)
        if pred != yi:
            # Mistake -> update and plot
            w = w + eta * yi * xi_aug
            update = eta * yi * xi_aug
            update_count += 1
            errors += 1
            # plot_state(w_vec=w, i_curr=i, update_idx=update_count)
            print(f"norm w = {np.linalg.norm(w):.2f}")
        plot_state(w_vec=w, i_curr=i, update_idx=update_count, update=update)
    if errors == 0:
        converged = True
        break

print("Final weights [w1, w2, b]:", w)
print("Converged:", converged)
print(f"Total updates plotted: {update_count}")
if SAVE_FRAMES:
    print(f"Frames saved in: {out_dir}")

