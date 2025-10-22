# Momentum vs Nesterov on a toy quadratic
# ---------------------------------------
# f(w) = 0.5 * (a*x^2 + b*y^2), with a << b to create a narrow valley.
# Updates:
#   - Vanilla:    w <- w - eta * grad(w)
#   - Momentum:   v <- mu*v - eta * grad(w);             w <- w + v
#   - Nesterov:   v <- mu*v - eta * grad(w + mu*v_prev); w <- w + v

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(0)

# ----- problem definition -----
a, b = 1.0, 50.0  # curvatures: strong anisotropy => visible oscillations
def f(w):
    x, y = w
    return 0.5*(a*x**2 + b*y**2)

def grad(w):
    x, y = w
    return np.array([a*x, b*y])

# ----- optimizers -----
def run_vanilla(w0, eta, steps):
    w = w0.copy()
    traj = [w.copy()]
    for _ in range(steps):
        g = grad(w)
        w = w - eta*g
        traj.append(w.copy())
    return np.array(traj)

def run_momentum(w0, eta, mu, steps):
    w = w0.copy()
    v = np.zeros_like(w)
    traj = [w.copy()]
    for _ in range(steps):
        g = grad(w)
        v = mu*v - eta*g
        w = w + v
        traj.append(w.copy())
    return np.array(traj)

def run_nesterov(w0, eta, mu, steps):
    w = w0.copy()
    v = np.zeros_like(w)
    traj = [w.copy()]
    for _ in range(steps):
        v_prev = v.copy()
        g = grad(w + mu*v_prev)
        v = mu*v_prev - eta*g
        w = w + v
        traj.append(w.copy())
    return np.array(traj)

# ----- experiment settings -----
w0    = np.array([2.5, 2.5])  # start away from optimum (0,0)
steps = 60
eta   = 0.01                  # learning rate small enough for stability
mu    = 0.5                   # momentum coefficient

traj_van   = run_vanilla(w0, eta=eta, steps=steps)
traj_mom   = run_momentum(w0, eta=eta, mu=mu, steps=steps)
traj_nest  = run_nesterov(w0, eta=eta, mu=mu, steps=steps)

# ----- report final values -----
def summarize(name, traj):
    print(f"{name:>10s} | final w = {traj[-1]},  f(w) = {f(traj[-1]):.4e}")

summarize("Vanilla",  traj_van)
summarize("Momentum", traj_mom)
summarize("Nesterov", traj_nest)

# ----- plotting (contours + trajectories) -----
# Contour grid
x = np.linspace(-2.6, 2.6, 400)
y = np.linspace(-2.6, 2.6, 400)
X, Y = np.meshgrid(x, y)
Z = 0.5*(a*X**2 + b*Y**2)

plt.figure(figsize=(7, 6))
cs = plt.contour(X, Y, Z, levels=np.geomspace(1e-3, 8, 18))
plt.clabel(cs, inline=True, fontsize=7)

# Trajectories
plt.plot(traj_van[:,0],  traj_van[:,1],  marker='o', markersize=2, linewidth=4, label='Vanilla GD')
plt.plot(traj_mom[:,0],  traj_mom[:,1],  marker='s', markersize=2, linewidth=2, label='Momentum (μ=0.8)')
plt.plot(traj_nest[:,0], traj_nest[:,1], marker='v', markersize=2, linewidth=1, label='Nesterov (μ=0.8)')

# Start / end markers
plt.scatter([w0[0]], [w0[1]], s=60, marker='x', label='Start')
plt.scatter([0],[0], s=60, marker='*', label='Optimum (0,0)')

plt.title('Momentum vs Nesterov on an Ill-Conditioned Quadratic')
plt.xlabel('w1'); plt.ylabel('w2'); plt.legend(loc='upper right')
plt.axis('equal'); plt.tight_layout()

# Save to your course figure path if you want:
outdir = Path("figs/Chapter-7"); outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(outdir / "momentum_vs_nesterov.png", dpi=200)
plt.show()
