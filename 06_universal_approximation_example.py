#============================================================
# Universal Approximation Theorem demo
# Approximating f(x) = sin(3x) + 0.3x with shallow MLPs 
# of different widths
#============================================================

import torch, torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
x = torch.linspace(-1, 1, 100).unsqueeze(1)
# f_true = torch.sin(3*x) + 0.3*x
f_true = torch.sin(3*x) +  torch.cos(10*x) + torch.cos(4*x) + 0.3*x

# Fit a shallow network with M hidden units
# def fit_network(M, steps=600, lr=0.01):
def fit_network(M, steps=6000, lr=0.01):
# def fit_network(M, steps=600, lr=1):
    # shallow network
    net = nn.Sequential(nn.Linear(1,M), nn.Tanh(), nn.Linear(M,1))
    # optimizer
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # training loop
    for _ in range(steps):
        # compute loss over all data
        loss = ((net(x)-f_true)**2).mean()
        # gradient step
        opt.zero_grad(); loss.backward(); opt.step()
        # print loss every 10 steps
        if _ % 100 == 0:
            print(f"Step {_:4d}, Loss: {loss.item():.6f}, lr: {lr:.6f}")
    return net(x).detach()

plt.figure(figsize=(6,4))
# for M in [1, 2, 3, 5, 10]:
for M in [10, 1000]:
    y_hat = fit_network(M)
    plt.plot(x, y_hat, label=f"M={M}")
plt.plot(x, f_true, 'k--', label="target $f(x)$")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Universal Approximation: effect of hidden units M")
plt.legend(); plt.tight_layout()
plt.show()
