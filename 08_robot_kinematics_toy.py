# Bishop Ch.6 (Figures 6.17–6.19) + MDN
# ---------------------------------------------------------------
# - Data: x ~ U(0,1), t = x + 0.3*sin(2πx) + U(-0.1,0.1)    [Fig. 6.17 text]
# - Forward fit (t|x): 2-layer net, 6 tanh hidden, linear output, MSE loss
# - Inverse fit (x|t): same architecture, shows poor least-squares fit
# - MDN: K=3 Gaussians; 2-layer net with 5 tanh hidden units; 9 outputs (μ, σ, π)
#   Plots: (a) π_k(x), (b) μ_k(x), (c) density p(t|x), (d) mode approx

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Repro 
# -------------------------
np.random.seed(0)
torch.manual_seed(0)


# -------------------------
# Two-layer regressors (6 tanh hidden, linear output) — Fig. 6.17
# -------------------------
class TinyNet(nn.Module):
    def __init__(self, hidden=6):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, z):
        return self.fc2(torch.tanh(self.fc1(z)))

def train_ls(model, X, T, epochs=5000, lr=1e-1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        y = model(X)
        loss = F.mse_loss(y, T)   # sum-of-squares (least squares)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


# -------------------------
# Mixture Density Network (K=3; 2-layer; 5 tanh hidden) — Fig. 6.19
# p(t|x) = sum_k pi_k(x) N(t | mu_k(x), sigma_k^2(x))
# -------------------------
class MDN(nn.Module):
    def __init__(self, hidden=5, K=3):
        super().__init__()
        self.K = K
        self.h1 = nn.Linear(1, hidden)
        self.h2 = nn.Linear(hidden, hidden)
        self.out_pi = nn.Linear(hidden, K)   # logits -> softmax
        self.out_mu = nn.Linear(hidden, K)   # means
        self.out_ls = nn.Linear(hidden, K)   # log sigma
    
    def forward(self, x):
        # compute embeddings
        h = torch.tanh(self.h1(x))
        h = torch.tanh(self.h2(h))
        
        # compute mixture params (pi, mu, sigma)
        pi = F.softmax(self.out_pi(h), dim=-1)     # [B,K], sum to 1
        mu = self.out_mu(h)                        # [B,K]
        sigma = torch.exp(self.out_ls(h))          # [B,K], >0

        return pi, mu, sigma
    
    def compute_pdf(self, x, t):
        with torch.no_grad():    
            # compute p(t|x) on a grid of (x,t) points
            pi, mu, sigma = self.forward(x)
            t = t.expand_as(mu)  # [B,K] (= target replicated K times)
            log_norm = -0.5*torch.log(2*np.pi*sigma**2 + 1e-12)
            log_exp  = -0.5*((t - mu)**2) / (sigma**2 + 1e-12)
            log_comp = log_norm + log_exp           # [B,K]
            log_mix  = torch.logsumexp(torch.log(pi + 1e-12) + log_comp, dim=2)  # [B]
        return torch.exp(log_mix)               # [B]
    
    def predict(self, x):
        # compute the conditional average E[t|x] = sum_k pi_k(x) * mu_k(x)
        # and compute the variance Var[t|x] = sum_k pi_k(x) * (sigma_k^2(x) + mu_k^2(x)) - E[t|x]^2
        # return mean, var
        with torch.no_grad():
            pi, mu, sigma = self.forward(x)
            mean = (pi * mu).sum(dim=1, keepdim=True)
            var = (pi * (sigma**2 + mu**2)).sum(dim=1, keepdim=True) - mean**2
        return mean, var            


def mdn_nll(pi, mu, sigma, t):
    # Negative log-likelihood of 1D Gaussian mixture
    t = t.expand_as(mu)  # [B,K] (= target replicated K times) why? -> broadcasting, what is broadcasting? -> https://pytorch.org/docs/stable/notes/broadcasting.html
    log_norm = -0.5*torch.log(2*np.pi*sigma**2 + 1e-12)
    log_exp  = -0.5*((t - mu)**2) / (sigma**2 + 1e-12)
    log_comp = log_norm + log_exp           # [B,K]
    log_mix  = torch.logsumexp(torch.log(pi + 1e-12) + log_comp, dim=1)  # [B]
    return -(log_mix.mean())


if __name__ == "__main__":

    # -------------------------
    # Data (matches book)
    # -------------------------
    N = 250
    x = np.random.rand(N)
    t = x + 0.3*np.sin(2*np.pi*x) + np.random.uniform(-0.1, 0.1, size=N)

    X_fwd = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # input for forward net
    T_fwd = torch.tensor(t, dtype=torch.float32).unsqueeze(1)  # target for forward net

    # Inverse problem: swap roles of x and t
    X_inv = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
    T_inv = torch.tensor(x, dtype=torch.float32).unsqueeze(1)

    K = 3  # number of mixture components for MDN 


    # -------------------------
    # Example 1: Plot least-squares forward and inverse fits (Fig. 6.17)
    # -------------------------

    net_fwd = train_ls(TinyNet(6), X_fwd, T_fwd)
    net_inv = train_ls(TinyNet(6), X_inv, T_inv)

    # Smooth predictions (test points)
    xx = torch.linspace(0, 1, 400).unsqueeze(1)
    with torch.no_grad():
        y_fwd = net_fwd(xx).squeeze(1).numpy()
    tt = torch.linspace(float(t.min()), float(t.max()), 400).unsqueeze(1)
    with torch.no_grad():
        y_inv = net_inv(tt).squeeze(1).numpy()

    # Plot Figure 6.17
    plt.figure(figsize=(7.4, 3.6))
    plt.subplot(1,2,1)
    plt.scatter(x, t, s=10, alpha=0.65)
    plt.plot(xx.numpy(), y_fwd, 'r', lw=2)
    plt.title("Forward: least-squares fit")
    plt.xlabel("$x$"); plt.ylabel("$t$")
    plt.subplot(1,2,2)
    plt.scatter(t, x, s=10, alpha=0.65)
    plt.plot(tt.numpy(), y_inv, 'r', lw=2)
    plt.title("Inverse: least-squares (poor)")
    plt.xlabel("input $x$ (here original $t$)"); plt.ylabel("target $t$ (here original $x$)")
    plt.tight_layout()
    # plt.savefig(OUT / "Figure_17.pdf"); plt.close()


    # -------------------------
    # Example 2: Train MDN on inverse problem (Fig. 6.19)
    # -------------------------


    mdn = MDN(hidden=5, K=K)
    opt = torch.optim.Adam(mdn.parameters(), lr=1e-3)

    X_mdn, T_mdn = X_inv, T_inv  # model p(t|x) with input=x:=original t, target=t:=original x
    for ep in range(10000):
        idx = torch.randint(0, X_mdn.shape[0], (256,))
        xb, tb = X_mdn[idx], T_mdn[idx]
        pi, mu, sg = mdn(xb)
        loss = mdn_nll(pi, mu, sg, tb)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 500 == 0:
            print(f"MDN epoch {ep+1:4d} | NLL loss {loss.item():.4f}")

    # Grid over input x for plots
    xg = torch.linspace(float(X_mdn.min()), float(X_mdn.max()), 500).unsqueeze(1)
    with torch.no_grad():
        pi_g, mu_g, sg_g = mdn(xg)

    # # (a) Mixing coefficients π_k(x)
    # plt.figure(figsize=(6.6, 3.4))
    # for k in range(mdn.K):
    #     plt.plot(xg.squeeze().numpy(), pi_g[:,k].numpy(), label=fr"$\pi_{k+1}(x)$")
    # plt.xlabel("$x$"); plt.ylabel(r"$\pi_k(x)$"); plt.title("Mixing coefficients vs $x$")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(OUT / "Figure_18.pdf"); plt.close()

    # (b)(c) Conditional density p(t|x) heatmap + component means
    t_min, t_max = -0.1, 1.1
    tg = torch.linspace(t_min, t_max, 500).unsqueeze(0)   # [1,T]

    with torch.no_grad():
        PI = pi_g.unsqueeze(-1)            # [G,K,1]
        MU = mu_g.unsqueeze(-1)            # [G,K,1]
        SG = sg_g.unsqueeze(-1) + 1e-6     # [G,K,1]
        TT = tg.unsqueeze(1)               # [1,G,1] -> broadcast to [G,1,T]
        log_norm = -0.5*torch.log(2*np.pi*SG**2)
        log_gaus = log_norm - 0.5*((TT - MU)**2)/(SG**2)
        log_mix  = torch.logsumexp(torch.log(PI) + log_gaus, dim=1)  # [G,T]
        pdf = torch.exp(log_mix).numpy()     # [G,T]

    # mode approximation: mean of most-probable component
    with torch.no_grad():
        top = torch.argmax(pi_g, dim=1)
        mode_approx = mu_g[torch.arange(mu_g.shape[0]), top].numpy()

    fig = plt.figure(figsize=(7.6, 6.2))
    # (a) π_k(x) again (top-left)
    ax1 = plt.subplot(2,2,1)
    for k in range(mdn.K):
        ax1.plot(xg.squeeze().numpy(), pi_g[:,k].numpy(), label=fr"$\pi_{k+1}(x)$")
    ax1.set_title("(a) Mixing coefficients"); ax1.set_xlabel("$x$"); ax1.set_ylabel(r"$\pi_k(x)$")
    ax1.legend(fontsize=8)

    # (b) μ_k(x) (top-right)
    ax2 = plt.subplot(2,2,2)
    for k in range(mdn.K):
        ax2.plot(xg.squeeze().numpy(), mu_g[:,k].numpy())
    ax2.set_title("(b) Component means"); ax2.set_xlabel("$x$"); ax2.set_ylabel(r"$\mu_k(x)$")

    # plot the conditional density p(t|x) as heatmap as a function of t and x
    # make a meshgrid of (xg, tg) for plotting the density heatmap
    # xg = np.linspace(0.0, 1.0, 10)
    # tg = np.linspace(t_min, t_max, 10)
    xxg, ttg = np.meshgrid(xg.squeeze().numpy(), tg.squeeze().numpy(), indexing='ij')
    # pdf already computed above
    # pdf = np.exp(log_mix.numpy())   # [G,T]
    # plot the density as an image
    # overlay the component means μ_k(x)
    xxg = torch.tensor(xxg, dtype=torch.float32).unsqueeze(-1)  # [G,T,1]
    ttg = torch.tensor(ttg, dtype=torch.float32).unsqueeze(-1)  # [G,T,1]
    zz = mdn.compute_pdf(xxg, ttg)  # [G,T] density values for color map
    ax3 = plt.subplot(2,2,3)
    im = ax3.contourf(xxg.squeeze().numpy(), ttg.squeeze().numpy(), zz.numpy(), levels=50, cmap='viridis')
    
    # (xxg, ttg, zz.numpy(), cmap='viridis', shading='auto')

    # (c) density + means (bottom-left)
    # ax3 = plt.subplot(2,2,3)
    # extent = [xg.min().item(), xg.max().item(), t_min, t_max]
    # im = ax3.imshow(pdf.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')
    # for k in range(mdn.K):
    #     ax3.plot(xg.squeeze().numpy(), mu_g[:,k].numpy(), lw=1.0, alpha=0.9)
    # ax3.set_title(r"(c) Conditional density $p(t|x)$")
    # ax3.set_xlabel("$x$"); ax3.set_ylabel("$t$")
    # cb = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04); cb.set_label("density")

    # (d) mode approx over scatter (bottom-right)
    ax4 = plt.subplot(2,2,4)
    ax4.scatter(X_mdn.numpy(), T_mdn.numpy(), s=6, alpha=0.4)
    # ax4.plot(xg.squeeze().numpy(), mode_approx, 'r.', ms=2.5)
    ax4.plot(xg.squeeze().numpy(), mode_approx, 'r.', ms=2.5)
    ax4.set_title("(d) Approximate conditional mode")
    ax4.set_xlabel("$x$"); ax4.set_ylabel("$t$")

    # also plot the predictive distribution: sum_k pi_k(x) N(t | mu_k(x), sigma_k^2(x))
    t, var = mdn.predict(xg)
    ax4.plot(xg.squeeze().numpy(), t.numpy(), 'k-', lw=1.5, label="predictive mean")
    # plot uncertainty as shaded area (mean ± stddev)
    ax4.fill_between(xg.squeeze().numpy(),
                     (t - var.sqrt()).squeeze().numpy(),
                     (t + var.sqrt()).squeeze().numpy(),
                     color='gray', alpha=0.3, label="predictive stddev")
    plt.tight_layout()

    plt.show()