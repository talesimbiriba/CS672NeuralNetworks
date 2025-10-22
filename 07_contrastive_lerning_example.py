import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --------------------------
# 1. Make synthetic data
# --------------------------
def make_blobs(n_per_class=256, delta=2.5, cov=0.4):
    mean0, mean1 = [-delta,0], [delta,0]
    cov_m = cov * np.eye(2)
    X0 = np.random.multivariate_normal(mean0, cov_m, size=n_per_class)
    X1 = np.random.multivariate_normal(mean1, cov_m, size=n_per_class)
    X = np.vstack([X0,X1]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

X, y = make_blobs(n_per_class=256)
# moons dataset alternative
from sklearn.datasets import make_moons
X_moons, y_moons = make_moons(n_samples=512, noise=0.1, random_state=42)
X = torch.from_numpy(X_moons.astype(np.float32))
y = torch.from_numpy(y_moons.astype(np.int64))


# --------------------------
# 2. Simple augmentations
# --------------------------
def augment(x, jitter_std=0.1, rot_std_deg=15.0):
    B = x.shape[0]
    j = torch.randn_like(x) * jitter_std
    xj = x + j
    theta = (torch.randn(B) * (rot_std_deg*np.pi/180.0))
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([cos_t,-sin_t],-1),
                     torch.stack([sin_t, cos_t],-1)],1)  # [B,2,2]
    return torch.bmm(R, xj.unsqueeze(-1)).squeeze(-1)

# --------------------------
# 3. Encoder network
# --------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU(),
            nn.Linear(64,2)
        )
    def forward(self,x):
        z = self.net(x)
        return F.normalize(z,p=2,dim=-1)  # unit norm

enc = Encoder()

# --------------------------
# 4. InfoNCE loss
# --------------------------
def info_nce(za,zp,zn,tau=0.2):
    pos_logits = (za*zp).sum(-1,keepdim=True)/tau
    neg_logits = torch.einsum('bd,bkd->bk', za, zn)/tau
    logits = torch.cat([pos_logits,neg_logits],1)
    labels = torch.zeros(za.size(0),dtype=torch.long)
    return F.cross_entropy(logits,labels)

# --------------------------
# 5. Build training pairs
# --------------------------
def build_pairs(x,y,batch_size=128,K=16):
    idx = torch.randint(0,len(x),(batch_size,))
    xa,ya = x[idx], y[idx]
    xp = augment(xa)
    zn_list=[]
    for i in range(batch_size):
        neg_idx = torch.nonzero(y!=ya[i]).squeeze(1)
        choice = neg_idx[torch.randint(0,len(neg_idx),(K,))]
        zn_list.append(augment(x[choice]))
    zn = torch.stack(zn_list,0) # [B,K,2]
    return xa,xp,zn

# --------------------------
# 6. Train
# --------------------------
opt = torch.optim.Adam(enc.parameters(),lr=1e-3)
loss_hist=[]
for ep in range(50):
    xa,xp,zn = build_pairs(X,y)
    za,zp = enc(xa), enc(xp)
    zn_enc = enc(zn.view(-1,2)).view(zn.size(0),zn.size(1),-1)
    loss = info_nce(za,zp,zn_enc)
    opt.zero_grad(); loss.backward(); opt.step()
    loss_hist.append(loss.item())

# --------------------------
# 7. Plot embeddings before/after
# --------------------------
def plot_embeddings(ax,enc,X,y,title):
    with torch.no_grad():
        z = enc(X).numpy()
    ax.scatter(z[y==0,0],z[y==0,1],s=8,label="class 0")
    ax.scatter(z[y==1,0],z[y==1,1],s=8,label="class 1")
    ax.set_title(title); ax.set_aspect("equal"); ax.legend()

fig,ax = plt.subplots(1,2,figsize=(10,4))
plot_embeddings(ax[0],Encoder(),X,y,"Before training (random)")
plot_embeddings(ax[1],enc,X,y,"After InfoNCE training")
plt.show()

# Plot loss curve
plt.plot(loss_hist); plt.xlabel("epoch"); plt.ylabel("InfoNCE loss"); plt.show()
