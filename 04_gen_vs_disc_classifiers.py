# ============================================
# Decision Theory + Generative vs Discriminative (from Chap. 5)
# - Synthetic 2D data from two Gaussians
# - Generative LDA (shared Σ) + QDA (separate Σ) posteriors
# - Discriminative Logistic Regression (gradient descent, cross-entropy)
# - Bayes decision with loss matrix + reject option
# - Confusion metrics + ROC curve
# - Decision boundary plots
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

rng = np.random.default_rng(50)

# -----------------------------
# 1) Generate synthetic data
# -----------------------------
N1, N2 = 200, 200
mu1 = np.array([0.0, 0.0])
mu2 = np.array([2.0, 1.5])
Sigma_shared = np.array([[1.0, 0.4],[0.4, 1.0]])      # shared Σ (LDA)
Sigma1 = Sigma_shared                                  # class 1 Σ (QDA)
Sigma2 = np.array([[1.2, -0.3],[-0.3, 0.8]])           # class 2 Σ (QDA)

X1 = rng.multivariate_normal(mu1, Sigma1, size=N1)
X2 = rng.multivariate_normal(mu2, Sigma2, size=N2)
X = np.vstack([X1, X2])
t = np.hstack([np.zeros(N1, dtype=int), np.ones(N2, dtype=int)])  # 0=C1, 1=C2

# train/val split
perm = rng.permutation(len(X))
X, t = X[perm], t[perm]
split = int(0.7*len(X))
Xtr, ytr = X[:split], t[:split]
Xte, yte = X[split:], t[split:]

# ----------------------------------------
# 2) Utilities: Gaussian pdf, posteriors
# ----------------------------------------
def gaussian_pdf(x, mu, Sigma):
    # x: (..., D)
    D = mu.shape[0]
    xmu = x - mu
    iS = np.linalg.inv(Sigma)
    logdet = np.linalg.slogdet(Sigma)[1]
    quad = np.einsum('...i,ij,...j->...', xmu, iS, xmu)
    logp = -0.5*(D*np.log(2*np.pi) + logdet + quad)
    return np.exp(logp)

def estimate_generative_params(X, y, shared_cov=True):
    # Priors
    pi1 = (y==0).mean()
    pi2 = 1 - pi1
    # Means
    mu1 = X[y==0].mean(axis=0)
    mu2 = X[y==1].mean(axis=0)
    # Covariances
    if shared_cov:
        S1 = np.cov(X[y==0].T, bias=False)
        S2 = np.cov(X[y==1].T, bias=False)
        # pooled (ML estimate, proportional to within-class scatter)
        n1, n2 = (y==0).sum(), (y==1).sum()
        Sigma = (n1*S1 + n2*S2) / (n1 + n2)
        return dict(pi1=pi1, pi2=pi2, mu1=mu1, mu2=mu2, Sigma=Sigma)
    else:
        S1 = np.cov(X[y==0].T, bias=False)
        S2 = np.cov(X[y==1].T, bias=False)
        return dict(pi1=pi1, pi2=pi2, mu1=mu1, mu2=mu2, Sigma1=S1, Sigma2=S2)

def posteriors_LDA(X, params):
    # p(C1|x) with shared Σ (linear boundary)  [Eqs. 5.47–5.50]
    pi1, pi2 = params['pi1'], params['pi2']
    mu1, mu2 = params['mu1'], params['mu2']
    Sigma = params['Sigma']
    p1 = gaussian_pdf(X, mu1, Sigma) * pi1
    p2 = gaussian_pdf(X, mu2, Sigma) * pi2
    s = p1 + p2
    return (p1/s, p2/s)

def posteriors_QDA(X, params):
    # p(C1|x) with class-specific Σ (quadratic boundary)
    pi1, pi2 = params['pi1'], params['pi2']
    mu1, mu2 = params['mu1'], params['mu2']
    S1, S2 = params['Sigma1'], params['Sigma2']
    p1 = gaussian_pdf(X, mu1, S1) * pi1
    p2 = gaussian_pdf(X, mu2, S2) * pi2
    s = p1 + p2
    return (p1/s, p2/s)

# ---------------------------------------------------------
# 3) Logistic regression (discriminative) with cross-entropy
# ---------------------------------------------------------
def sigmoid(a):
    return 1.0/(1.0+np.exp(-a))

def logistic_train(X, y, lr=0.1, steps=3000, l2=0.0):
    # Add bias
    Phi = np.hstack([np.ones((X.shape[0],1)), X])
    w = np.zeros(Phi.shape[1])
    for _ in range(steps):
        a = Phi @ w
        yhat = sigmoid(a)
        # gradient of cross-entropy: sum_n (yhat - y) * phi_n  [Eq. 5.75]
        grad = Phi.T @ (yhat - y) / len(y) + l2 * np.r_[0, w[1:]]  # no penalty on bias
        w -= lr * grad
    return w

def logistic_posteriors(X, w):
    Phi = np.hstack([np.ones((X.shape[0],1)), X])
    p1 = sigmoid(Phi @ w)
    return (p1, 1.0 - p1)

# ---------------------------------------------------------
# 4) Bayes decision with loss matrix + reject option (Sec. 5.2)
# ---------------------------------------------------------
def decide_with_loss(p1, p2, L, reject_threshold=None):
    # L is 2x2: rows=true class {0,1}, cols=decisions {0,1}
    # Conditional risk R(decision=j|x) = sum_k L_{k,j} p(C_k|x)
    R0 = L[0,0]*p1 + L[1,0]*p2
    R1 = L[0,1]*p1 + L[1,1]*p2
    decision = (R1 < R0).astype(int)
    if reject_threshold is not None:
        maxp = np.maximum(p1, p2)
        decision = np.where(maxp < reject_threshold, -1, decision)  # -1 = reject
    return decision

# ---------------------------------------------------------
# 5) Metrics, Confusion, ROC
# ---------------------------------------------------------
def confusion_counts(y_true, y_pred):
    mask = y_pred != -1  # ignore rejects in counts
    yt, yp = y_true[mask], y_pred[mask]
    TP = np.sum((yt==1)&(yp==1))
    TN = np.sum((yt==0)&(yp==0))
    FP = np.sum((yt==0)&(yp==1))
    FN = np.sum((yt==1)&(yp==0))
    return TP, FP, TN, FN, mask.sum()

def metrics_from_counts(TP, FP, TN, FN):
    N = TP+FP+TN+FN
    acc = (TP+TN)/N if N>0 else np.nan
    prec = TP/(TP+FP) if (TP+FP)>0 else np.nan
    rec = TP/(TP+FN) if (TP+FN)>0 else np.nan
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else np.nan
    fpr = FP/(FP+TN) if (FP+TN)>0 else np.nan
    tpr = rec
    return dict(N=N, acc=acc, prec=prec, rec=rec, f1=f1, fpr=fpr, tpr=tpr)

def roc_curve(y_true, score):  # score = posterior for class 1
    # Sweep threshold τ on p(C1|x) to compute (FPR, TPR)
    taus = np.linspace(0,1,201)
    FPR, TPR = [], []
    for tau in taus:
        y_pred = (score >= tau).astype(int)
        TP = np.sum((y_true==1)&(y_pred==1))
        TN = np.sum((y_true==0)&(y_pred==0))
        FP = np.sum((y_true==0)&(y_pred==1))
        FN = np.sum((y_true==1)&(y_pred==0))
        fpr = FP/(FP+TN) if (FP+TN)>0 else 0.0
        tpr = TP/(TP+FN) if (TP+FN)>0 else 0.0
        FPR.append(fpr); TPR.append(tpr)
    return np.array(FPR), np.array(TPR)

def auc(FPR, TPR):
    # Trapezoidal rule
    idx = np.argsort(FPR)
    return np.trapezoid(TPR[idx], FPR[idx])

# ---------------------------------------------------------
# 6) Fit models
# ---------------------------------------------------------
lda_params = estimate_generative_params(Xtr, ytr, shared_cov=True)
qda_params = estimate_generative_params(Xtr, ytr, shared_cov=False)
w_logreg   = logistic_train(Xtr, ytr, lr=0.2, steps=4000, l2=1e-4)

p1_lda_te, p2_lda_te = posteriors_LDA(Xte, lda_params)
p1_qda_te, p2_qda_te = posteriors_QDA(Xte, qda_params)
p1_log_te, p2_log_te = logistic_posteriors(Xte, w_logreg)

# Loss matrix (example like Fig. 5.6): false negative much worse than false positive
L = np.array([[0, 1],
              [100, 0]], dtype=float)

# Decisions with/without reject option (θ)
yhat_lda = decide_with_loss(p1_lda_te, p2_lda_te, L, reject_threshold=None)
yhat_qda = decide_with_loss(p1_qda_te, p2_qda_te, L, reject_threshold=None)
yhat_log = decide_with_loss(p1_log_te, p2_log_te, L, reject_threshold=None)

# ---------------------------------------------------------
# 7) Evaluate + ROC
# ---------------------------------------------------------
def eval_and_print(name, y_true, y_pred, score):
    TP, FP, TN, FN, Nused = confusion_counts(y_true, y_pred)
    m = metrics_from_counts(TP, FP, TN, FN)
    FPR, TPR = roc_curve(y_true, score)
    A = auc(FPR, TPR)
    print(f"{name}: N_used={Nused}  acc={m['acc']:.3f}  prec={m['prec']:.3f}  rec={m['rec']:.3f}  F1={m['f1']:.3f}  AUC={A:.3f}")
    return FPR, TPR, A, m

FPR_lda, TPR_lda, AUC_lda, _ = eval_and_print("LDA (generative)", yte, yhat_lda, p2_lda_te)
FPR_qda, TPR_qda, AUC_qda, _ = eval_and_print("QDA (generative)", yte, yhat_qda, p2_qda_te)
FPR_log, TPR_log, AUC_log, _ = eval_and_print("LogReg (discriminative)", yte, yhat_log, p2_log_te)

# ---------------------------------------------------------
# 8) Plot decision boundaries + ROC
# ---------------------------------------------------------
def plot_decision(ax, model_name, posterior_fun, params=None, w=None):
    # grid
    x_min, x_max = X[:,0].min()-1.0, X[:,0].max()+1.0
    y_min, y_max = X[:,1].min()-1.0, X[:,1].max()+1.0
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    G = np.c_[gx.ravel(), gy.ravel()]
    if w is not None:
        p1, _ = logistic_posteriors(G, w)
    else:
        p1, _ = posterior_fun(G, params)

    Z = (p1 >= 0.5).astype(int).reshape(gx.shape)
    ax.contourf(gx, gy, Z, alpha=0.25, levels=[-0.5,0.5,1.5],
                colors=['#6baed6','#fd8d3c'])
    # boundary
    CS = ax.contour(gx, gy, p1.reshape(gx.shape), levels=[0.5], linewidths=2, colors='k')
    ax.clabel(CS, fmt={0.5:'p=0.5'}, inline=True, fontsize=9)

    ax.scatter(Xtr[ytr==0,0], Xtr[ytr==0,1], s=16, c='#3182bd', label='C1 (train)', alpha=0.7)
    ax.scatter(Xtr[ytr==1,0], Xtr[ytr==1,1], s=16, c='#e6550d', label='C2 (train)', alpha=0.7)
    ax.set_title(model_name)
    ax.legend(loc='upper left', fontsize=9)

fig, axes = plt.subplots(1,3, figsize=(14,4.3), constrained_layout=True)
plot_decision(axes[0], "LDA (shared Σ)", posteriors_LDA, params=lda_params)
plot_decision(axes[1], "QDA (separate Σ)", posteriors_QDA, params=qda_params)
plot_decision(axes[2], "Logistic Regression", posteriors_LDA, w=w_logreg)  # reuse helper

plt.show()

# ROC
plt.figure(figsize=(5.5,4.5))
plt.plot(FPR_lda, TPR_lda, label=f"LDA (AUC={AUC_lda:.3f})")
plt.plot(FPR_qda, TPR_qda, label=f"QDA (AUC={AUC_qda:.3f})")
plt.plot(FPR_log, TPR_log, label=f"LogReg (AUC={AUC_log:.3f})")
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ---------------------------------------------------------
# 9) (Optional) Reject option illustration
# ---------------------------------------------------------
thetas = np.linspace(0.5, 0.95, 10)
rej_rate, acc_eff = [], []
p1, p2 = p1_log_te, p2_log_te
for th in thetas:
    yrej = decide_with_loss(p1, p2, L, reject_threshold=th)
    used = (yrej!=-1)
    if used.sum()==0:
        rej_rate.append(1.0); acc_eff.append(np.nan); continue
    TP, FP, TN, FN, _ = confusion_counts(yte, yrej)
    m = metrics_from_counts(TP, FP, TN, FN)
    rej_rate.append(1 - used.mean())
    acc_eff.append(m['acc'])

plt.figure(figsize=(5.5,4))
plt.plot(rej_rate, acc_eff, marker='o')
plt.xlabel("Rejection rate")
plt.ylabel("Accuracy on non-rejected")
plt.title("Reject option trade-off (Logistic)")
plt.grid(alpha=0.3)
plt.show()
