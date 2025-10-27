#============================================
# Create a GIF animation demonstrating gradient 
# descent on a complex loss surface. The 
# animation should show the optimization path 
# taken by gradient descent starting from a 
# given initial point.
# Let's use pytorch to compute gradients
#============================================
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import imageio

# create optimizer:
class SGDOptimizer:
    def __init__(self, params, lr=0.01, momentum=0.0, nesterov=False):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = [torch.zeros_like(p) for p in self.params]
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.momentum > 0.0:
                v = self.velocities[i]
                v.mul_(self.momentum).add_(grad)
                if self.nesterov:
                    update = grad + self.momentum * v
                else:
                    update = v
            else:
                update = grad
            p.data -= self.lr * update
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# complex loss function
def complex_loss(xy):
    x, y = xy[0], xy[1]
    return torch.sin(3*x)*torch.cos(3*y) + 0.5*(x**2 + y**2)

# create grid for contour plot
xlin = np.linspace(-3, 3, 200)
ylin = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(xlin, ylin)
Z = complex_loss(torch.tensor([X, Y], dtype=torch.float32)).numpy()
# create output directory
out_dir = "temp/gradient_descent_frames"
os.makedirs(out_dir, exist_ok=True)
# initialize point
xy = torch.tensor([2.5, 2.5], dtype=torch.float32, requires_grad=True)
# optimizer = SGDOptimizer([xy], lr=0.01, momentum=0.9, nesterov=False)
optimizer = SGDOptimizer([xy], lr=0.5, momentum=0.0, nesterov=False)
# optimizer = SGDOptimizer([xy], lr=0.5, momentum=0.0, nesterov=False)
# store frames for GIF
frames = []
# gradient descent loop
for epoch in range(50):
    loss = complex_loss(xy)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # plot current state
    plt.figure(figsize=(6, 6))
    
    # make plot surface using plot_surface
    plt.plot_surface = True  # set to True for 3D surface plot
    if plt.plot_surface:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
        # ax.scatter(xy[0].item(), xy[1].item(), loss.item(), color='r', s=50)
        ax.scatter3D(xy[0].item(), xy[1].item(), loss.item()+1, color='r', s=50, alpha=1.0)
        # Set camera angle 
        ax.view_init(elev=45, azim=-65)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Loss')
        ax.set_title(f'Gradient Descent Step {epoch+1}')
    else:
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Loss')
        plt.plot(xy[0].item(), xy[1].item(), 'ro', markersize=10)
        plt.title(f'Gradient Descent Step {epoch+1}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
    # save frame
    frame_path = os.path.join(out_dir, f'frame_{epoch:03d}.png')
    plt.savefig(frame_path)
    plt.close()
    frames.append(imageio.imread(frame_path))

# create GIF
gif_path = 'gradient_descent.gif'
imageio.mimsave(gif_path, frames, fps=5)
print(f"GIF saved to {gif_path}")


# # cleanup frames
# for epoch in range(50):
#     frame_path = os.path.join(out_dir, f'frame_{epoch:03d}.png')
#     os.remove(frame_path)
# os.rmdir(out_dir)
# print("Temporary frames cleaned up.")

