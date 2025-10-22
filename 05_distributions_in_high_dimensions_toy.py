# ============================================
# Code to visualize distributions in high 
# dimensions using numpy
# ============================================

import numpy as np

# Gaussian number generator for 1D
def gaussian_1d(size=1):
    return np.random.randn(size)


# uniform number generator for 1D
def uniform_1d(low=-0.5, high=0.5, size=1):
    return low + (high - low) * np.random.rand(size)

# random number gnerator of ND uncorrelated Gaussian using the 1D generator
def gaussian_nd(d, size):
    samples = np.empty((size, d))
    for i in range(d):
        samples[:, i] = gaussian_1d(size)
    return samples

# random number generator of ND uniform using the 1D generator
def uniform_nd(d, size):
    samples = np.empty((size, d))
    for i in range(d):
        samples[:, i] = uniform_1d(size=size)
    return samples

if __name__ == "__main__":

    # 1D Examples
    import matplotlib.pyplot as plt
    np.random.seed(0)
    n_samples = 1000
    mu_1d = 0.0
    sigma_1d = 1.0
    samples_1d = gaussian_1d(n_samples)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(samples_1d, bins=30, density=True, alpha=0.6, color='g')
    plt.title('1D Gaussian Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    low_1d = -2.0
    high_1d = 2.0
    samples_uniform_1d = uniform_1d(low_1d, high_1d, n_samples)
    plt.subplot(1, 2, 2)
    plt.hist(samples_uniform_1d, bins=30, density=True, alpha=0.6, color='b')
    plt.title('1D Uniform Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

    # Plot histograms for different dimensions in different subplots
    dimensions = [1, 2, 5, 10, 50, 100]
    plt.figure(figsize=(12, 10))
    for i, d in enumerate(dimensions):
        gaussian_samples = gaussian_nd(d, n_samples)
        r_samples = np.linalg.norm(gaussian_samples, axis=1)
        # print shape of r_samples
        plt.subplot(3, 2, i + 1)
        plt.hist(r_samples, bins=30, density=True, alpha=0.6, color='k')
        # plt.xlim(0, np.max(r_samples)*1.1)
        plt.xlim(0, 12)
        plt.title(f'Radius Distribution in {d}D Gaussian')
        plt.xlabel('Radius')
        plt.ylabel('Density')
        plt.grid(True)
        plt.tight_layout()

    plt.show()
    
    # # Plot histograms for different dimensions in different subplots an uniform distribution
    plt.figure(figsize=(12, 10))
    for i, d in enumerate(dimensions):
        uniform_samples = uniform_nd(d, n_samples)
        r_samples = np.linalg.norm(uniform_samples, axis=1)
        plt.subplot(3, 2, i + 1)
        plt.hist(r_samples, bins=30, density=True, alpha=0.6, color='m')
        # plt.xlim(0, np.max(r_samples)*1.1)
        plt.xlim(0, 3.5)
        plt.title(f'Radius Distribution in {d}D Uniform')
        plt.xlabel('Radius')
        plt.ylabel('Density')
        plt.grid(True)
        plt.tight_layout()
    plt.show()

    # d = 2
    # i = 1
    # uniform_samples = uniform_nd(d, n_samples)
    # r_samples = np.linalg.norm(uniform_samples, axis=1)
    # plt.subplot(3, 2, i + 1)
    # plt.hist(r_samples, bins=30, density=True, alpha=0.6, color='m')
    # # plt.xlim(0, np.max(r_samples)*1.1)
    # plt.title(f'Radius Distribution in {d}D Uniform')
    # plt.xlabel('Radius')
    # plt.ylabel('Density')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()