#Example of batch normalization and layer normalization in a neural network using PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# Set random seed for reproducibility
torch.manual_seed(0)
# Define a simple neural network with BatchNorm and LayerNorm
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, useBN=True, useLN=True):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.useBN = useBN
        self.useLN = useLN

    def forward(self, x):
        if self.useBN:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        if self.useLN:
            x = F.relu(self.ln1(self.fc2(x)))
        else:
            x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

    
# Generate synthetic data
def generate_data(num_samples=1000, input_size=10):
    X = torch.randn(num_samples, input_size)
    y = (X.sum(dim=1) > 0).long()  # Simple binary classification
    return X, y
# Training function
def train_model(model, X, y, epochs=20, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    save_loss = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        save_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    # Plot training loss
    plt.plot(save_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    
# Main execution
if __name__ == "__main__":
    input_size = 10
    hidden_size = 20
    output_size = 2
    num_samples = 1000

    # Generate data
    X, y = generate_data(num_samples, input_size)

    # Initialize and train the model
    # model = SimpleNN(input_size, hidden_size, output_size)
    model = SimpleNN(input_size, hidden_size, output_size, useBN=True, useLN=True)
    train_model(model, X, y)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).float().mean()
        print(f'Accuracy: {accuracy.item()*100:.2f}%')

    # Visualize the effect of BatchNorm and LayerNorm
    with torch.no_grad():
        # Get outputs before and after normalization layers
        x = F.relu(model.fc1(X))
        bn_output = model.bn1(x)
        x = F.relu(model.fc2(bn_output))
        ln_output = model.ln1(x)
        # Plot histograms
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(bn_output.numpy().flatten(), bins=30, alpha=0.7, color='b')
        plt.title('BatchNorm Output Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.subplot(1, 2, 2)
        plt.hist(ln_output.numpy().flatten(), bins=30, alpha=0.7, color='g')
        plt.title('LayerNorm Output Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
