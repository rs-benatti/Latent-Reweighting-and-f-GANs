import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

        # Applying spectral normalization to each linear layer can stabilize GAN training
        self.fc1 = nn.utils.spectral_norm(self.fc1)
        self.fc2 = nn.utils.spectral_norm(self.fc2)
        self.fc3 = nn.utils.spectral_norm(self.fc3)
        self.fc4 = nn.utils.spectral_norm(self.fc4)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

        # Dropout
        self.dropout = nn.Dropout(0.3)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        x = self.dropout(x)
        return torch.sigmoid(self.fc4(x))
