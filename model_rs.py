import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

        #Dropout layers
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout1(x)

        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout2(x)

        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout1(x)

        return torch.sigmoid(self.fc4(x))

class WeightNetwork(nn.Module):
    def __init__(self):
        super(WeightNetwork, self).__init__()
        # Define the architecture of the weight network here
        self.fc1 = nn.Linear(100, 128)  # Assuming the latent vector size is 100
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output a single weight value

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(z))  # Output in range [0,1]

def gradient_penalty(discriminator, real_data, fake_data):
    batch_size = real_data.size(0)

    # Génération de points interpolés
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_data).to(device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)

    # Calcul des prédictions du discriminateur
    interpolated_preds = discriminator(interpolated)

    # Calcul des gradients
    gradients = torch.autograd.grad(
        outputs=interpolated_preds,
        inputs=interpolated,
        grad_outputs=torch.ones(interpolated_preds.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Calcul de la pénalité de gradient
    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = torch.mean((gradients_norm - 1) ** 2)
    return gradient_penalty


def train_weight_network(real_data, G, D, w_net, w_optimizer, D_optimizer, criterion, lambda1, lambda2, m, nd=1):
    w_net.train()
    D.train()

    batch_size = real_data.size(0)
    real_data = real_data.view(batch_size, -1).to(device)

    # Update Discriminator Dα
    for _ in range(nd):  # nd is the number of D updates per w_net update
        z = torch.randn(batch_size, 100).to(device)
        w_net_output = w_net(z)
        real_output = D(real_data)
        fake_data = G(z)
        fake_output = D(fake_data)

        emd = torch.mean(real_output - w_net_output * fake_output)
        gp = gradient_penalty(D, real_data, fake_data)  # Function to compute gradient penalty
        D_optimizer.zero_grad()
        loss_D = -emd + gp
        loss_D.backward()
        D_optimizer.step()

    # Update Weight Network w_ϕ
    z = torch.randn(batch_size, 100).to(device)
    w_net_output = w_net(z)
    fake_data = G(z)
    fake_output = D(fake_data)
    emd = torch.mean(w_net_output * fake_output)
    rnorm = torch.mean(w_net_output) - 1
    rclip = torch.mean(F.relu(w_net_output - m))
    loss_w = emd + lambda1 * rnorm ** 2 + lambda2 * rclip ** 2
    w_optimizer.zero_grad()
    loss_w.backward()
    w_optimizer.step()

def latent_rejection_sampling(w_net, m, z_dim=100):
    w_net.eval()
    while True:
        # Sample z from the prior Z (assuming a normal distribution)
        z = torch.randn(1, z_dim).to(device)

        # Sample alpha uniformly between 0 and 1
        alpha = torch.rand(1).item()

        # Calculate the importance weight for z
        w = w_net(z).item()

        # Check if the sampled z should be accepted
        if w / m >= alpha:
            break

    return z
