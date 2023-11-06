import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import model
import utils

# Set up data loading and transformations for MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
mnist_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)
mnist_dim = 784
discriminator = model.Discriminator(mnist_dim)
discriminator = utils.load_discriminator(discriminator, 'checkpoints')

# Set the model to evaluation mode
discriminator.eval()

correct_real, correct_fake = 0, 0
total_real, total_fake = 0, 0

# Evaluate on real images
with torch.no_grad():
    for real_batch, _ in dataloader:
        real_output = discriminator(real_batch.view(-1, 28 * 28))
        predicted_real = real_output.int()
        correct_real += (predicted_real == 1).sum().item()
        total_real += real_output.size(0)

# Evaluate on fake/generated images
samples_folder = "samples"

for i in range(len(dataloader.dataset)):
    image_path = os.path.join(samples_folder, f"{i}.png")
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = transform(image)
    fake_output = discriminator(image.view(-1, 28 * 28))
    predicted_fake = fake_output.int()
    if (predicted_fake.item() == 0):
        correct_fake += 1
    total_fake += fake_output.size(0)

# Calculate accuracy for real and fake images
accuracy_real = correct_real / total_real
accuracy_fake = correct_fake / total_fake

print(f"Accuracy on real images: {accuracy_real:.2f}")
print(f"Accuracy on fake images: {accuracy_fake:.2f}")
