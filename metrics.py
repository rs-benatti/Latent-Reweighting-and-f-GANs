import os
import torch
from torchvision import datasets, transforms
from PIL import Image

# Define the transformation
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create the directory if it does not exist
output_dir = 'originals'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the images
for i, (image, _) in enumerate(mnist_dataset):
    if i >= 10000:  # Stop after saving 10,000 images
        break
    # Convert the tensor to PIL image
    pil_image = transforms.ToPILImage()(image)
    pil_image.save(os.path.join(output_dir, f'mnist_{i}.png'))

print("Images saved.")
