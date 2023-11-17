import torch 
import torchvision
import os
import argparse


from model_rs import Generator, WeightNetwork, latent_rejection_sampling, latent_rejection_sampling_and_gradient_ascent
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    parser.add_argument("--N", type=int, default=10,
                        help="Number of gradient ascent steps for Latent Gradient Ascent.")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Step size for Latent Gradient Ascent.")

    args = parser.parse_args()

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('GPU is available')
    else:
        device = torch.device("cpu")
        print('GPU is not available')


    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    # Loading the generator
    model = Generator(g_output_dim = mnist_dim).to(device)
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    # Loading the Weight Network
    w_net = WeightNetwork().to(device)
    # w_net = load_model(w_net, 'checkpoints')  # Load saved weight network
    w_net = torch.nn.DataParallel(w_net).to(device)

    print('Model loaded.')

    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0

    m, epsilon, N = 3, 0.01, 10

    with torch.no_grad():
        while n_samples < 10000:
            # Generate z using Latent Rejection Sampling
            z_latent = latent_rejection_sampling(w_net, m, z_dim=100).to(device)

            # Generate the image
            x = model(z_latent)

            x = x.reshape(1, 28, 28)
            torchvision.utils.save_image(x, os.path.join('samples', f'{n_samples}.png'))

            n_samples += 1
