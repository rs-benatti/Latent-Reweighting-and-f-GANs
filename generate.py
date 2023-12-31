import torch 
import torchvision
import os
import argparse


from model import Generator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    parser.add_argument("--sample_size", type=int, default=10000,
                      help="The batch size to use for training.")
    parser.add_argument("--divergence", type=int, default=0,
                      help="The divergence used.")
    args = parser.parse_args()

    divergences_dict = {
        0: "BCELoss",
        1: "GAN",
        2: "KL",
        3: "ReverseKL",
        4: "Pearson",
        5: "Hellinger",
        6: "Jensen_Shannon",
    }

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

    model = Generator(g_output_dim = mnist_dim).to(device)
    model = load_model(model, 'checkpoints', args.divergence)
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    print('Model loaded.')

    if args.divergence == 0:
        dirname = 'samples'
    else:
        dirname = 'divergence_samples/samples' + divergences_dict[args.divergence]

    print('Start Generating')
    os.makedirs(dirname, exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<args.sample_size:
            z = torch.randn(args.batch_size, 100).to(device)
            x = model(z)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<args.sample_size:
                    torchvision.utils.save_image(x[k:k+1], os.path.join(dirname, f'{n_samples}.png'))         
                    n_samples += 1


    
