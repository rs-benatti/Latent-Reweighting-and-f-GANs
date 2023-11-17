import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


from model_rs import Generator, Discriminator, WeightNetwork, train_weight_network
from utils import D_train, G_train, save_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=400,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Size of mini-batches for SGD")
    parser.add_argument("--mnist_size", type=int, default=10000,
                        help="number of real images taken")


    args = parser.parse_args()

        # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('GPU is available')
    else:
        device = torch.device("cpu")
        print('GPU is not available')


    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    mnist_size = args.mnist_size
    if mnist_size != 0:
        mnist_trainset_reduced = torch.utils.data.random_split(train_dataset, [mnist_size, len(train_dataset)-mnist_size])[0] 
        train_loader = torch.utils.data.DataLoader(mnist_trainset_reduced, batch_size=args.batch_size, shuffle=True)
        # download test dataset
        max_mnist_size = mnist_size // 2
        mnist_testset_reduced = torch.utils.data.random_split(test_dataset, [max_mnist_size, len(test_dataset)-max_mnist_size])[0] 
        test_loader = torch.utils.data.DataLoader(mnist_testset_reduced, batch_size=args.batch_size, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).to(device)
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).to(device)


    # model = DataParallel(model).to(device)
    print('Model loaded.')
    # Optimizer 

    # define loss
    criterion = nn.BCELoss()

    w_net = WeightNetwork().to(device)
    w_optimizer = optim.Adam(w_net.parameters(), lr=args.lr)

    # Hyperparameters for weights training : w_ϕ
    lambda1, lambda2, m, nd = 10, 3, 3, 1

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr, betas=(0.5, 0.999))

    print('Start Training :')
    
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch+1, leave=True):           
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).to(device)

            # Training the discriminator
            D_train(x, G, D, D_optimizer, criterion)

            # Training the generator
            G_train(x, G, D, G_optimizer, criterion)

            # Training the weight network
            train_weight_network(x, G, D, w_net, w_optimizer, D_optimizer, criterion, lambda1, lambda2, m)

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

    print('Training done')
