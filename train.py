import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import Generator, Discriminator
from utils import D_train, G_train, save_models

def main(epochs = 100, lr = 0.0002, batch_size = 64, mnist_size = 0, f_divergence = 0):
    divergences_dict = {
        0: "BCE Loss",
        1: "GAN",
        2: "KL",
        3: "Reverse KL",
        4: "Pearson X^2",
        5: "Squared Hellinger",
        6: "Jensen-Shannon",
    }

    print(f"Used divergence: {divergences_dict[f_divergence]}")
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

    if mnist_size != 0:
        mnist_trainset_reduced = torch.utils.data.random_split(train_dataset, [mnist_size, len(train_dataset)-mnist_size])[0] 
        train_loader = torch.utils.data.DataLoader(mnist_trainset_reduced, batch_size=batch_size, shuffle=True)
        # download test dataset
        max_mnist_size = mnist_size // 2
        mnist_testset_reduced = torch.utils.data.random_split(test_dataset, [max_mnist_size, len(test_dataset)-max_mnist_size])[0] 
        test_loader = torch.utils.data.DataLoader(mnist_testset_reduced, batch_size=batch_size, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, shuffle=False)
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

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr = lr, betas=(0.5, 0.999))

    print('Start Training :')
    G_loss_history = []
    D_loss_history = []
    n_epoch = epochs
    for epoch in trange(1, n_epoch+1, leave=True):           
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            for k in range(0, 1):
                D_loss = D_train(x, G, D, D_optimizer, criterion, f_divergence)
            G_loss = G_train(x, G, D, G_optimizer, criterion, f_divergence)
            D_loss_history.append(D_loss)
            G_loss_history.append(G_loss)
        
        if epoch % 10 == 0:
            generate(G)
            try:
                save_models(G, D, 'checkpoints')
            except:
                pass
    
    print('Training done')
    return G, D, G_loss_history, D_loss_history        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")
    parser.add_argument("--mnist_size", type=int, default=0, 
                        help="Size of mini-batches for SGD")
    parser.add_argument("--f_divergence", type=int, default=0, 
                        help="Size of mini-batches for SGD")
    args = parser.parse_args()
    main(epochs = args.epochs, lr = args.lr, batch_size = args.batch_size, mnist_size = args.mnist_size, f_divergence = args.f_divergence)
  

def generate(G, batch_size=2048):
    n_samples = 9
    if torch.cuda.is_available():
        device = torch.device("cuda")
        #print('GPU is available')
    else:
        device = torch.device("cpu")
        #print('GPU is not available')


    #print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = G
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    #print('Model loaded.')



    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    i = 0
    r = 3
    c = 3

    with torch.no_grad():
        fig, axs = plt.subplots(r, c)
        for i in range(r):
          for j in range(c):
            #black and white images
            z = torch.randn(9, 100).to(device)
            x = model(z)
            x = x.reshape(9, 28, 28)
            for k in range(x.shape[0]):
                img = x[k].cpu().numpy()
                axs[i,j].imshow(img, cmap='gray')
                axs[i,j].axis('off')
        plt.show()