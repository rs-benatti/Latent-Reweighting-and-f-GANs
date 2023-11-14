import torch
import os

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU is available')
else:
    device = torch.device("cpu")
    print('GPU is not available')


def D_train(x, G, D, D_optimizer, criterion, f_divergence):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output = D(x_real)

    if f_divergence == 0: # BCE Loss
        D_real_loss = criterion(torch.sigmoid(D_output), y_real) 
    elif f_divergence == 1: # Regular GAN
        D_real_loss = -GAN_loss(D_output)
        D_real_loss.backward()
    elif f_divergence == 2: # Regular GAN
        D_real_loss = -KL_loss(D_output)
        D_real_loss.backward()
    elif f_divergence == 3: # Regular GAN
        D_real_loss = -reverse_KL_loss(D_output)
        D_real_loss.backward()
    elif f_divergence == 4: # Pearson 
        D_real_loss = -pearson_chi_loss(D_output)
        D_real_loss.backward()
    elif f_divergence == 5: # Squared Hellinger
        D_real_loss = -squared_hellinger_loss(D_output)
        D_real_loss.backward()
    elif f_divergence == 6: # Jensen Shannon
        D_real_loss = -jensen_shannon(D_output)
        D_real_loss.backward()

    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).to(device)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(device)

    D_output =  D(x_fake)
    
    if f_divergence == 0: # BCE Loss
        D_fake_loss = criterion(torch.sigmoid(D_output), y_fake) 
    elif f_divergence == 1: # Regular GAN
        D_fake_loss = -GAN_loss_conjugate(D_output)
        D_fake_loss.backward()
        D_optimizer.step()
    elif f_divergence == 2: # Regular GAN
        D_fake_loss = -KL_loss_conjugate(D_output)
        D_fake_loss.backward()
        D_optimizer.step()
    elif f_divergence == 3: # Regular GAN
        D_fake_loss = -reverse_KL_loss_conjugate(D_output)
        D_fake_loss.backward()
        D_optimizer.step()
    elif f_divergence == 4: # Pearson 
        D_fake_loss = -pearson_chi_loss_conjugate(D_output)
        D_fake_loss.backward()
        D_optimizer.step()
    elif f_divergence == 5: # Squared Hellinger
        D_fake_loss = -squared_hellinger_loss_conjugate(D_output)
        D_fake_loss.backward()
        D_optimizer.step()
    elif f_divergence == 6: # Jensen Shannon
        D_fake_loss = -jensen_shannon_conjugate(D_output)
        D_fake_loss.backward()
        D_optimizer.step()
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    if f_divergence == 0: # BCE Loss
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()
    else:
        D_loss = D_real_loss + D_fake_loss
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion, f_divergence):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)
    y = torch.ones(x.shape[0], 1).to(device)
                 
    G_output = G(z)
    D_output = D(G_output)

    if f_divergence == 0: # BCE Loss
        G_loss = criterion(torch.sigmoid(D_output), y)
    elif f_divergence == 1: # Regular GAN
        G_loss = -GAN_loss(D_output)
    elif f_divergence == 2: # KL
        G_loss = -KL_loss(D_output)
    elif f_divergence == 3: # Reverse KL
        G_loss = -reverse_KL_loss(D_output)
    elif f_divergence == 4: # Pearson 
        G_loss = -pearson_chi_loss(D_output)
    elif f_divergence == 5: # Squared Hellinger
        G_loss = -squared_hellinger_loss(D_output)
    elif f_divergence == 6: # Jensen Shannon
        G_loss = -jensen_shannon(D_output)
    
    
    
    #print(f"G_loss = {G_loss}")
    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder, f_divergence):
    if f_divergence == 0: # BCE Loss
        model = 'G.pth'
        print(f"Used model: {model}")
    elif f_divergence == 1: # Regular GAN
        model = 'G_f_GAN.pth'
        print(f"Used model: {model}")
    elif f_divergence == 2: # KL
        model = 'G_KL.pth'
        print(f"Used model: {model}")
    elif f_divergence == 3: # Reverse KL
        model = 'G_reverse_KL.pth'
        print(f"Used model: {model}")
    elif f_divergence == 4: # Pearson 
        model = 'G_pearson.pth'
        print(f"Used model: {model}")
    elif f_divergence == 5: # Squared Hellinger
        model = 'G_hellinger.pth'
        print(f"Used model: {model}")
    elif f_divergence == 6: # Jensen Shannon
        model = 'G_jensen_shannon.pth'
        print(f"Used model: {model}")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        ckpt = torch.load(os.path.join(folder,model))
    else:
        ckpt = torch.load(os.path.join(folder,model), map_location=torch.device('cpu'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def load_discriminator(D, folder):
    # Check if GPU is available
    if torch.cuda.is_available():
        ckpt = torch.load(os.path.join(folder,'D.pth'))
    else:
        ckpt = torch.load(os.path.join(folder,'D.pth'), map_location=torch.device('cpu'))
    D.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return D

def GAN_loss(output): # The minus is because we do a gradient ascending
    return torch.mean(-torch.log(1.0+torch.exp(-output)))

def GAN_loss_conjugate(output):
    return torch.mean(torch.log(1.0-torch.exp(-torch.log(1.0+torch.exp(-output)))))

def KL_loss(output):
    return torch.mean(output)

def KL_loss_conjugate(output):
    return -torch.mean(torch.exp(output - 1))

def reverse_KL_loss(output):
    return torch.mean(-torch.exp(-output))

def reverse_KL_loss_conjugate(output):
    return -torch.mean(-1 - torch.log(torch.exp(-output)))

def pearson_chi_loss(output):
    return torch.mean(output)

def pearson_chi_loss_conjugate(output):
    return -torch.mean(0.25 * output**2 + output)

def squared_hellinger_loss(output):
    return torch.mean(1 - torch.exp(-output))

def squared_hellinger_loss_conjugate(output):
    return -torch.mean((1 - torch.exp(-output))/(1 - (1 - torch.exp(-output))))

def jensen_shannon(output):
    return torch.mean(torch.log(torch.tensor(2.))-torch.log(1.0+torch.exp(-output)))

def jensen_shannon_conjugate(output):
    return -torch.mean(-torch.log(2 - torch.exp(torch.log(torch.tensor(2.))-torch.log(1.0+torch.exp(-output)))))