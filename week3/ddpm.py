# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms, datasets
from unet import Unet

class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x0):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """
        
        ### Implement Algorithm 1 here ###
        # Algorithm 1 in DDPM paper
        batch_size = x0.shape[0]
        t = torch.randint(1, T, (batch_size,), device=x0.device)
        epsilon = torch.randn_like(x0) # sample from ε ~ 𝒩(0, I)
        alpha_bar_t = self.alpha_cumprod[t]
        
        # negative elbo eq. 14
        epsilon_theta = self.network(torch.sqrt(alpha_bar_t).unsqueeze(1) * x0 + torch.sqrt(1 - alpha_bar_t.unsqueeze(1)) * epsilon, t.unsqueeze(1))
        loss = F.mse_loss(epsilon_theta, epsilon)
        elbo = torch.mean(loss, dim=list(range(1, loss.ndim))) # mean over all dimensions but the batch dimension
        
        return elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            t_tensor = torch.full((shape[0], 1), t, device=self.alpha.device, dtype=torch.float)
            ### Implement the remaining of Algorithm 2 here ###
            z = torch.randn_like(x_t).to(self.alpha.device) if t > 1 else torch.zeros_like(x_t).to(self.alpha.device)
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_cumprod[t]
            beta_t = self.beta[t]
            epsilon_theta = self.network(x_t, t_tensor)
            sigma_t = torch.sqrt(beta_t)
            
            x_t = (1/torch.sqrt(alpha_t)) * (x_t - ( ((1-alpha_t)/torch.sqrt(1-alpha_bar_t)) * epsilon_theta)) + sigma_t * z
            
        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


# python3 ddpm.py test --data tg --model model.pt --device mps --batch-size 64 --epochs 1 --lr 1e-3
# python3 ddpm.py train --data tg --model model.pt --device mps --batch-size 64 --epochs 1 --lr 1e-3
# python3 ddpm.py sample_MNIST --data tg --model model.pt --device mps --batch-size 64 --epochs 1 --lr 1e-3 --samples samples.png

class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)

if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test', 'sample_MNIST'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),  # Add small random noise
        transforms.Lambda(lambda x: (x - 0.5) * 2.0),  # Normalize to [-1, 1]
        transforms.Lambda(lambda x: x.flatten())  # Flatten the image
    ])

    train_data = datasets.MNIST('data/',
                                train=True,
                                download=True,
                                transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    
    # Generate the data for toy example
    # n_data = 10000000
    # toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
    # transform = lambda x: (x-0.5)*2.0
    # train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
    # Get the dimension of the dataset
    # D = next(iter(train_loader)).shape[1]
    # num_hidden = 64

    # Define the network
    network = Unet()

    # Set the number of steps in the diffusion process
    T = 1000

    # Define model
    model = DDPM(network, T=T, beta_T=1e-3).to(args.device)

    # Choose mode to run
    if args.mode == 'train':
        print(f"Training will run on: {args.device}")
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample_MNIST':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()
        with torch.no_grad():
            # U-Net model expects input tensors with the shape (batch_size, 1, 28, 28) but flattened: (batch_size, 1*28*28) -> (batch_size, 784)
            # so we can generate 10 MNIST images like so
            samples = model.sample((10, 784))
        print(samples)
        samples_img = samples.reshape(-1, 1, 28, 28)  # shape: (10, 1, 28, 28)
        # since we have data in [-1,1], convert it to [0,1]
        samples_img = (samples_img + 1) / 2
        save_image(samples_img, "sample_mnist.png", nrow=5)
        
        
    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((10000,D))).cpu() 

        # Transform the samples back to the original space
        samples = samples /2 + 0.5

        # Plot the density of the toy data and the model samples
        coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
        prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
        ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
        ax.set_xlim(toy.xlim)
        ax.set_ylim(toy.ylim)
        ax.set_aspect('equal')
        fig.colorbar(im)
        plt.savefig(args.samples)
        plt.close()
