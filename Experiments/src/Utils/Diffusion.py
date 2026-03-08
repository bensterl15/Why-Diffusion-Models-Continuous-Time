import numpy as np
import torch
from tqdm import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
import Plot
import torch.nn.utils as nn_utils

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="bensterl15-stony-brook-university",
    # Set the wandb project where this run will be logged.
    project="Overfitting",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

N_STEPS = 1000
# ====================================================================
# Training Configuration Class
# ====================================================================
class TrainingConfig:
    '''
    TrainingConfig: Class containing all information on the data, device, LR,
    Number of SGD steps, paths for saving, etc.
    '''
    DATASET = ''                # Dataset name (MNIST, CIFAR, Imagenet, CelebA)
    IMG_SHAPE = (3, 32, 32)     # Fixed input image size
    BATCH_SIZE = 128            # Batch size
    DEVICE = 'cuda:0'           # Name of the device to be used
    LR = 1e-4                   # Learning rate
    N_STEPS = int(1e5)+1        # Number of SGD steps
    TIMESTEPS = N_STEPS            # Define number of diffusion timesteps
    path_save = ''              # Path for saving plots and models
    path_data = ''              # Path for the data
    CENTER = True               # Whether the dataset should be centered
    STANDARDIZE = False         # Whether the dataset should be standardized
    n_images = 500              # Number of images per class
    NUM_WORKERS = 2             # Number of workers
    
    mean = 0                    # Mean of the dataset (to be computed)
    std = 0                     # Std of the dataset (to be computed)

def get(element: torch.Tensor, t: torch.Tensor, dim: int=2):
    '''
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    '''
    ele = element.gather(-1, t)
    if dim == 4:
        return ele.reshape(-1, 1, 1, 1)
    elif dim == 3:
        return  ele.reshape(-1, 1, 1)
    elif dim == 2:
        return  ele.reshape(-1, 1)


# ====================================================================
# Diffusion class
# ====================================================================

NUM_EPS = 1e-4

class DiffusionConfig:
    '''
    ClassDiffusion: Class containing information related to the 
    diffusion process (number of steps, device, variance, etc.)
    '''
    def __init__(self, n_steps=N_STEPS, img_shape=(3, 32, 32), device='cpu'):
        self.beta0 = 0.001
        self.beta1 = 10.0
        self.n_steps = n_steps
        self.img_shape = img_shape
        self.device = device

    def beta(self, t):
        return self.beta0 + t * (self.beta1 - self.beta0)

    def B(self, t):
        return self.beta0*t + 0.5*(self.beta1 - self.beta0)*t**2

    
# ====================================================================
# Diffusion functions
# ==================================================================== 
def forward_diffusion(df, x0, timesteps, config):
    dim = len(x0.shape)
    # Generate noise realisation with the same size as a batch of images
    eps = torch.randn_like(x0)

    df_B = df.B(timesteps).view(-1, 1, 1, 1) if dim == 4 else df.B(timesteps).view(-1, 1, 1) if dim == 3 else df.B(timesteps).view(-1, 1)
    # Apply the forward diffusion kernel at times timesteps
    mean      = x0 * torch.exp(-df_B/2) 
    std_dev   = torch.sqrt(1 - (1 - NUM_EPS)*torch.exp(-df_B))
    sample_a  = mean + std_dev * eps 
    
    # Return the noisy image and the noise realisation
    return sample_a, eps

''' I believe this is unused:
@torch.no_grad()
def sample_diffusion(model, n_images=25, config=TrainingConfig()):
    # Gaussian random fields
    x_init = torch.randn(n_images, config.IMG_SHAPE[0], config.IMG_SHAPE[1], config.IMG_SHAPE[2]).to(config.DEVICE)
    x = x_init.clone()
    for i in trange(config.TIMESTEPS):
        t = i
        # For each time step, generate a denoised image
        x = model(x, torch.full((n_images, 1), t, dtype=torch.long, device=config.DEVICE))
    return x, x_init
'''


@torch.no_grad()
def sample_diffusion_from_noise(model, n_images=25, config=TrainingConfig(), 
                                df=DiffusionConfig(), dim=3):
    
    # Generate n_images starting points from N(0, 1)
    if dim == 4: # Assumes [B, C, H, W] for 2d
        x_init = torch.randn(n_images, config.IMG_SHAPE[0], config.IMG_SHAPE[1], 
                             config.IMG_SHAPE[2]).to(config.DEVICE)
    elif dim == 3: # Assumes [B, C, N] for 1d
        x_init = torch.randn(n_images, config.IMG_SHAPE[0], 
                             config.IMG_SHAPE[1]).to(config.DEVICE)
    elif dim == 2: # Assumes [B, N] for 1d (no channels)
        x_init = torch.randn(n_images, config.IMG_SHAPE[1]).to(config.DEVICE)
    x = x_init.clone()
    
    model.eval()
    dt = torch.tensor(1.0 / N_STEPS, device=config.DEVICE, dtype=torch.float32)
    for t in reversed(range(1, N_STEPS + 1)):
        # Time tensor
        ts = torch.ones(n_images, dtype=torch.long, device=config.DEVICE) * t / N_STEPS
        #print(f'ts: {ts[0].item():.4f}, t: {t:d}', flush=True)


        # Generate one realisation of the noise
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        
        # Predict the noise at times ts
        score = model(x, ts)
        
        # Get scaling quantities
        beta_t = df.beta(ts).view(n_images, 1, 1, 1) if dim == 4 else df.beta(ts).view(n_images, 1, 1) if dim == 3 else df.beta(ts).view(n_images, 1)
        
        #print(f'beta_t statistics: mean {beta_t.mean().item():.6f}, min {beta_t.min().item():.6f}, max {beta_t.max().item():.6f}', flush=True)

        # Langevin sampling from VPSDE:
        x = x + beta_t * (x / 2 + score) * dt
        x = x + torch.sqrt(beta_t) * z * torch.sqrt(dt)

    return x, x_init

 
#==========================================
# Training functions
#==========================================
def train_one_batch(X, model, optimizer, loss_fn, 
                    config=TrainingConfig(), 
                    df=DiffusionConfig()):
    model.train()
    
    # Generate random times (continuous uniform in [0,1))
    ts = torch.rand(X.shape[0], device=config.DEVICE)
    
    # Extract noisy images from times t
    X_t, noise_t = forward_diffusion(df, X, ts, config)
    X_t = X_t.to(config.DEVICE)
    
    # Apply the model
    score = model(X_t.float(), ts)
    df_B = df.B(ts).view(-1, 1, 1, 1) if len(X.shape) == 4 else df.B(ts).view(-1, 1, 1) if len(X.shape) == 3 else df.B(ts).view(-1, 1)
    std_dev = torch.sqrt(1 - (1 - NUM_EPS)*torch.exp(-df_B))
    
    #print(f"std_dev min: {std_dev.min().item()}", flush=True)
    
    Y = - std_dev * score.float()
    # The loss is comparing the predicted and true noises
    loss = loss_fn(noise_t, Y)
    
    run.log({"loss": loss.item()})

    # Update parameters of the model
    optimizer.zero_grad()
    loss.backward()
    #nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Return the current loss and the batch of noisy images
    return loss.detach().item(), X_t


def train(model, trainloader, optimizer, config, df, loss_fn,
          sweep=1., times_save=[], offset=0, suffix='', generate=False):
    
    n_steps = offset    # Number of SGD steps
    k_steps = 100       # Number of steps before printing
    
    bar = trange(config.N_STEPS, leave=True, position=0)
    bar.update(offset)
    
    while n_steps < config.N_STEPS:
        for i, X in enumerate(trainloader):
            X = X.to(config.DEVICE)
            
            shallSave = n_steps in times_save #(n_steps%save_every == 0)
            if n_steps >= config.N_STEPS:
                shallSave = 1
            
            if shallSave == 1:
                # Save model
                p = config.path_save + suffix + 'Models/' + 'Model_{:d}'.format(n_steps)
                torch.save(model.state_dict(), p)
                
                if generate:
                    # Sample a small batch and save it to check quality visually
                    if len(X.shape) == 4: # For images, assumes [B, C, H, W]
                        samples, samples_init = sample_diffusion_from_noise(model, 64, config, df, dim=4)
                        fig = Plot.imshow(samples.cpu(), config.mean, config.std)
                        wandb.log({"generate/samples": wandb.Image(fig)})
                        fig.savefig(config.path_save + suffix + 'Images/' + 'Sample_{:d}.pdf'.format(n_steps), bbox_inches='tight')
                        plt.close('all')
            
            loss, _, = train_one_batch(X, model, optimizer, loss_fn, config, df)
            n_steps += 1            # Update number of steps
            
                        
            # Update the bar (every k steps)
            if n_steps%k_steps == 0:
                bar.set_description(f'loss: {loss:.5f}, n_steps: {n_steps:d}')
                bar.update(k_steps)
            
            # If we performed all the steps, exit
            if n_steps >= config.N_STEPS:
                break
            
    # Return nothing
    return

@torch.no_grad()
def sample_diffusion_from_noise_DDIM(model, n_images=25, config=TrainingConfig(), 
                                df=DiffusionConfig(), dim=3, eta=0.0, ddim_steps=None):
    """
    Generates images using the DDIM sampling procedure with a subsampled schedule.
    
    Parameters:
      model: The noise prediction network.
      n_images: Number of images to generate.
      config: TrainingConfig with attributes such as IMG_SHAPE, TIMESTEPS, DEVICE, etc.
      df: DiffusionConfig with diffusion schedule tensors
      dim: Dimensionality of the tensor (e.g., 2 for [B, N], 3 for [B, C, N], 4 for [B, C, H, W]).
      eta: Hyperparameter controlling stochasticity (eta=0 yields a deterministic process).
      ddim_steps: Number of steps S to use for sampling. If None, use all timesteps.
    
    Returns:
      x: The final generated images.
      x_init: The initial noise samples.
    """
    # Determine the number of steps to use
    total_steps = N_STEPS
    if ddim_steps is None:
        ddim_steps = total_steps

    # Create a schedule of timesteps (linearly spaced and then reversed)
    time_steps = np.linspace(0, total_steps - 1, ddim_steps, dtype=int)[::-1]
    time_steps = list(time_steps)

    # Generate initial noise
    if dim == 4:  # Assumes [B, C, H, W] for 2D images
        x_init = torch.randn(n_images, config.IMG_SHAPE[0], config.IMG_SHAPE[1],
                             config.IMG_SHAPE[2]).to(config.DEVICE)
    elif dim == 3:  # Assumes [B, C, N] for 1D signals with channels
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                             config.IMG_SHAPE[1]).to(config.DEVICE)
    elif dim == 2:  # Assumes [B, N] for 1D signals (no channels)
        x_init = torch.randn(n_images, config.IMG_SHAPE[1]).to(config.DEVICE)
    x = x_init.clone()

    model.eval()
    dt = torch.tensor(1.0 / ddim_steps, device=config.DEVICE, dtype=torch.float32)
    for t in tqdm(reversed(range(1, ddim_steps + 1)), desc='Sampling', leave=True):
        # Time tensor
        ts = torch.ones(n_images, dtype=torch.long, device=config.DEVICE) * t / ddim_steps
        
        # Predict the noise at times ts
        score = model(x, ts)
        
        # Get scaling quantities
        beta_t = df.beta(ts).view(n_images, 1, 1, 1) if dim == 4 else df.beta(ts).view(n_images, 1, 1) if dim == 3 else df.beta(ts).view(n_images, 1)
        
        # Langevin sampling from VPSDE:
        x = x + beta_t * (x + score) * dt / 2
    return x, x_init