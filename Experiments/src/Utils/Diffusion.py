import numpy as np
import torch
from tqdm import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
import Plot
import torch.nn.utils as nn_utils

import math

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

def cholesky_unroll(A):
    """
    Unrolled Cholesky decomposition for arbitrary-size SPD matrices.
    A: (B, n, n) batch of SPD matrices
    Returns: (B, n, n) lower-triangular matrices L such that A ≈ LLᵀ
    """
    B, n, _ = A.shape
    L = torch.zeros_like(A, device=A.device)

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                sum_k = torch.sum(L[:, i, :j] ** 2, dim=1)
                L[:, i, j] = torch.sqrt(A[:, i, i] - sum_k)
            else:
                sum_k = torch.sum(L[:, i, :j] * L[:, j, :j], dim=1)
                L[:, i, j] = (A[:, i, j] - sum_k) / L[:, j, j]
    return L

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
NUM_EPS = 1e-6
n = None
L_inv = 1.0
alpha = 0.08
lambdastar = None
Bs = None
device = 'cuda:0'
# Set to 1 because this is grayscale:
d = 1
Sigma_0_diag = None
data_shape = (d, 32, 32,)
F_matrix = None
xi = None
hold_T = 5.0

def HOLD_init(n_):
    global n, lambdastar, Sigma_0_diag, xi, F_matrix, Bs
    n = n_
    lambdastar = -math.sqrt(2*n - 3)

    # nOLD specific parameters:
    # Initialize diagonal of Sigma_0
    Sigma_0_diag = torch.full((n,), alpha*L_inv, device=device).double()
    Sigma_0_diag[0] = 0
    Sigma_0_diag = Sigma_0_diag-L_inv

    # Compute gammas and xi
    gammas = torch.tensor([(n*n - i*i) / (4*i*i - 1) for i in range(1, n)], device=device)
    gammas = -lambdastar * torch.flip(gammas, dims=[0]).sqrt()
    xi = -n * lambdastar
    
    # Construct F
    F_matrix = torch.diag(gammas, 1) - torch.diag(gammas, -1)
    F_matrix[-1, -1] = -xi

    # precompute taylor expansion coefficient matrices of expFt
    F_shifted = F_matrix - lambdastar*torch.eye(n, device=device)
    F_shifted = F_shifted.double()
    F_k = torch.eye(n, device=device).double()
    Bs = [F_k]
    for k in range(1, n):
        F_k = F_k @ F_shifted / k
        Bs.append(F_k)

    Bs = torch.stack(Bs, dim=-1)

class DiffusionConfig:
    '''
    ClassDiffusion: Class containing information related to the 
    diffusion process (number of steps, device, variance, etc.)
    '''
    def __init__(self, n_steps=N_STEPS, img_shape=(3, 32, 32), device='cuda:0', n=2):
        self.n_steps = n_steps
        self.img_shape = img_shape
        HOLD_init(n)
        
        #F_matrix = torch.kron(F_matrix, torch.eye(3, device=device)).double()

    
# ====================================================================
# Diffusion functions
# ==================================================================== 
def forward_diffusion(df, x0, expFt, config):
    dim = len(x0.shape)
    # Generate noise realisation with the same size as a batch of images (float32 for efficiency)
    eps = torch.randn_like(x0).float()

    # Apply the forward diffusion kernel at times timesteps
    d = x0.shape[1] // n
    expFt_ = torch.kron(expFt, torch.eye(d, device=device)).double()
    mean = torch.einsum("bik,bk...->bi...", expFt_, x0.double())

    Sigma_t = torch.einsum("bik,k,bjk->bij", expFt, Sigma_0_diag, expFt)
    Sigma_t.diagonal(dim1=-2, dim2=-1).add_(L_inv + NUM_EPS)
    Sigma_t = torch.kron(Sigma_t, torch.eye(d, device=device)).double()
    L_t = cholesky_unroll(Sigma_t)
    L_t = torch.nan_to_num(L_t, nan=0.0).float()

    noise = torch.einsum("bij,bj...->bi...", L_t, eps)
    sample_a = mean + noise
    
    # Return the noisy image and the noise realisation
    return sample_a, eps, L_t


@torch.no_grad()
def sample_diffusion_from_noise(model, n_images=25, config=TrainingConfig(), 
                                df=DiffusionConfig(), dim=3):
    if n is None:
        HOLD_init(config.model_order)

    # Generate n_images starting points from N(0, 1)
    if dim == 4: # Assumes [B, C, H, W] for 2d
        shape = (n_images, n, config.IMG_SHAPE[0], config.IMG_SHAPE[1], config.IMG_SHAPE[2])
        x_init = torch.randn(shape).to(config.DEVICE)
    elif dim == 3: # Assumes [B, C, N] for 1d
        shape = (n_images, n, config.IMG_SHAPE[0], config.IMG_SHAPE[1])
        x_init = torch.randn(shape).to(config.DEVICE)
    elif dim == 2: # Assumes [B, N] for 1d (no channels)
        shape = (n_images, n * config.IMG_SHAPE[0], config.IMG_SHAPE[1])
        x_init = torch.randn(shape).to(config.DEVICE)

    x = math.sqrt(L_inv) * x_init.clone()
    
    model.eval()
    dt = torch.tensor(hold_T / N_STEPS, device=config.DEVICE, dtype=torch.float32)
    d_coef = 2 * xi * L_inv * dt
    for t in reversed(range(1, N_STEPS + 1)):
        # Time tensor
        ts = torch.ones(n_images, dtype=torch.long, device=config.DEVICE) * hold_T * t / N_STEPS

        # Generate one realisation of the noise
        z = torch.randn((n_images, *data_shape), device=device) if t > 0 else torch.zeros((n_images, *data_shape), device=device)
        
        # Predict the noise at times ts
        score = model(x.view(-1, n, 32, 32).float(), ts)
        #print(f'F_matrix shape: {F_matrix.shape}, score shape: {score.shape}', flush=True)
        x -= torch.einsum("ij, bj...->bi...", F_matrix, x) * dt
        drift_diff = d_coef*score + torch.sqrt(d_coef) * z
        x[:, -1, ...] += drift_diff

    x_init = x_init[:, 0].view(-1, *config.IMG_SHAPE)
    x = x[:, 0].view(-1, *config.IMG_SHAPE)

    return x, x_init

 
#==========================================
# Training functions
#==========================================
def train_one_batch(Q, model, optimizer, loss_fn, 
                    config=TrainingConfig(), 
                    df=DiffusionConfig()):
    model.train()
    batch_size = Q.shape[0]
    shape = (batch_size, n-1, *data_shape)
    noise = alpha * L_inv * torch.randn(shape, device=device)
    X = torch.cat((Q[:, None, ...], noise), dim=1)

    # Generate random times (continuous uniform in [0,1))
    t = hold_T * torch.rand((X.shape[0], 1), device=config.DEVICE)
    
    powers = t ** torch.arange(n, device=device).double()
    expFt = torch.einsum(
        "bk,ijk->bij",
        (torch.exp(lambdastar * t) * powers),
        Bs.double()  # make sure Bs is also double
    )

    # Extract noisy images from times t
    X_t, noise_t, L_t = forward_diffusion(df, X, expFt, config)
    X_t = X_t.to(device)
    
    # Apply the model
    score = model(X_t.view(-1, n, 32, 32).float(), t.view(-1))
    est = torch.einsum("b...,b->b...",
                    score,
                    -L_t[:, -1, -1])
    noise_t = noise_t[:, -1]
    noise_t = noise_t.view(-1, 1, 32, 32)
    
    # The loss is comparing the predicted and true noises
    loss = loss_fn(noise_t, est)
    
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
