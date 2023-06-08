import pyro
import pyro.distributions as dist
from torch import nn



class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, x_dim=50, z_dim=50, hidden_dim=10, p_dim=50, use_cuda=True):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(x_dim, z_dim, hidden_dim).to(device)
        self.decoder = nn.Linear(z_dim, x_dim).to(device)
        self.perturb = nn.Linear(p_dim, z_dim, bias=False).to(device)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        
    def encode_mean(self, x):
        return self.encoder(x)[0]
    
    def encode_var(self, x):
        return self.encoder(x)[1]

    # define the model p(x|z)p(z)
    def model(self, x, p):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        pyro.module("perturb", self.perturb)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            # z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_loc = self.perturb(p)
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder(z)
            # score against actual images
            pyro.sample("obs", dist.Normal(loc_img, tfn(1)).to_event(1), obs=x)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, p):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img