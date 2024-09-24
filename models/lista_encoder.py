import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Logistics
        self.im_size = min(args.im_size, args.patch_size) if args.patch_size > 0 else args.im_size
        assert self.im_size > 0
        self.n_channels = args.n_channels
        self.code_dim = args.code_dim
        self.cuda = args.cuda
        self.device = torch.device("cuda" if self.cuda else "cpu")

        print(f"Encoder im size: {self.im_size}")
        print(f"Encoder n channels: {self.n_channels}")
        print(f"Encoder code dim: {self.code_dim}")
        
        # Architecture
        self.T = args.num_iter_LISTA
        self.W = nn.Linear(self.n_channels * self.im_size ** 2, self.code_dim, bias=True)
        self.W.bias.data.fill_(0)
        self.S = nn.Linear(self.code_dim, self.code_dim, bias=False) # no bias in second layer
        self.relu = nn.ReLU()
        
        print(f"Encoder W shape: {self.W.weight.shape}")
        

    def forward(self, y):
        #print(f"Encoder y shape: {y.shape}")
        #print(f"Encoder y shape: {y.view(y.shape[0], -1).shape}")
        B = self.W(y.view(y.shape[0], -1))
        #print(f"Encoder B shape: {B.shape}")
        Z = self.relu(B)
        #print(f"Encoder Z shape: {Z.shape}")
        # LISTA loop
        for t in range(self.T):
            C = B + self.S(Z)
            #print(f"Encoder C shape: {C.shape}")
            Z = self.relu(C)
            #print(f"Encoder Z shape: {Z.shape}")
        Z_final = Z.view(Z.shape[0], -1)
        #print(f"Encoder Z_final shape: {Z_final.shape}")
        #assert()
        return Z_final

    def load_pretrained(self, path, freeze=False):
        # Load pretrained model
        pretrained_model = torch.load(f=path, map_location="cuda" if self.cuda else "cpu")
        msg = self.load_state_dict(pretrained_model)
        print(msg)

        # Freeze pretrained parameters
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
