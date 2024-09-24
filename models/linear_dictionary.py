import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Model
        self.patch_size = args.patch_size 
        self.n_channels = args.n_channels
        self.code_dim = args.code_dim
        self.im_size = min(args.im_size, args.patch_size) if args.patch_size > 0 else args.im_size
        assert self.im_size > 0
        self.output_dim = (self.im_size ** 2) * self.n_channels
        self.decoder = nn.Linear(self.code_dim, self.output_dim, bias=False)
        self.cuda = args.cuda
        self.device = torch.device("cuda" if self.cuda else "cpu")

        print(f"Decoder: {self.decoder}")
        print(f"Decoder patch size: {self.patch_size}")
        print(f"Decoder output dim: {self.output_dim}")
        print(f"Decoder im size: {self.im_size}")
        print(f"Decoder n channels: {self.n_channels}")
        print(f"Decoder code dim: {self.code_dim}")
        print(f"Decoder cuda: {self.cuda}")
        print(f"Decoder device: {self.device}")
        
        
    def forward(self, code):
        output = self.decoder(code)
        #print(f"Decoder output shape: {output.shape}")
        output = output.view(output.shape[0], self.n_channels, self.im_size, -1)
        #print(f"Decoder output shape: {output.shape}")
        #assert()
        return output

    def initZs(self, batch_size):
        Zs = torch.zeros(size=(batch_size, self.code_dim), device=self.device)
        return Zs

    def viz_columns(self, n_samples=24, norm_each=False):
        # Visualize columns of linear decoder
        cols = []
        W = self.decoder.weight.data # Weights of decoder
        #print(f"Decoder Weight shape: {W.shape}")
        #print(f'Number of atoms: {W.shape[1]}')
        #print(f"Size of each atom {self.n_channels} x {self.im_size} x {self.im_size} = {W.shape[0]}")
        assert self.n_channels*self.im_size*self.im_size== W.shape[0]
        max_abs = W.abs().max() # Max absolute value of weights
        # Iterate over columns
        for c in range(n_samples): 
            column = W[:, c].detach().clone() # Get column
            if norm_each:
                max_abs = column.abs().max()
            # Map values to (-0.5, 0.5) interval
            if max_abs > 0:
                column /= (2 * max_abs)
            # Map 0 to gray (0.5)
            column += 0.5
            # Reshape column to output shape
            #print(f"Decoder column shape: {column.shape}")
            column = column.view(self.n_channels, self.im_size, -1)
            #print(f"Decoder column shape: {column.shape}")
            cols.append(column)
        cols = torch.stack(cols)
        #print(f"\ncols1 shape: {cols.shape}")
        #print(f"cols1 max: {cols.max()}")
        #print(f"cols1 min: {cols.min()}\n")
        
        return cols

    def viz_columns2(self, n_samples=24, norm_each=False):
        return self.viz_columns(n_samples=self.decoder.weight.data.shape[1], norm_each=norm_each), 0 

    def get_n_atoms(self):
        return self.decoder.weight.data.shape[1]
     
    def load_pretrained(self, path, freeze=True):
        # Load pretrained model
        pretrained_model = torch.load(f=path, map_location="cuda" if self.cuda else "cpu")
        msg = self.load_state_dict(pretrained_model)
        print(msg)

        # Freeze pretrained parameters
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
