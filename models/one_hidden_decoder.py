import torch
import torch.nn as nn

def L0(z, reduction='mean'):
    """
    :param z: (B, C) or (B, C, W, H) tensor
    :return: average of proportion of zero elements in each element in batch
    """
    with torch.no_grad():
        assert (len(z.shape) == 2 or len(z.shape) == 4)
        dims = 1 if len(z.shape) == 2 else (1, 2, 3)
        prop_0s_each_sample = (z.abs() == 0).float().mean(dims)
        if reduction == 'sum':
            return prop_0s_each_sample.sum()
        if reduction == 'mean':
            return prop_0s_each_sample.mean()

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Model
        self.im_size = min(args.im_size, args.patch_size) if args.patch_size > 0 else args.im_size
        self.n_channels = args.n_channels
        self.code_dim = args.code_dim
        self.output_dim = (self.im_size ** 2) * self.n_channels
        self.hidden_dim = args.hidden_dim
        self.layer1 = nn.Linear(self.code_dim, self.hidden_dim, bias=True)
        self.layer1.bias.data.fill_(0) # Fill bias with 0s
        self.layer2 = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        self.relu = nn.ReLU()
        self.Zs_init_val = args.Zs_init_val
        self.cuda = args.cuda
        self.device = torch.device("cuda" if self.cuda else "cpu")

    def forward(self, code):
        output = self.layer1(code)
        with torch.no_grad():
            self.frac_0s_hidden_pre_relu = L0(output, 'mean')
        output = self.relu(output)
        with torch.no_grad():
            self.frac_0s_hidden_post_relu = L0(output, 'mean')
        output = self.layer2(output)
        return output.view(output.shape[0], self.n_channels, self.im_size, -1)

    def initZs(self, batch_size):
        Zs = torch.zeros(size=(batch_size, self.code_dim), device=self.device).fill_(self.Zs_init_val)
        return Zs
    
    def get_n_atoms(self):
        return self.layer2.weight.data.shape[1]

    def viz_columns(self, n_samples=24, norm_each=False):
        # Visualize columns of the linear layer closest to reconstruction
        cols = []
        W = self.layer2.weight.data
        max_abs = W.abs().max()
        # Iterate over columns
        for c in range(n_samples):
            column = W[:, c].clone().detach()
            if norm_each:
                max_abs = column.abs().max()
            # Map values to (-0.5, 0.5) interval
            if max_abs > 0:
                column /= (2 * max_abs)
            # Map 0 to gray (0.5)
            column += 0.5
            # Reshape column to output shape
            column = column.view(self.n_channels, self.im_size, -1)
            cols.append(column)
        cols = torch.stack(cols)
        return cols
    
    def viz_columns2(self, n_samples=24, norm_each=False):
        # Visualize columns of both linear layers 
        Ws = [self.layer2.weight.data, self.layer1.weight.data]
        cols1=[]
        cols2=[]
        for i,W in enumerate(Ws):
            #print(f"Decoder Weight shape: {W.shape}")
            #print(f'Number of atoms: {W.shape[1]}')
            #print(f"Size of each atom {self.n_channels} x {self.im_size} x {self.im_size} = {W.shape[0]}")
            n_samples = W.shape[1]
            cols = []
            max_abs = W.abs().max()
            # Iterate over columns
            for c in range(n_samples):
                column = W[:, c].clone().detach()
                if norm_each:
                    max_abs = column.abs().max()
                # Map values to (-0.5, 0.5) interval
                if max_abs > 0:
                    column /= (2 * max_abs)
                # Map 0 to gray (0.5)
                column += 0.5
                # Reshape column to output shape
                if i==1:
                    column = column.view(1, self.im_size, -1)
                else:    
                    column = column.view(self.n_channels, self.im_size, -1)
                cols.append(column)
            # End of for over n_samples
            cols = torch.stack(cols)
            if i==1:
                cols1 = cols
            else:
                cols2 = cols
        # End of for over Ws
        print(f"\ncols1 shape: {cols1.shape} --- cols2 shape: {cols2.shape}")
        print(f"cols1 max: {cols1.max()} --- cols2 max: {cols2.max()}")
        print(f"cols1 min: {cols1.min()} --- cols2 min: {cols2.min()}\n")
        
        return cols1,cols2

    def viz_codes(self, fill_vals, n_samples=24):
        # Visualize reconstructions from singe active code componnet
        codes = torch.zeros(n_samples, self.code_dim).to(self.device)
        # Visualize reconstuctions from bias (if there is one)
        with torch.no_grad():
            recs_bias = self.forward(codes)
        # Generate codes with a single active component
        for c in range(n_samples):
            codes[c, c] = fill_vals[c]
        # Reconstructions from codes with a single active component
        with torch.no_grad():
            recs = self.forward(codes)
        return recs - recs_bias

    def load_pretrained(self, path, freeze=False):
        # Load pretrained model
        pretrained_model = torch.load(f=path, map_location="cuda" if self.cuda else "cpu")
        msg = self.load_state_dict(pretrained_model)
        print(msg)

        # Freeze pretrained parameters
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
