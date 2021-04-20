import torch
import numpy as np
from torch import nn


class DNN():
    
    def __init__(self, d_in, d_out, hidden_size=[]):
        
        self.hs = [d_in] + hidden_size + [d_out]
        self.N = sum([(inp+1)*out for inp, out in zip(self.hs[:-1], self.hs[1:])])
        
    def __call__(self, x, Wb):
        # Computes forward pass over the DNN with weights Wb
        params = self.unflatten_params(Wb)
        z = x
        for W, b in params[:-1]:
            z = torch.relu(z @ W + b)

        W, b = params[-1]

        z = z @ W + b
        return z
    
    def gen_params(self):
        # Initializes a vector of weights of compatible size with the DNN
        Wb = torch.rand(self.N)

        cum = 0
        for inp, out in zip(self.hs[:-1], self.hs[1:]):
            k = np.sqrt(1/inp)
            Wb[cum:cum+(inp*out)+out] = 2*k*Wb[cum:cum+(inp*out)+out] - k
            cum += (inp+1)*out

        return nn.Parameter(Wb)
        
    def unflatten_params(self, Wb):
        # Returns a list of tuples (W_h, b_h) for each layer h in the BNN
        params = []
        cum = 0
        for inp, out in zip(self.hs[:-1], self.hs[1:]):
            W = Wb[cum:cum+(inp*out)].view(inp, out)
            b = Wb[cum+(inp*out):cum+(inp*out)+out]
            params.append((W, b)) 

            cum += (inp+1)*out

        return params
