import torch
import rotMatcuda
import numpy as np
from cuda.GivensRotations import RotMat, RotMatOpt


class SVDLinear(torch.nn.Module):
    def __init__(self, N, depth, M=None):
        super(SVDLinear, self).__init__()
        
        if M > N:
            raise Exception("M must be <= N")
        assert depth > 0

        self.N = N
        self.M = N if M is None else M
        self.depth = depth
        self.odd_depth = self.depth % 2 != 0

        print("N, M, DEPTH", N, M, depth)
        self.U = RotMat(N,M)

        # Construct the sigma layer
        self.sigma = torch.nn.Parameter(torch.ones(M,1).to(device="cuda"))
        torch.nn.init.normal_(self.sigma, mean=0, std=0.11)

        self.V = RotMat(N,M)
        

    def forward(self, X):
        return self.U.forward(self.get_sigma() *  self.V.forward(X))

    def get_U(self, forward_pass=True):
        return self.U.getU()

    def get_Vt(self, forward_pass=True):
        return self.V.getU()

    def get_sigma(self):
        return torch.pow(torch.abs(self.sigma), self.depth) if self.odd_depth else torch.pow(torch.abs(self.sigma), self.depth-1) * self.sigma

    def detach_SVD(self):
        signed_S = self.get_sigma().detach()
        S = torch.abs(signed_S)

        U = self.get_U().detach()
        # Vt can be a reflection matrix
        V = self.get_Vt().detach().t() if self.depth %2 == 0 else ((signed_S/ S) * self.get_Vt().detach()).t()

        return U, S, V
