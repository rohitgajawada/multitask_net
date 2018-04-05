import torch
import torch.nn as nn
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.opt = opt

        self.smax = nn.Softmax()
        self.fc_0_m1 = nn.Linear(169, 25)
        self.fc_0_1 = nn.Linear(169, 25)
        self.fc_m1_0 = nn.Linear(169, 25)
        self.fc_1_0 = nn.Linear(169, 25)
        self.fc_0_0 = nn.Linear(169, 25)

    def forward(self, xi, xh, xp):
        outxi = self.salpath(xi)
        outxh = self.gazepath(xh, xp)
        output = outxi * outxh
        if self.opt.shiftedflag == False:
            output = output.view(-1, 169)
            return output
