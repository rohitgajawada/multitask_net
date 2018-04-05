import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.opt = opt

        alex = models.alexnet(pretrained=False)
        self.features = alex.features
        self.classifier = nn.Sequential(
                nn.Linear(18432, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 42),
                )

    def forward(self, x, xprev):

        x1 = self.features(x)
        x1 = x1.view(-1, 9216)
        x2 = self.features(xprev)
        x2 = x2.view(-1, 9216)
        x = torch.cat((x1, x2), 1)
        x = self.classifier(x)
        return x

x = Variable(torch.randn(1, 3, 224, 224))
xprev = Variable(torch.randn(1, 3, 224, 224))
net = Net()
out = net(x, xprev)
print(out.size())
