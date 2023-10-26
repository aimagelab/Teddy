import torch
from util import RandReducingScheduler, LinearScheduler
from model.ocr import OrigamiNet
import matplotlib.pyplot as plt
from tqdm import tqdm


class FakeNet(torch.nn.Module):
    def __init__(self):
        super(FakeNet, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x * self.a


device = 'cuda:0'

net = FakeNet().to(device)
net.eval()

ocr_scheduler = LinearScheduler(net, net.state_dict(), net.state_dict(), period=10000)
# ocr_scheduler = RandReducingScheduler(net, net.state_dict(), net.state_dict())

alphas = []
indixes = []
for i in tqdm(range(100000)):
    ocr_scheduler.step()
    alphas.append(ocr_scheduler.last_alpha)
    indixes.append(i)

plt.plot(indixes, alphas)
plt.savefig(f'files/{ocr_scheduler.__class__.__name__}.png')
