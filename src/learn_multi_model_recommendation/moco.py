import torch
import torch.nn as nn
import torch.optim as optim
from torch.models import resnet50
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10

to_tensor = ToTensor()
CIFAR10_dataset = CIFAR10(root='./data', train=True, transform=to_tensor, download=True)
Dataloader = DataLoader(CIFAR10_dataset, batch_size=64, shuffle=True)

def get_resnet50(output_dim):
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.features,output_dim)
    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

C = 1024
N = Dataloader.batch_size
K = 4096

f_q = get_resnet50(C).to(device)
f_k = get_resnet50(C).to(device)
f_k.load_state_dict(f_q.state_dict())

queue = torch.randn(K,C).to(device)
queue_ptr = 0

m = 0.9999

optimizer =optim.Adam(f_q.parameters(), lr=0.001)

def aug(x):
    return x+0.1*torch.randn_like(x)

def info_nce_loss(q,k,queue,temperature=0.07):
    q =nn.functional.normalize(q,p=2,dim=1) # 64,1024
    k =nn.functional.normalize(k,p=2,dim=1) # 64,1024
    queue = nn.functional.normalize(queue,p=2,dim=0) # 1024,4096
    
    postive_sim = torch.bmm(q.view(N,1,C),k.view(1,N,C)) # 64,64,1024
    negative_sim = torch.mm(q,queue) # 64,4096,1024

    sim = torch.cat([postive_sim.squeeze(-1),negative_sim], dim=1) # 64,41096,1024
    sim = torch.clamp(sim, min=-1, max=1)
    sim = torch.clamp(sim, min=0, max=1)
    return -torch.log(sim
    )


for x, _ in Dataloader:
    x = x.to(device)
    x_q = aug(x)
    x_k = aug(x)
    
    q = f_q(x_q)
    k = f_k(x_k)

    k.detach()
