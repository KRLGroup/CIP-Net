import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

class wrapper:
    def __init__(self, model, devices=['cpu'], parallelization="DP", rank = -1):
        super().__init__()
        self.parallelization = parallelization
        self.devices = devices
        self.rank = rank
        if self.parallelization=="DP":
            self.model = nn.DataParallel(model, device_ids=self.devices)
            self.module = self.model.module
        elif self.parallelization=="DDP":
            # print("WRAPPER DDP")
            self.devices = self.devices[1:] #check
            self.model = DDP(model, device_ids=[self.rank], find_unused_parameters=True)
            self.module = self.model.module
        else:
            self.module = model
            
    
    def __call__(self, *inputs, **kwargs):
        if 'cpu' in self.devices:
            return self.module.__call__(*inputs, **kwargs)
        else:
            return self.model.__call__(*inputs, **kwargs)
    
    def to(self, device):
        if 'cpu' in self.devices:
            self.module = self.module.to(device)
        else:
            self.model = self.model.to(device)
        return self
    
    def train(self):
        if 'cpu' in self.devices:
            return self.module.train()
        else:
            return self.model.train()

    def eval(self):
        if 'cpu' in self.devices:
            return self.module.eval()
        else:
            return self.model.eval()
    
    def named_parameters(self):
        if 'cpu' in self.devices:
            return self.module.named_parameters()
        else:
            return self.model.named_parameters()
    
    def parameters(self):
        if 'cpu' in self.devices:
            return self.module.parameters()
        else:
            return self.model.parameters()

