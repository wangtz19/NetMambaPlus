import torch
import torch.nn.functional as F
import numpy as np
    

class LDAMLoss(torch.nn.Module):
    
    def __init__(self, cls_num_list, device, max_m=0.5, weight=None, s=30,
                 label_smoothing=0):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.tensor(m_list, device=device, dtype=torch.float32)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.to(x.device).float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight, 
                               label_smoothing=self.label_smoothing)
