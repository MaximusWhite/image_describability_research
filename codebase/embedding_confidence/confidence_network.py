import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class SemaConfNet(nn.Module):
        
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(2048, 50, bias=True)
#         self.dense2 = nn.Linear(4096, 4096, bias=True)
#         self.batch_norm1 = nn.BatchNorm1d(50)
#         self.dense3 = nn.Linear(4096, 1, bias=True)
#         self.drop1 = nn.Dropout(p=0.3)
#         self.dense4 = nn.Linear(4096, 4096, bias=True)
#         self.batch_norm2 = nn.BatchNorm1d(4096)
#         self.dense5 = nn.Linear(4096, 2048, bias=True)
#         self.dense6 = nn.Linear(2048, 2048, bias=True)
#         self.batch_norm3 = nn.BatchNorm1d(2048)
#         self.dense7 = nn.Linear(2048, 1024, bias=True)
#         self.dense8 = nn.Linear(1024, 1024, bias=True)
#         self.batch_norm4 = nn.BatchNorm1d(1024)
        self.dense9 = nn.Linear(50, 1, bias=True)
        self.batch_norm5 = nn.BatchNorm1d(1)
        self.activation = nn.Sigmoid()

    def forward(self, x): 
        x = self.dense1(x)
#         x = self.dense2(x)
#         x = self.batch_norm1(x)
#         x = self.dense3(x)
#         x = self.drop1(x)
#         x = self.dense4(x)
#         x = self.batch_norm2(x)
#         x = self.dense5(x)
#         x = self.dense6(x)
#         x = self.batch_norm3(x)
#         x = self.dense7(x)
#         x = self.dense8(x)
#         x = self.batch_norm4(x)
        x = self.dense9(x)
        x = self.batch_norm5(x)
#         print('Before sigmoid: {}'.format(x.cpu().detach().numpy()))
        output = self.activation(x)
        return output

    
class SemaConfNetSplit(nn.Module):
    '''
                |           input           |
                          /       \
                         v         v
                |  img embed | | cap embed  |
                       |              |
                       v              v
                |  Linear    | |  Linear    |
                        \           /
                          \       /
                            \   /
                              v
                |           Linear          |
                              |
                              v
                            .....
    '''
    def __init__(self):
        super().__init__()
        self.dense1_1 = nn.Linear(2048, 2048, bias=True)
        self.dense1_2 = nn.Linear(768, 768, bias=True)
        self.batch_norm1_1 = nn.BatchNorm1d(2048)
        self.batch_norm1_2 = nn.BatchNorm1d(768)
        self.relu1_1 = nn.ReLU()
        self.relu1_2 = nn.ReLU()
        
        self.dense2 = nn.Linear(2816, 2048, bias=True)         # 2816 for ResNet
        self.relu2 = nn.ReLU()
#         self.batch_norm1 = nn.BatchNorm1d(4096)
        self.dense3 = nn.Linear(2048, 4096, bias=True)
        self.relu3 = nn.ReLU()
#         self.drop1 = nn.Dropout(p=0.3)
        self.dense4 = nn.Linear(4096, 7000, bias=True)
        self.relu4 = nn.ReLU()
#         self.batch_norm2 = nn.BatchNorm1d(4096)
#         self.dense5 = nn.Linear(7000, 7000, bias=True)
#         self.dense6 = nn.Linear(7000, 7000, bias=True)
#         self.batch_norm3 = nn.BatchNorm1d(7000)
        self.dense7 = nn.Linear(7000, 2048, bias=True)
        self.relu5 = nn.ReLU()
        self.dense8 = nn.Linear(2048, 2048, bias=True)
        self.batch_norm4 = nn.BatchNorm1d(2048)
        self.relu6 = nn.ReLU()
        self.dense9 = nn.Linear(2048, 1, bias=True)
        self.batch_norm5 = nn.BatchNorm1d(1)
        self.activation = nn.Sigmoid()

    def forward(self, x): 
        img_portion = torch.narrow(x, 1, 0, 2048)        # 2048 for ResNet
        cap_portion = torch.narrow(x, 1, 2048, 768)
        x1 = self.dense1_1(img_portion)
        x2 = self.dense1_2(cap_portion)
        
        x1 = self.batch_norm1_1(x1)
        x2 = self.batch_norm1_2(x2)
        
        x1 = self.relu1_1(x1)
        x2 = self.relu1_2(x2)
        
#         x = self.batch_norm1(x)
        x = torch.cat((x1, x2), 1)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        x = self.relu3(x)
#         x = self.drop1(x)
        x = self.dense4(x)
        x = self.relu4(x)
#         x = self.batch_norm2(x)
#         x = self.dense5(x)
#         x = self.dense6(x)
#         x = self.batch_norm3(x)
        x = self.dense7(x)
        x = self.relu5(x)
        x = self.dense8(x)
        x = self.relu6(x)
#         x = self.batch_norm4(x)
        x = self.dense9(x)
        x = self.batch_norm5(x)
#         print('Before sigmoid: {}'.format(x.cpu().detach().numpy()))
        output = self.activation(x)
        return output    
    