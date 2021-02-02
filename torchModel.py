import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FeedForward(nn.Module):
    
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)    
    
    def __init__(self, input_size):
        super().__init__()
        
        self.net = nn.Sequential(
            
            nn.Linear(input_size, 20), 
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Dropout(),
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
#             nn.Dropout(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
#             nn.Dropout(),
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),            
            nn.Dropout(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),            
#             nn.Dropout(),            
            nn.Linear(20, 1)
        )
        
        self.net.apply(self.init_weights)
    
    def forward(self, X):
        return self.net(X)
    
    def predict(self, X):
        Y_pred = self.forward(X)
        return Y_pred