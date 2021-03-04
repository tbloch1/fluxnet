import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,n_feature, n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, int(n_hidden/2))
        self.hidden3 = torch.nn.Linear(int(n_hidden/2), int(n_hidden/4))
        self.hidden4 = torch.nn.Linear(int(n_hidden/4), int(n_hidden/4))
        self.predict = torch.nn.Linear(int(n_hidden/4), n_output)
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.predict(x) 
        return x.view(-1)

class MultiNet(nn.Module):
    def __init__(self,n_feature, n_hidden,n_output):
        super(MultiNet,self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, int(n_hidden/2))
        self.hidden3 = torch.nn.Linear(int(n_hidden/2), int(n_hidden/4))
        self.hidden4 = torch.nn.Linear(int(n_hidden/4), int(n_hidden/4))
        self.predict = torch.nn.Linear(int(n_hidden/4), n_output)
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.predict(x)
        return x


class Multihead(nn.Module):
    def __init__(self,n_feature, n_hidden,n_output):
        super(Multihead,self).__init__()
        self.model1 = nn.Sequential(
                            nn.Linear(n_feature, n_hidden),
                            nn.ReLU(),
                            nn.Linear(n_hidden, int(n_hidden*2)),
                            nn.ReLU(),
                            nn.Linear(int(n_hidden*2), n_hidden),
                            nn.ReLU(),
                            nn.Linear(n_hidden, int(n_hidden/2)),
                            nn.ReLU()
                            )
                            
        self.model2 = nn.ModuleList([nn.Sequential(
                                     nn.Linear(int(n_hidden/2),
                                               int(n_hidden/4)),
                                               nn.ReLU(),
                                               nn.Linear(int(n_hidden/4),1)
                                               )
                                     for i in range(n_output)])

    
    def forward(self, x):
        x = self.model1(x)
        x = torch.cat([head(x) for head in self.model2],1)
        return x


class MultiheadBNN(nn.Module):
    def __init__(self,n_feature, n_hidden,n_output):
        super(MultiheadBNN,self).__init__()
        self.model1 = nn.Sequential(
                            nn.Linear(n_feature, n_hidden),
                            nn.ReLU(),
                            nn.Linear(n_hidden, int(n_hidden*2)),
                            nn.ReLU(),
                            nn.Linear(int(n_hidden*2), n_hidden),
                            nn.ReLU(),
                            nn.Linear(n_hidden, int(n_hidden/2)),
                            nn.ReLU()
                            )
                            
        self.model2 = nn.ModuleList([nn.Sequential(
                                     nn.Linear(int(n_hidden/2),
                                               int(n_hidden/4)),
                                               nn.ReLU(),
                                               nn.Linear(int(n_hidden/4),2)
                                               )
                                     for i in range(n_output)])

    
    def forward(self, x):
        x1 = self.model1(x)
        x2 = torch.cat([head(x1).view(-1,1,2) for head in self.model2],1)
        return x2