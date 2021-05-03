import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, in_states, device, dtype):
        super().__init__()
        self.in_states = in_states
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(self.in_states,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,1)

    def forward(self, x):
        x = self.fc1(x)
        x = x*T.sigmoid(x)
        x = self.fc2(x)
        x = x*T.sigmoid(x)
        x = F.relu(self.fc3(x))

        return x

    def get_max_min(self,x,y):
        self.x_max = np.max(np.max(x,axis=0),axis=1).reshape(-1,1)
        self.x_min = np.min(np.min(x,axis=0),axis=1).reshape(-1,1)
        self.y_max = np.max(y,axis=1).reshape(-1,1)
        self.y_min = np.min(y,axis=1).reshape(-1,1)

    def standardize_input(self, x):
        z = (x - self.x_min)/(self.x_max - self.x_min)
        return z

    def standardize_output(self, y):
        z = (y - self.y_min)/(self.y_max - self.y_min)
        return z


class Agent():
    def __init__(self, in_states, device='cpu', dtype=T.float):
        self.net = Net(in_states, device, dtype).to(device)
        self.dtype = dtype
        self.device = device
        self.in_states = in_states

    def update_weights(self, alpha, R, G, P, x, u, x_l):
        x = T.tensor(x, device=self.device, dtype=self.dtype)
        x_l = T.tensor(x_l, device=self.device, dtype=self.dtype)
        u = T.tensor(u, device=self.device, dtype=self.dtype).view(-1)
        data = T.cat((x,x_l,u),0) # Concatenate along axis 0

        self.net.zero_grad()
        out = self.net(data)
        out.backward()
        coefficient = alpha*(R + G - P)
        for parameter in self.net.parameters():
            parameter.data.sub_(coefficient*parameter.grad.data)

    def eval(self, x, x_l, u):
        x = T.tensor(x, device=self.device, dtype=self.dtype)
        x_l = T.tensor(x_l, device=self.device, dtype=self.dtype)
        u = T.tensor(u, device=self.device, dtype=self.dtype).view(-1)
        data = T.cat((x,x_l,u),0) # Concatenate along axis 0

        return self.net(data).cpu().detach().numpy()

    def action(self,x,x_l,u):
        x = T.tensor(x, device=self.device, dtype=self.dtype)
        x_l = T.tensor(x_l, device=self.device, dtype=self.dtype)
        u = T.tensor(u, device=self.device, dtype=self.dtype).view(-1)
        data = T.cat((x,x_l,u),0) # Concatenate along axis 0
        data.requires_grad_(True)
        out = self.net(data)

        cont = control()
        optimizer = optim.Adam(cont.parameters(),lr=0.001)

        max_iter = 3000
        iter = 0
        tol = 1e-6
        while iter < max_iter:
            iter += 1
            optimizer.zero_grad()
            loss = self.net(T.cat((x,x_l,cont()),0))
            if np.abs(loss.detach().cpu().numpy()) < tol:
                break
            print(loss)
            loss.backward()
            optimizer.step()

        return cont().detach().cpu().numpy()


class control(nn.Module):
    def __init__(self,device='cpu',dtype=T.float):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1,1)

    def forward(self):
        u = self.fc1(T.ones(1,device=self.device,dtype=self.dtype,
                                                    requires_grad=True))
        return u
