import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from network import Net
from network import Agent
import torch as T

'''
July 2020
by Jing Wang (RLconsensus_NL_Con.m)
Reproduced by Elias Wilson

Consensus Based Reinforcement Learning Algorithm

Current System: Cart Pole
'''

##### Inputs
# Time
Time = 5
dt = 0.01

# Cart-Pole Parameters
L, m, M, g = 1, 0.5, 2, 9.81

# Initial Learning Rate
alpha0 = 1
kappa = 0.05

# Turn on Data Export to .mat
save_data = False

# Intitial State (axis: 0-Agents  1-states )
# These are now filled inside of the main loop
x0_sigma = [0, 0, 0.1, 0.1] # Variance of random initial conditions
x_l0_sigma = [0, 0, 0, 0] # Leader
theta_ref = np.pi # Pole Angle (0 is up but at pi the system is locally stable)


adj_l = [1, 1, 1] # Agents connection to the leader
A = np.array([[0, 1, 1],[1, 0, 1],[1, 1, 0]]) # Adjacency Matrix
A_ = np.array([[0, 1, 1],[1, 0, 1],[1, 1, 0]]) # Used for comparison

# Weighting Matricies
Q = [0, 0, 1, 1]*np.identity(4) # Size of single agent state space
Q_l = [0, 0, 1, 1]*np.identity(4) # Leader weighting
U = 0.01*np.identity(3) # Size of total system action spaces

# Neural Net based Agents
device = 'gpu'
dtype = T.float
n_inputs = 9 # states + leader states + control inputs (4 + 4 + 1)
EPOCHS = 1000
# This dictionary stores the agent object for each agent. This object contains
# neural network and realted functions
Agents = dict({0: Agent(n_inputs),
            1: Agent(n_inputs),
            2: Agent(n_inputs)})

##### Functions
# Create dictionary with agent keys and a vector of indices for what other
# agents it can see
def find_adjacencies(N,A):
    adj = dict()
    for i in range(N):
        x = np.argwhere(A[i,:] > 0).reshape(-1)
        adj.update({i: x})
    return adj

# Euler Integration
def Eul(f,x,u):
    return x + dt*f(x,u)

# Runge-Kutta 4
def rk4(f,x,u):
    f1 = dt*f(x,u)
    f2 = dt*f(x+0.5*f1,u)
    f3 = dt*f(x+0.5*f2,u)
    f4 = dt*f(x+f3,u)
    return x + 1/6*(f1 + 2*f2 + 2*f3 + f4)

# System Dynamics (Cart Pole)
def f(x,u):
    xddot = (u[0] - m*np.sin(x[2])*(L*x[3]*x[3] + \
     g*np.cos(x[2])))/(M + m*(1 + np.cos(x[2])*np.cos(x[2])))
    thetaddot = (np.cos(x[2])*xddot + g*np.sin(x[2]))/L
    return np.array([x[1],xddot,x[3],thetaddot])

# Cost Function
def reward(x,u,x_l,adj_l,adj,N):
    cost = np.zeros(N)
    for i in adj.keys():
        ids = adj[i]
        for j in ids:
            cost[i] += (x[i,:]-x[j,:]).dot(Q).dot(x[i,:]-x[j,:])
        cost += adj_l[i]*(x[i,:]-x_l).dot(Q_l).dot(x[i,:]-x_l)
    return cost + (u[:]*u[:]).T.dot(U)

# Update Estimates
def estimate(R,G,P,r,_phi,phi,phi_,ii,N,adj,w,d):
    for i in range(N):
        ids = adj[i]
        R_sum, G_sum, P_sum = 0, 0, 0
        for j in ids:
            R_sum += A[i,j]*(R[ii-1,j] - R[ii-1,i])
            G_sum += A[i,j]*(G[ii-1,j] - G[ii-1,i])
            P_sum += A[i,j]*(P[ii-1,j] - P[ii-1,i])

        R[ii,i] = R[ii-1,i] + 1/(1+d[i])*R_sum + 1/w[i]*(r[ii,i] - r[ii-1,i])
        G[ii,i] = G[ii-1,i] + 1/(1+d[i])*G_sum + 1/w[i]*(phi_[i] - phi[i])
        P[ii,i] = P[ii-1,i] + 1/(1+d[i])*P_sum + 1/w[i]*(phi[i] - _phi[i])

    return R, G, P

# Evaluate Network
def Phi(x,u,adj,x_l,adj_l,N):
    phi = np.zeros(N)
    for i in adj.keys():
        ids = adj[i]
        sum1 = np.zeros(len(x[0]))
        for j in ids:
            sum1 += (x[j] - x[i])**2
        phi[i] = Agents[i].eval(sum1,adj_l[i]*(x_l - x[i]),u[i])
    return phi

# Update Neural Net Weights
def update_weights(alpha, R, P, G, x, u, adj, x_l, adj_l, N):
    for i in adj.keys():
        ids = adj[i]
        sum1 = np.zeros(len(x[0]))
        for j in ids:
            sum1 += (x[j] - x[i])**2
        Agents[i].update_weights(alpha, R[i], P[i], G[i], sum1,
                                                u[i], adj_l[i]*(x_l - x[i]))

# Calculate Actions
def action(x,u,adj,x_l,adj_l,N):
    a = np.zeros_like(u)
    for i in adj.keys():
        ids = adj[i]
        sum1 = np.zeros(len(x[0]))
        for j in ids:
            sum1 += (x[j] - x[i])**2
        a[i,0] = Agents[i].action(sum1,adj_l[i]*(x_l - x[i]),u[i])
    return a

##### Main
N = len(A[:,0]) # Number of agents
adj = find_adjacencies(N,A) # find the adjacenies (who can agent i see)
adj_ = find_adjacencies(N,A_)
t = np.arange(0, Time, dt) # time vector

for epoch in range(EPOCHS):

    # Allocate memory and fill ICs
    nb_states = len(x0_sigma)
    x = np.zeros((len(t),N,nb_states))
    x[0,:,:] = x0_sigma*np.random.randn(N,nb_states)
    x[0,:,2] += theta_ref
    x_l = np.zeros((len(t),len(x_l0_sigma)))
    x_l[0,:] = x_l0_sigma*np.random.randn(nb_states)
    x_l[0,2] += theta_ref
    u = np.zeros((len(t),N,1))
    r = np.zeros((len(t)-1,N))
    R = np.zeros((len(t)-1,N))
    G = np.zeros(np.shape(R))
    P = np.zeros(np.shape(R))

    # Left eigenvector of F
    d = np.sum(A, axis=1) # degree vector
    D = np.diag(d) # degree matrix
    La = D - A # Laplacian
    F = np.identity(N) - np.linalg.inv(np.identity(N) + D).dot(La)
    W,V = np.linalg.eig(F.T)
    w_id = np.argmin(np.power((W-1),2)) # find the index of eigenvalue 1
    v = V[:,w_id] # left eigenvector associated with eigenvalue 1
    if v[0] < 0: # Keep the eigenvector positive
        v = -v
    v = v/np.sum(v)

    for i in range(len(t)-1):

        ## Leader
        x_l[i+1] = x_l[i] # Steady system

        ## RL Consensus
        alpha = alpha0*np.exp(-kappa*i)

        u[i] = action(x[i],u[i],adj,x_l[i],adj_l,N)

        phi = Phi(x[i],u[i],adj,x_l[i],adj_l,N)

        for j in range(N):
            x[i+1,j] = Eul(f,x[i,j],u[i,j])

        u[i+1] = action(x[i+1],u[i],adj,x_l[i+1],adj_l,N)

        r[i,:] = reward(x[i],u[i],x_l[i],adj_l,adj,N)

        if i == 0:
            R[0,:] = r[0,:]/v
            G[0,:] = Phi(x[i+1],u[i+1],adj,x_l[i+1],adj_l,N)/v
            P[0,:] = phi/v

        if i > 0:
            _phi = Phi(x[i-1],u[i-1],adj,x_l[i-1],adj_l,N)
            phi_ = Phi(x[i+1],u[i+1],adj,x_l[i+1],adj_l,N)
            R, G, P = estimate(R,G,P,r,_phi,phi,phi_,i,N,adj,v,d)

        update_weights(alpha,R[i],G[i],P[i],x[i],u[i],adj,x_l[i],adj_l,N)

rf = np.sum(np.flip(r,axis=0),axis=1)
J = np.zeros(len(t)-1)
J[0] = rf[0]
for i in range(len(t)-2):
    J[i+1] = dt*rf[i+1] + J[i]
J = np.flip(J,axis=0)

### Export Data
if save_data:
    results = {'x': x, 'u': u, 'R': R, 'G': G, 'P': P, 'theta_phi': theta_phi, \
                'r': r, 't': t, 'theta': theta, 'r_': r_, \
                'theta_phi_': theta_phi_, 'mus': mus}
    savemat("vdp_Q_results.mat", results)

#### Plotting
plt.figure()
plt.subplot(321)
for i in range(N):
    plt.plot(t,x[:,i,2],label=i)
plt.plot(t,x_l[:,2],label='leader')
plt.xlabel('$time, s$')
plt.ylabel('$x$')
plt.legend()
plt.subplot(323)
for i in range(N):
    plt.plot(t,x[:,i,3])
plt.plot(t,x_l[:,3])
plt.xlabel('$time, s$')
plt.ylabel(r'$\dot{x}$')
plt.subplot(325)
for i in range(N):
    plt.plot(t,u[:,i])
plt.xlabel('$time, s$')
plt.ylabel('$u$')
# plt.subplot(322)
# for i in range(N):
#     plt.plot(t,theta[:,i])
# plt.xlabel('$time, s$')
# plt.ylabel(r'$\theta$')
# plt.subplot(324)
# plt.plot(t[:len(t)-1],theta_phi,linewidth=1.5,label='True')
# plt.plot(t[:len(t)-1],P[:,0],'--',linewidth=1,label='Est')
# plt.xlabel('$time, s$')
# plt.ylabel(r'$\sum_{i} {\Phi_{i}}^{T} \theta_{i}$')
# plt.legend()
plt.subplot(326)
plt.plot(t[1:len(t)],np.sum(r[:len(t)-1,:],axis=1),linewidth=1.5)
plt.plot(t[:len(t)-1],R[:,1],'--',linewidth=1)
plt.xlabel('$time, s$')
plt.ylabel(r'$\sum_{i}\, \rho_{i}$')

plt.show()
