import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
'''
July 2020
by Jing Wang (RLconsensus_NL_Con.m)
Reproduced by Elias Wilson
Consensus Based Reinforcement Learning Algorithm
Current System: Van der Pol Oscillator (2-state)
                \ddot{x} = mu*(1-x^2)*\dot{x} - x + u
'''
##### Inputs
# Time
T = 25
dt = 0.01
# Turn on Data Export to .mat
save_data = True
x0 = np.random.randn(8,2) # Intitial State (axis: 0-Agents  1-states)
mus = 0.1*np.random.randn(8) + 1 # Van der Pol system parameters (mu)
A = np.diag(1*np.ones(7), -1) + np.diag(1*np.ones(7), 1)
A_ = np.ones((8,8)) - np.identity(8) # Used for comparison
# Weighting Matricies
Q = [1, 1]*np.identity(2) # Size of single agent state space
U = 0.01*np.identity(8) # Size of total system action spaces
alpha0 = 0.01 # Initial Learning Rate
gamma = 0.99 # Discount Factor --Not Currently Used--
policy_iterations = 1
theta0 = np.ones((8,1)) # RBF Intitial Weights
sigma = np.array([3, 6, 6, 6, 6, 6, 6, 3]) # RBF Variance
sigma_ = 24/6*np.array([6, 6, 6, 6, 6, 6, 6, 6]) # RBF Variance
##### Functions
# System Dynamics (Van der Pol)
def f(x,u,mu=1):
    return x + dt*np.array([x[1], mu*(1-x[0]*x[0])*x[1] - x[0] + u])
# Control Derivative
def df_du():
    return dt*np.array([0, 1]).T    
# Jacobian
def df_dx(x,mu=1):
    return np.identity(2) + \
        dt*np.array([[0, 1],[-2*x[0]*x[1]-1, mu*(1-x[0]*x[0])]])
# Create dictionary with agent keys and a vector of indices for what other
# agents it can see
def find_adjacencies(N,A):
    adj = dict()
    for i in range(N):
        x = np.argwhere(A[i,:] > 0).reshape(-1)
        adj.update({i: x})
    return adj
# Gaussian Bell
def gaussian(x,sigma):
    return np.exp(-x.T.dot(x)/(sigma*sigma))
# Derivative of the Gaussian with respect to x
def gaussian_derivative(x,sigma):
    return -2*x/(sigma*sigma)*np.exp(-x.T.dot(x)/(sigma*sigma))
# RBF Evaluation
def Phi(x,adj,N,sigma,find_sum=False,theta=None):
    phis = dict()
    phi = np.ones(N)
    for i in adj.keys():
        ids = adj[i]
        phis.update({i: dict()})
        for j in ids:
            phi_x = gaussian(x[i,:]-x[j,:],sigma[i])
            phi[i] = phi_x*phi[i]
            phis[i].update({j: phi_x})

    if find_sum:
        action_sum = np.zeros(N)
        dphi_dx = np.zeros((N,len(x0[0,:])))
        for i in adj.keys():
            ids = adj[i]
            # i component
            for j in ids:
                dG_dx = gaussian_derivative(x[i,:] - x[j,:],sigma[i])
                for k in ids:
                    if k != j:
                        dG_dx = dG_dx*phis[i][j]
            dphi_dx[i,:] += dG_dx*theta[i,:]

            # j components
            for j in ids:
                dG_dx = gaussian_derivative(x[j,:] - x[i,:],sigma[i])
                jds = adj[j]
                for k in jds:
                    if k != i:
                        dG_dx = dG_dx*phis[j][k]
                dphi_dx[i,:] += dG_dx*theta[j,:]

            action_sum[i] = df_du().T.dot(dphi_dx[i,:].T)
        return phi, action_sum

    return phi
# Cost Function
def reward(x,u,adj,N):
    cost = np.zeros(N)
    for i in adj.keys():
        ids = adj[i]
        for j in ids:
            cost[i] += (x[i,:]-x[j,:]).dot(Q).dot((x[i,:]-x[j,:]).T)
    return cost + (u*u).dot(U)
# Cost Function
def left_eig(v,adj,N):
    v_ = np.zeros((N,N))
    for i in adj.keys():
        ids = adj[i]
        for j in ids:
            v_[i] += 1/(1+d[i])*(v[j] - v[i])
        v_[i] += v[i]
    return v_
# Calculate Action
def action(x,theta,action_sum):
    return (-1/2*np.linalg.inv(U).dot(action_sum.reshape(-1,1))).reshape(-1)
# Update Estimates
def estimate(R,G,P,r,_phi,phi,phi_,ii,theta,N,adj,w,d):

    # R_, G_, P_ = np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(N):
        ids = adj[i]
        R_sum, G_sum, P_sum = 0, 0, 0
        for j in ids:
            R_sum += A[i,j]*(R[ii-1,j] - R[ii-1,i])
            G_sum += A[i,j]*(G[ii-1,j] - G[ii-1,i])
            P_sum += A[i,j]*(P[ii-1,j] - P[ii-1,i])

        R[ii,i] = R[ii-1,i] + 1/(1+d[i])*R_sum + 1/w[i,i]*(r[ii,i] - r[ii-1,i])
        G[ii,i] = G[ii-1,i] + 1/(1+d[i])*G_sum + 1/w[i,i]*(phi_[i]*theta[ii,i] - \
                phi[i]*theta[ii-1,i])
        P[ii,i] = P[ii-1,i] + 1/(1+d[i])*P_sum + 1/w[i,i]*(phi[i]*theta[ii,i] - \
                _phi[i]*theta[ii-1,i])

    return R, G, P
##### Main
N = len(A[:,0]) # Number of agents
adj = find_adjacencies(N,A) # find the adjacenies (who can agent i see)
adj_ = find_adjacencies(N,A_)
t = np.arange(0, T, dt) # time vector
# Allocate memory and fill ICs
x = np.zeros((len(t),N,len(x0[0,:])))
x[0,:,:] = x0
x_ = np.zeros((len(t),N,len(x0[0,:])))
x_[0,:,:] = x0
u = np.zeros((len(t)-1,len(U[:,0])))
u_ = np.zeros((len(t)-1,len(U[:,0])))
theta = np.zeros((len(t),len(theta0[:,0]),len(theta0[0,:])))
theta[0,:,:] = theta0
theta_ = np.zeros((len(t),len(theta0[:,0]),len(theta0[0,:])))
theta_[0,:,:] = theta0
theta_phi = np.zeros(len(t)-1)
theta_phi_ = np.zeros(len(t)-1)
r = np.zeros((len(t)-1,N))
r_ = np.zeros((len(t)-1,N))
R = np.zeros((len(t)-1,N))
G = np.zeros(np.shape(R))
P = np.zeros(np.shape(R))
v = np.zeros((len(t),N,N))
v[0] = np.identity(N)
print('VDP Parameters')
print(mus)
# Left eigenvector of F
d = np.sum(A, axis=1) # degree vector
D = np.diag(d) # degree matrix
La = D - A # Laplacian
F = np.identity(N) - np.linalg.inv(np.identity(N) + D).dot(La)
W,V = np.linalg.eig(F.T)
w_id = np.argmin(np.power((W-1),2)) # find the index of eigenvalue 1
v_true = V[:,w_id] # left eigenvector associated with eigenvalue 1
if v_true[0] < 0: # Keep the eigenvector positive
    v_true = -v_true
v_true = v_true/np.sum(v_true)
print('Left Eigenvector')
print(v_true)
for i in range(len(t)-1):
    alpha = alpha0/(dt*i+1)
    v[i+1] = left_eig(v[i],adj,N)
    # Estimate
    phi, action_sum = Phi(x[i,:,:],adj,N,sigma,find_sum=True,theta=theta[i,:,:])
    u[i,:] = action(x[i,:,:],theta[i,:,:],action_sum)
    for j in range(N):
        x[i+1,j,:] = f(x[i,j,:],u[i,j],mu=mus[j])
    # Policy Iteration
    for _ in range(policy_iterations-1):
        _, action_sum = Phi(x[i+1,:,:],adj,N,sigma,find_sum=True,theta=theta[i,:,:])

        u[i,:] = action(x[i+1,:,:],theta[i,:,:],action_sum)

        for j in range(N):
            x[i+1,j,:] = f(x[i,j,:],u[i,j],mu=mus[j])

    r[i,:] = reward(x[i,:,:],u[i,:],adj,N)

    if i == 0:
        R[0,:] = r[0,:]/v_true
        G[0,:] = Phi(x[i+1,:,:],adj,N,sigma)*theta[0,:,:].reshape(-1)/v_true
        P[0,:] = Phi(x[i,:,:],adj,N,sigma)*theta[0,:,:].reshape(-1)/v_true

    if i > 0:
        _phi = Phi(x[i-1,:,:],adj,N,sigma)
        phi_ = Phi(x[i+1,:,:],adj,N,sigma)
        R, G, P = estimate(R,G,P,r,_phi,phi,phi_,i,theta,N,adj,v[i],d)

    theta[i+1,:,:] = theta[i,:,:] + \
            alpha*((R[i,:] + G[i,:] - P[i,:])*phi).reshape(-1,1)

    phi = Phi(x[i,:,:],adj_,N,sigma)
    phi_ = Phi(x[i+1,:,:],adj_,N,sigma)

    theta_phi[i] = phi.dot(theta[i,:,:])
    theta_phi_[i] = phi_.dot(theta[i,:,:])

    # True
    phi, action_sum = Phi(x_[i,:,:],adj_,N,sigma_,find_sum=True,theta=theta_[i,:,:])

    u_[i,:] = action(x_[i,:,:],theta_[i,:,:],action_sum)

    for j in range(N):
        x_[i+1,j,:] = f(x_[i,j,:],u_[i,j],mu=mus[j])

    # Policy Iteration
    for _ in range(policy_iterations-1):
        _, action_sum = Phi(x_[i+1,:,:],adj_,N,sigma_,find_sum=True,theta=theta_[i,:,:])

        u_[i,:] = action(x_[i+1,:,:],theta_[i,:,:],action_sum)

        for j in range(N):
            x_[i+1,j,:] = f(x_[i,j,:],u_[i,j],mu=mus[j])

    r_[i,:] = reward(x_[i,:,:],u_[i,:],adj_,N)

    phi_ = Phi(x_[i+1,:,:],adj_,N,sigma_)

    theta_[i+1,:,:] = theta_[i,:,:] + alpha*(np.sum(r_[i,:]) + \
            phi_.dot(theta_[i,:,:])-phi.dot(theta_[i,:,:]))*phi.reshape(-1,1)


### Export Data
if save_data:
    results = {'x': x, 'u': u, 'R': R, 'G': G, 'P': P, 'theta_phi': theta_phi, \
                'r': r, 't': t, 'theta': theta, 'theta_phi_': theta_phi_,
                 'x_': x_, 'u_': u_, 'r_': r_, 'theta_': theta_}
    savemat("vdp_results.mat", results)

#### Plotting
plt.figure()
plt.subplot(321)
for i in range(N):
    plt.plot(t,x[:,i,0],label=i)
plt.xlabel('$time, s$')
plt.ylabel('$x$')
plt.legend()
plt.subplot(323)
for i in range(N):
    plt.plot(t,x[:,i,1])
plt.xlabel('$time, s$')
plt.ylabel(r'$\dot{x}$')
plt.subplot(325)
for i in range(N):
    plt.plot(t[:len(t)-1],u[:,i])
plt.xlabel('$time, s$')
plt.ylabel('$u$')
plt.subplot(322)
for i in range(N):
    plt.plot(t,theta[:,i])
plt.xlabel('$time, s$')
plt.ylabel(r'$\theta$')
plt.subplot(324)
plt.plot(t[:len(t)-1],theta_phi,linewidth=1.5,label='True')
plt.plot(t[:len(t)-1],P[:,0],'--',linewidth=1,label='Est')
plt.xlabel('$time, s$')
plt.ylabel(r'$\sum_{i} {\Phi_{i}}^{T} \theta_{i}$')
plt.legend()
plt.subplot(326)
plt.plot(t[1:len(t)],np.sum(r[:len(t)-1,:],axis=1),linewidth=1.5)
plt.plot(t[:len(t)-1],R[:,1],'--',linewidth=1)
plt.xlabel('$time, s$')
plt.ylabel(r'$\sum_{i}\, \rho_{i}$')

plt.figure()
plt.subplot(321)
for i in range(N):
    plt.plot(t,x_[:,i,0],label=i)
plt.xlabel('$time, s$')
plt.ylabel('$x$')
plt.legend()
plt.subplot(323)
for i in range(N):
    plt.plot(t,x_[:,i,1])
plt.xlabel('$time, s$')
plt.ylabel(r'$\dot{x}$')
plt.subplot(325)
for i in range(N):
    plt.plot(t[:len(t)-1],u_[:,i])
plt.xlabel('$time, s$')
plt.ylabel('$u$')
plt.subplot(322)
for i in range(N):
    plt.plot(t,theta_[:,i])
plt.xlabel('$time, s$')
plt.ylabel(r'$\theta$')
plt.subplot(324)
plt.plot(t[:len(t)-1],theta_phi,linewidth=1.5,label='True')
plt.plot(t[:len(t)-1],P[:,0],'--',linewidth=1,label='Est')
plt.xlabel('$time, s$')
plt.ylabel(r'$\sum_{i} {\Phi_{i}}^{T} \theta_{i}$')
plt.legend()
plt.subplot(326)
plt.plot(t[1:len(t)],np.sum(r[:len(t)-1,:],axis=1),linewidth=1.5)
plt.plot(t[1:len(t)],np.sum(r_[:len(t)-1,:],axis=1),linewidth=1.5)
plt.plot(t[:len(t)-1],R[:,1],'--',linewidth=1)
plt.xlabel('$time, s$')
plt.ylabel(r'$\sum_{i}\, \rho_{i}$')

v_plot = np.zeros((len(t),N))
for i in range(len(t)):
    v_plot[i] = np.diag(v[i])

plt.figure()
plt.plot(t,v_plot)
for i in range(N):
    plt.plot([t[0],t[len(t)-1]],[v_true[i],v_true[i]],'--')

plt.show()
