import numpy as np
from scipy.cluster.vq import kmeans2


def update_tau_one_step(tau, alpha, pi, A, TAUMIN):
    part1 = np.tensordot(np.tensordot(np.sum(A, axis=0), tau, axes=(1, 0)), np.log(pi), axes=(1, 0))
    part21 = np.tensordot(np.tensordot(A.shape[0] - np.sum(A, axis=0), tau, axes=(1,0)), np.log(1-pi), axes=(1, 0))
    part22 = A.shape[0] * np.tensordot(tau, np.log(1-pi), axes=(1, 0))
    temp = part1 + part21 - part22
    temp2 = np.apply_along_axis(lambda x: alpha*np.exp(x-np.max(x)), 1, temp)
    temp3 = np.apply_along_axis(lambda x: np.true_divide(x, np.sum(x)), 1, temp2)
    temp4 = np.maximum(temp3, TAUMIN)
    return np.apply_along_axis(lambda x: np.true_divide(x, np.sum(x)), 1, temp4)

def update_alpha(tau, MSTEPPARAMMIN):
    temp = np.mean(tau, axis=0)
    temp2 = np.maximum(temp, MSTEPPARAMMIN)
    return np.true_divide(temp2, np.sum(temp2))

def update_pi(tau, A, MSTEPPARAMMIN):
    numerator = np.tensordot(np.tensordot(tau.T, np.sum(A, axis=0), axes=(1,0)), tau, axes=(1,0))
    denomenator = A.shape[0]*(np.multiply.outer(np.sum(tau, axis=0), np.sum(tau, axis=0)) - np.tensordot(tau.T, tau, axes=(1,0)))
    
    temp = np.true_divide(numerator, denomenator, out=np.ones_like(numerator)*np.mean(A), where = (denomenator > 0))
    
    return np.minimum(np.maximum(temp, MSTEPPARAMMIN), 1-MSTEPPARAMMIN)

def emConvergency(alpha, pi, alpha_back, pi_back, eps):
    if np.max(np.true_divide(np.abs(alpha_back - alpha), alpha)) >= eps:
        return False
    if np.max(np.true_divide(np.abs(pi_back - pi), pi)) >= eps:
        return False
    return True

def VEM(init_labels, A, n, Q, T, tmax = 10, smax = 5, fpeps = 1e-04, emeps = 1e-10, TAUMIN = 1e-10, MSTEPPARAMMIN = 1e-6):
    """
    tmax: maximum number of iterations for EM
    smax: maximum number of iterations for E-step
    emeps: threshold for EM
    TAUMIN: minimum paramter value for E-step
    MSTEPPARAMMIN: minimum paramter value for M-step
    """
    tau_0 = list()
    for value in init_labels:
        letter = [0 for _ in range(Q)]
        letter[value] = 1
        tau_0.append(letter)
    tau_0 = np.array(tau_0)

    tau_ls = [tau_0]
    alpha_ls = [update_alpha(tau_0, MSTEPPARAMMIN)]
    pi_ls = [update_pi(tau_0, A, MSTEPPARAMMIN)]
    
    for t in range(tmax):
        tau_prev = np.copy(tau_ls[-1])
        for s in range(smax): 
            tau_new = update_tau_one_step(tau_prev, alpha_ls[-1], pi_ls[-1], A, TAUMIN)
            if np.max(np.abs(tau_new - tau_prev)) <= fpeps:
                break
            tau_prev = np.copy(tau_new)

        alpha_new = update_alpha(tau_new, MSTEPPARAMMIN)
        pi_new = update_pi(tau_new, A, MSTEPPARAMMIN)

        if emConvergency(alpha_new, pi_new, alpha_ls[-1], pi_ls[-1], emeps):
            break

        tau_ls.append(tau_new)
        alpha_ls.append(alpha_new)
        pi_ls.append(pi_new)
        
    return tau_new, alpha_new, pi_new

def get_Z_hat(tau, Q):
    """
    One-hot vectors
    """
    Z_hat = np.identity(Q)[np.argmax(tau, axis=1)]
    return Z_hat

def get_MLE(tau, pi, A, n, Q, T):
    """
    MLE for fitted model
    """
    Z_hat = get_Z_hat(tau, Q)
    pi_hat = update_pi(Z_hat, A, 1e-10)
    
    return Z_hat, pi_hat

def loglikelihood(Z_hat, pi_hat, A, n, Q, T):
    """
    log likelihood of model determined by Z_hat and pi_hat
    """  
    p1 = 0.5*np.sum(np.tensordot(np.tensordot(Z_hat.T, np.sum(A, axis=0), axes=(1, 0)), Z_hat, axes=(1,0)) * np.log(pi_hat))
    p21 = 0.5*np.sum(np.tensordot(np.tensordot(Z_hat.T, A.shape[0] - np.sum(A, axis=0), axes=(1, 0)), Z_hat, axes=(1,0)) * np.log(1-pi_hat))
    p22 = -0.5*A.shape[0]*np.sum(np.tensordot(Z_hat.T, Z_hat, axes=(1,0)) * np.log(1-pi_hat))
    return p1+p21+p22
    
def MDL(Z_hat, pi_hat, A, n, Q, T):
    l = loglikelihood(Z_hat, pi_hat, A, n, Q, T)
    sizes = np.sum(Z_hat, axis=0)
    Q_0 = np.sum(sizes > 0)
    possible_edges = np.multiply.outer(sizes, sizes)
    np.fill_diagonal(possible_edges, sizes*(sizes-1) / 2)

    return (n+1)*np.log(Q_0) + 0.5*0.5*(np.sum(np.log(np.maximum(1, possible_edges))) + np.sum(np.log(np.maximum(1, np.diag(possible_edges))))) - l
    