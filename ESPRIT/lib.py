import numpy as np


def get_array_output(theta, fc, N, S, array_pos, SNR=0):
    C = 3e8
    noise_power = 10**(-SNR/10)
    M = element_pos_vec.shape[1]
    d = theta.shape[0]
    ref_element_pos = element_pos_vec[:, :1]
    
    d_vec = np.asarray([np.cos(theta), np.sin(theta)])
    tau = np.dot((element_pos_vec - ref_element_pos).T, d_vec)/C
    phi = -2*np.pi*fc*tau
    
    return np.dot(np.exp(1j*phi), S) + noise_power*np.random.randn(M, N)

def get_ula_output(mu, N, M, S, SNR=0):
    C = 3e8
    noise_power = 10**(-SNR/10)
    d = mu.shape[0]
    wave_length = C/fc
    A = np.exp(np.repeat(1j*np.arange(M)[:, None], d, axis=1)*mu)
    
    return np.dot(A, S) + (np.random.randn(M, N) + 1j*np.random.randn(M, N))*(noise_power/np.sqrt(2))