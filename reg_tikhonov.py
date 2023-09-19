import numpy as np
from scipy.linalg import svd

def reg_tikhonov(A, b, lambda_val=1e-6):
    # Normalize the rows of A and b
    A_norm = A / np.linalg.norm(A, axis=1)[:, np.newaxis]
    b_norm = b / np.linalg.norm(b)
    
    m, n = A_norm.shape

    if m >= n:
        U, s, Vt = svd(A_norm, full_matrices=False)
    else:
        Vt, s, U = svd(A_norm.T, full_matrices=False)

    zeta = np.dot(U.T, b_norm)
    s_lambda = s / (s ** 2 + lambda_val ** 2)
    
    x_tikhonov = np.dot(Vt.T, zeta * s_lambda)
    
    return x_tikhonov

# Example usage:
# Replace 'A' and 'b' with your input matrix and vector
# x_tikhonov = reg_tikhonov(A, b, lambda_val=1e-6)
# print(x_tikhonov)
