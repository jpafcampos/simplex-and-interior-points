import numpy as np

def revised_simplex(c, A, b, eps_x, B):
    m, n = A.shape
    # Step 1: Initialize
    N = list(set(range(n)) - set(B))
    xB = np.linalg.solve(A[:, B], b)
    
    xb1 = xB
    iter = 0
    evolx = []
    while True:
        iter += 1

        # Step 2: Solve A_B^T * y = c_B and compute s_N
        # Extract submatrices A[:, B] and c[B]
        A_sub = A[:, B]
        A_sub = np.transpose(A_sub)
        c_sub = c[B]

        y = np.linalg.solve(A_sub, c_sub)
        #y = np.linalg.pinv(A_sub) @ c_sub
        sN = c[N] - A[:, N].T @ y
        sNmax = np.max(sN)
        k = np.argmax(sN)
        if sNmax <= eps_x:
            x = np.zeros(n)
            xB = xB.ravel()
            x[B] = xB
            obj = c.ravel() @ x
            evolx.append(x)
            return obj, x, y, iter, evolx, xb1, B

        # Step 3: Solve A_Bd = a_{N(k)}
        d = np.linalg.solve(A[:, B], A[:, N[k]])
        zz = np.where(d > eps_x)[0]
        if len(zz) == 0:
            raise Exception('System is unbounded')

        # Find theta

        theta, ii = np.min(xB[zz].reshape(-1,1) / d[zz].reshape(-1,1)), np.argmin(xB[zz].reshape(-1,1) / d[zz].reshape(-1,1))

        l = zz[ii]
  
        # Step 4: Update B and N
        temp = B[l]
        B[l] = N[k] 
        N[k] = temp
        xB = xB.ravel()
        xB -= theta * d
        xB[l] = theta

# Example usage:
# obj, x, y = revised_simplex(c, A, b, 1e-3, B)
