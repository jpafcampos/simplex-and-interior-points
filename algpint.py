import numpy as np
from scipy.linalg import solve
np.set_printoptions(precision=15) 

def algpint_01(A, B, C, REL, X0, XMIN, XMAX, epslon=None, KALPHA=None, MAXITER=None):
    # Specified tolerance such that a calculated DELTAX is considered zero
    if epslon is None:
        epslon = 1.0e-08

    #system epsilon
    eps = 2.220446049250313e-16

    # Specified value considered zero
    valzero = epslon ** 3

    # Specified value considered infinite
    INFMAX = 1 / (eps ** 4)

    # Default value for KALPHA
    if KALPHA is None:
        KALPHA = 0.9

    # Verifies if X0 is an interior point
    if np.sign(np.dot(A, X0) - B).any() != REL.any():
        print('   ---------------------------------- ')
        print('       X0 não é um ponto interior!    ')
        print('                                      ')
        return None, None, None, None, None, None, None

    # Define the values of N and M
    M, N = A.shape

    # Initialize variables
    X = X0
    EVOLX = X0
    FOBJ = np.dot(C.T, X0)
    EVOLFOBJ = [FOBJ]
    yk = None
    gap_final = None

    # Loop
    ITER = 1
    FLAG1 = 0
    Dk = np.zeros((M, M))

    while ITER <= MAXITER and FLAG1 == 0:
        # Step I.
        print("A e B")
        print(A)
        print(B)
        Vk = B - np.dot(A, X)
        #Vk = B - A @ X 
        Vk = np.maximum(Vk, eps ** 4)
        print(Vk)

        Dk = np.diag(1.0 / Vk.reshape(-1))
        print(Dk)

        # Step II.
        #ATA = A.T @ Dk @ Dk @ A
        #dk = solve(ATA, C, assume_a='gen')

        dk = np.linalg.solve(np.dot(np.dot(A.T, Dk), np.dot(Dk, A)), C)
        dv = -np.dot(A, dk)

        # Step III.
        Vaux = np.where(dv > -valzero, -INFMAX, Vk / dv)
        Valpha = np.max(Vaux)
        ALPHA = KALPHA
        print("alpha ", ALPHA)
        print("valpha ", Valpha)
        print("x old ", X)
        print("dk ", dk)
        print("termo ",  ALPHA * Valpha * dk)
        X = X - ALPHA * Valpha * dk
        print("x new ", X)  
        # Step IV.
        for J1 in range(N):
            if X[J1, 0] < XMIN[J1, 0]:
                X[J1, 0] = XMIN[J1, 0]
            if X[J1, 0] > XMAX[J1, 0]:
                X[J1, 0] = XMAX[J1, 0]

        if ITER != 1:
            DELTAX = X - XOLD
            #print("deltax ", DELTAX)
            if np.max(np.abs(DELTAX)) < epslon:
                XOLD = X
                FLAG1 = 1
                
            FOBJ = np.dot(C.T, X)
            EVOLFOBJ.append(FOBJ)

            yk = -np.dot(np.dot(Dk, Dk), dv)
            #print("yk ", yk)
            gap_final = np.abs((np.dot(B.T, yk) - FOBJ) / max([1, np.abs(FOBJ)]))
            print("gap final ", gap_final)
            if gap_final < epslon:
                FLAG1 = 1

        EVOLX = np.hstack((EVOLX, X))
        XOLD = X

        if ITER >= MAXITER:
            FLAG1 = 1
        elif FLAG1 == 0:
            ITER += 1

    return ITER, EVOLX, EVOLFOBJ, X, FOBJ, yk, gap_final

# Example usage:
# Replace A, B, C, REL, X0, XMIN, XMAX, epslon, KALPHA, and MAXITER with your data.
# ITER, EVOLX, EVOLFOBJ, X, FOBJ, yk, gap_final = algpint_01(A, B, C, REL, X0, XMIN, XMAX, epslon, KALPHA, MAXITER)

