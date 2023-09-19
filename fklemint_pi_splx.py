import numpy as np

def fkm_pi_splx(n, ncode, valb):
    a = np.zeros((n, n))
    b = np.zeros((n, 1))
    c = np.zeros((n, 1))
    rel = np.zeros((n, 1))
    xmin = np.zeros((n, 1))
    xmax = np.zeros((n, 1))

    for i1 in range(n):
        # Determination of C
        c[i1, 0] = 2 ** (n - i1 -1)

        # Determination of A
        a[i1, i1] = 1.0
        for j1 in range(n):
            if j1 < i1:
                a[i1, j1] = 2 ** (i1 + 1 - j1)

        # Determination of b
        b[i1, 0] = valb**(i1)

        # REL -> Inequality relations: -1 = "Ax <= B".
        rel[i1, 0] = -1

        # Limits of X. Xmax and Xmin.
        xmin[i1, 0] = 0.0
        xmax[i1, 0] = b[i1, 0]

    # Construction of the Interior Point problem
    if ncode == 1:
        # Determination of A
        for i1 in range(n):
            a = np.append(a, np.zeros((1, n)), axis=0)
            a[n+i1, i1] = -1
            b = np.append(b, np.array([[0]]), axis=0)
            rel = np.append(rel, np.array([[-1]]), axis=0)

    # Construction of the Simplex problem
    if ncode == 2:
        # Determination of C
        c = np.vstack((c, np.zeros((n, 1))))

        # Determination of A with the addition of slack variables
        for i in range(n):
            a = np.insert(a, n+i, 0, axis=1)
            a[i, n+i] = 1

    return a, b, c, rel, xmin, xmax

# Example usage:
# Replace n, ncode, and valb with your desired values.
# a, b, c, rel, xmin, xmax = fklemint_pi_splx(n, ncode, valb)
