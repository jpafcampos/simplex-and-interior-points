import numpy as np

def fkm_pi_splx_norm(n, ncode, kvalb):
    a = np.zeros((n - 1, n))
    b = np.zeros((n - 1, 1))
    c = np.zeros((n, 1))
    rel = np.zeros((n - 1, 1))
    xmin = np.zeros((n, 1))
    xmax = np.zeros((n, 1))

    xmin[0, 0] = 0.0
    xmax[0, 0] = 1.0

    for i1 in range(n - 1):
        # Determination of C
        c[i1, 0] = 0

        # Determination of A
        a[i1, i1] = kvalb
        a[i1, i1 + 1] = -1.0

        a[i1 + n - 1, i1] = kvalb
        a[i1 + n - 1, i1 + 1] = 1.0

        # Determination of b
        b[i1, 0] = 0
        b[i1 + n - 1, 0] = 1

        # REL -> Inequality relations: -1 = "Ax <= B".
        rel[i1, 0] = -1
        rel[i1 + n - 1, 0] = -1

        # Limits of X. Xmax and Xmin.
        xmin[i1 + 1, 0] = kvalb * xmax[i1, 0]
        xmax[i1 + 1, 0] = 1.0 - xmin[i1 + 1, 0]

    c[n - 1, 0] = 1

    # Construction of the Interior Point problem
    if ncode == 1:
        a = np.append(a, np.zeros((2, n)), axis=0)
        b = np.append(b, np.array([[0], [1]]), axis=0)
        rel = np.append(rel, np.array([[-1], [-1]]), axis=0)
        a[2 * n - 2, 0] = -1
        a[2 * n - 1, 0] = 1

    # Construction of the Simplex problem
    if ncode == 2:
        c = np.vstack((c, np.zeros((2 * n - 2, 1))))

        for i in range(2 * n - 2):
            a = np.insert(a, n + i, 0, axis=1)
            a[i, n + i] = 1

    return a, b, c, rel, xmin, xmax

# Example usage:
# Replace n, ncode, and kvalb with your desired values.
# a, b, c, rel, xmin, xmax = fkm_pi_splx_norm(n, ncode, kvalb)
