import numpy as np

def Murtyalgorithm_05(A, c, x):
    # Check if A is sparse and convert to full if necessary
    #if np.count_nonzero(A) / np.size(A) > 1e-4:
    #    A = np.array(A.todense())

    m, n = A.shape  # Number of inequality constraints and variables
    B = np.where(x > 0)[0]
    
    if len(B) == m and np.linalg.matrix_rank(A[:, B]) == m:
        return x, B  # Trivial case

    B = np.arange(1, n + 1)  # Index set of basic variables
    R, jb = rref(A, 0)  # Reduced row echelon form
    m = len(jb)  # Basis cardinality

    while len(B) > m:
        # Search direction
        a = np.arange(1, R.shape[1] + 1)
        a[jb] = []  # Columns that are not part of the identity matrix
        ia = np.argmax(x[B[a]])
        d = np.zeros(n)
        d[B[jb]] = -R[:m, a[ia]]
        d[B[a[ia]]] = 1

        # Step length
        z = np.dot(c, d)
        rate = x[B] / d[B]
        i = None

        if z <= 0:
            i = np.where((d[B] < 0) & (rate <= 0))[0]
            lambda_, ii = np.min(-rate[i]), np.argmin(-rate[i])
        elif z >= 0:
            i = np.where((d[B] > 0) & (rate >= 0))[0]
            lambda_, ii = np.max(-rate[i]), np.argmax(-rate[i])

        i = i[ii]  # Variable leaving basis

        # Stop criterion: unbounded optimal solution
        if i is None:
            B = []
            break

        # Step towards the search direction
        x = np.maximum(0, x + lambda_ * d)
        x[B[i]] = 0

        # Update basis
        B = np.delete(B, i)

        # Update reduced form
        if i in jb:
            # Pivot operation
            ib = np.where(R[:, i])[0]  # Row index
            rmax, in_ = np.max(np.abs(R[ib, a])), a[np.argmax(np.abs(R[ib, a]))]  # Column index

            if rmax > 1e-12:
                R[ib, :] = R[ib, :] / R[ib, in_]  # Normalization
                inp = np.arange(1, m + 1)
                inp[ib] = []
                R[inp, :] = R[inp, :] - R[inp, in_] * R[ib, :]  # Reduction
                jb[jb == i] = in_  # Update index of identity matrix
            else:
                R[ib, :] = 0
                R[ib[-1], :], R[m, :] = R[m, :].copy(), R[ib[-1], :].copy()  # Zero row comes last
                jb = jb[jb != i]
                m -= 1  # Reduce basis cardinality

    return x, B

# ==============================================================
# ============================================================== 
def rref(A, tol=None):
    m, n = A.shape
    r = 0  # Current row
    pivots = []  # Pivot columns
    for j in range(n):
        rows = np.where(np.abs(A[r:, j]) > tol)[0] + r  # Find rows with nonzero entries
        if len(rows) > 0:
            pivot_row = rows[0]  # Choose the first nonzero row as the pivot row
            pivots.append(j)
            A[[r, pivot_row]] = A[[pivot_row, r]]  # Swap current row with pivot row
            A[r, :] = A[r, :] / A[r, j]  # Normalize the pivot row
            for i in range(m):
                if i != r:
                    A[i, :] = A[i, :] - A[i, j] * A[r, :]  # Zero out the other rows
            r += 1  # Move to the next row
            if r == m:
                break
    return A, np.array(pivots)

# ==============================================================
# ============================================================== 
def rank(A, tol=None):
    if np.issparse(A):
        s = np.linalg.svd(A, compute_uv=False)
    else:
        s = np.linalg.svd(A)[1]
    if tol is None:
        tol = np.max(A.shape) * np.finfo(s.dtype).eps
    r = np.sum(s > tol)
    return r
