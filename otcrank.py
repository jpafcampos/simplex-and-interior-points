import numpy as np

def otcrank(matrix):
    # Initialize flags and parameters
    flag2 = False
    epslon = 1.0e-17
    linhas, colunas = matrix.shape
    
    if linhas > colunas:
        # Transpose the matrix if there are more rows than columns
        matrix = matrix.T
        linhas, colunas = colunas, linhas
        flag2 = True

    irank = linhas
    ic = np.arange(colunas)
    ir = np.arange(linhas)
    flag1 = False
    iout = -1

    while not flag1:
        iout += 1
        if iout > irank:
            flag1 = True
        else:
            piv = 0.0
            for j in range(iout, colunas):
                for i in range(iout, irank):
                    d = abs(matrix[i, j])
                    if d > piv:
                        li = i
                        jin = j
                        lj = j
                        piv = d
            
            # Detection of rank deficiency
            if piv <= epslon:
                irank = iout - 1
            elif li != iout:
                matrix[li, :], matrix[iout, :] = matrix[iout, :].copy(), matrix[li, :].copy()
                k = ir[li]
                ir[li] = ir[iout]
                ir[iout] = k

            # A Gauss-Jordan elimination step
            print(iout, jin)
            pivot = matrix[iout, jin]
            if pivot != 0:
                matrix[iout, :] = matrix[iout, :] / pivot

            for i in range(0, irank):
                if i != iout:
                    d = matrix[i, jin]
                    matrix[i, :] = matrix[i, :] - d * matrix[iout, :]

            k = ic[lj]
            ic[lj] = ic[iout]
            ic[iout] = k

    if flag2:
        # If the matrix was transposed, exchange ir and ic
        ir, ic = ic, ir

    return irank, ir, ic

# Example usage:
# Replace 'matrix' with your input matrix
# irank, ir, ic = otc_rank(matrix)
# print(f"Rank: {irank}")
# print(f"Row Indices: {ir}")
# print(f"Column Indices: {ic}")
