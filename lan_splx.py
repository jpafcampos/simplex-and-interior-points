import numpy as np
from revised_simplex import revised_simplex
from otcrank import otcrank
from fklemint_pi_splx_norm import fkm_pi_splx_norm
from fklemint_pi_splx import fkm_pi_splx
from numpy.linalg import matrix_rank

# Setting the precision to long format
np.set_printoptions(precision=15)

# Turn off warnings about singular or badly scaled matrices
np.warnings.filterwarnings("ignore")

# Input for the dimension of the space
print('  ------------------------------------------------------')
ndim = int(input('             Dimensão  do  espaço          =  '))
print('  ------------------------------------------------------')

# Set the problem type to Simplex
ncode = 2

print('  ------------------------------------------------------')
valb = float(input('      Define o vetor b (10 < valb < 100)   =  '))
print('  ------------------------------------------------------')

# Construct the problem using the fklemint_pi_splx function
AS, b, CS, REL, XMIN, XMAX = fkm_pi_splx(ndim, ncode, valb)

# Calculate the rank of matrix A
#IRANK, _, _ = otcrank(AS)
IRANK = matrix_rank(AS)

# Check if the rank condition is met
if IRANK < len(XMIN):
    print('  -----------------------------------------------------------')
    print('     Rank (A) < dim. vet X . The problem is not consistent. ')
    print('  -----------------------------------------------------------')
else:
    # Define epsilon
    eps_x = np.finfo(float).eps

    # Define the first point as the origin
    Xini = np.zeros((ndim, 1))

    # Find the initial feasible basis (FBS)
    FBS = np.where(Xini != 0)[0]
    aux = np.where(Xini == 0)[0] + len(b)
    FBS = np.concatenate((FBS, aux), axis=0)

    # Solve the linear programming problem using the revised simplex method

    obj, x, y, iter, _, _, _ = revised_simplex(CS, AS, b, eps_x, FBS)

    x = x[:ndim]
    #Xsol = np.concatenate((np.zeros((ndim - 1, 1)), valb**(np.arange(ndim - 1, dtype=float)).reshape(-1, 1)))
    Xsol = np.concatenate((np.zeros((ndim - 1)), [valb**(ndim-1)])).reshape(-1,1)
    Xsol = Xsol.flatten()

    print("Xsol ", Xsol)
    print("Xsol Simplex ", x)
    print("FOBJ ", obj)
    
    erro_xsplx = 100 * np.linalg.norm(x - Xsol) / valb**(ndim - 1)
    erro_fobjsplx = 100 * abs(obj / (valb**(ndim - 1)) - 1)

    # Display results
    print('  ---------------------------------------------------------------  ')
    print(f'    Theoretical number of iterations for the simplex is  = {2**ndim}')
    print(f'    Simplex number of iterations is                      = {iter - 1}')
    print(f'    Simplex erro porcentual optimal objective value is   = {erro_fobjsplx}')
    print(f'    Simplex erro porcentual vetor x soluçao              = {erro_xsplx}')
    print('  ---------------------------------------------------------------  ')
