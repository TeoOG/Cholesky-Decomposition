#Theodoros Goltsios 1991
import time
import numpy as np
from math import sqrt
from scipy.linalg import solve_triangular

start_time = time.time()

def mycholesky(A):
    """Performs a Cholesky decomposition of A, which must
    be a symmetric and positive definite matrix. The function
    returns the lower variant triangular matrix, L."""


    # Create zero matrix for L
    #L = [[0.0] * n for i in range(n)]

    n = len(A)
    L = np.zeros((n, n), dtype=float)

    # Perform the Cholesky decomposition

    for i in range(n):
        for k in range(i+1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))

            if (i == k):  # Diagonal elements
                # LaTeX: l_{kk} = \sqrt{ a_{kk} - \sum^{k-1}_{j=1} l^2_{kj}}
                L[i][k] = sqrt(A[i][i] - tmp_sum)
            else:
                # LaTeX: l_{ik} = \frac{1}{l_{kk}} \left( a_{ik} - \sum^{k-1}_{j=1} l_{ij} l_{kj} \right)
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    return L

############################################################################################################3
def teos_cholesky(A):
    """Custom Cholesky decomposition of A, which must
    be a symmetric and positive definite matrix and also pentadiagonial. The function
    returns the lower variant triangular matrix, L which has two elements below the main diagonial.Linear
    Time Execution ."""

    n = len(A)
    L = np.zeros((n, n), dtype=float)

    for i in range (n):
        # first row
        if(i==0):
            L[i][i]=sqrt(A[i][i])
        # second row
        if(i==1):
            L[i][i - 1] = (1.0 / L[i - 1][i - 1]) * A[i][i - 1]
            L[i][i] = sqrt(A[i][i] - (L[i][i-1])*(L[i][i-1]))
        # from third till end
        if( (i!=1) and (i!=0) ):
            L[i][i - 2] = (1.0 / L[i - 2][i - 2]) * A[i][i - 2]
            L[i][i - 1] = (1.0 / L[i-1][i-1] * (A [i][i-1] - (L[i][i-2]*L[i-1][i-2]) ) )
            L[i][i] = sqrt(A[i][i] - ( (L[i][i-1])*(L[i][i-1]) +(L[i][i-2])*(L[i][i-2])) )

    return L



n = 10

################## AAAAA matrix #############################################
A = np.zeros([n, n], dtype=float)  # initialize to f zeros

# ------------------first row
A[0][0] = 7
A[0][1] = -4
A[0][2] = 1
# ------------------second row
A[1][0] = -4
A[1][1] = 7
A[1][2] = -4
A[1][3] = 1
# --------------two last rows-----
# n-2 row
A[- 2][- 1] = -4
A[- 2][- 2] = 7
A[- 2][- 3] = -4
A[- 2][- 4] = 1
# n-1 row
A[- 1][- 1] = 7
A[- 1][- 2] = -4
A[- 1][- 3] = 1

# --------------------------- from second to n-2 row --------------------------#
j = 0
for i in range(2, n - 2):
    if j == (n - 4):
        break
    A[i][j] = 1
    j = j + 1

j = 1
for i in range(2, n - 2):
    if j == (n - 3):
        break
    A[i][j] = -4
    j = j + 1

j = 2
for i in range(2, n - 2):
    if j == (n - 2):
        break
    A[i][j] = 7
    j = j + 1

j = 3
for i in range(2, n - 2):
    if j == (n - 1):
        break
    A[i][j] = -4
    j = j + 1

j = 4
for i in range(2, n - 2):
    if j == (n):
        break
    A[i][j] = 1
    j = j + 1
# -----------------------------end coding of 2nd to n-2 r-------------#
print("\nMatrix A is : \n", A)

####### b matrix ######################################
b = np.zeros(n,float).reshape((n,1))
b[0] = 3
b[1] = -1
#b[len(b) - 1] = 3
#b[len(b) - 2] = -1
b[[0,-1]]=3; b[[1,-2]]=-1

print("\nMatrix b is \n", b)

#################### result ########################

#    The Cholesky decomposition is often used as a fast way of solving
#
#                        Ax= b
#
#     (when A is both Hermitian/symmetric and positive-definite).
#
#     First, we solve for y in
#
#                        Ly = b,
#
#     and then for x in
#
#                       L.H x = y.


#myL=np.linalg.cholesky(A)
myL=teos_cholesky(A)


#check_x = np.linalg.solve(A, b)

#check if the composition was done right
myLT=myL.T.conj() #transpose matrix
#Ac=np.dot(myL,myLT) #should give the original matrix A


y=solve_triangular(myL,b,lower=True)

x=solve_triangular(myLT, y,lower=False)
print("--- %s seconds ---" % (time.time() - start_time))
