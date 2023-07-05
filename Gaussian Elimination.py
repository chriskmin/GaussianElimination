
import numpy as np
import warnings

def swapRows(A, i, j):
    """
    interchange two rows of A
    operates on A in place
    """
    tmp = A[i].copy()
    A[i] = A[j]
    A[j] = tmp

def relError(a, b):
    """
    compute the relative error of a and b
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return np.abs(a-b)/np.max(np.abs(np.array([a, b])))
        except:
            return 0.0

def rowReduce(A, i, j, pivot):
    """
    reduce row j using row i with pivot pivot, in matrix A
    operates on A in place
    """
    factor = A[j][pivot] / A[i][pivot]
    for k in range(len(A[j])):
        if np.isclose(A[j][k], factor * A[i][k]):
            A[j][k] = 0.0
        else:
            A[j][k] = A[j][k] - factor * A[i][k]


# stage 1 (forward elimination)
def forwardElimination(B):
    """
    Return the row echelon form of B
    """
    A = B.copy().astype(float)
    m, n = np.shape(A)
    for i in range(m-1):
        # Let leftmostNonZeroCol be the position of the leftmost nonzero value 
        # in row i or any row below it 
        leftmostNonZeroRow = m
        leftmostNonZeroCol = n
        ## for each row below row i (including row i)
        for h in range(i,m):
            ## search, starting from the left, for the first nonzero
            for k in range(i,n):
                if (A[h][k] != 0.0) and (k < leftmostNonZeroCol):
                    leftmostNonZeroRow = h
                    leftmostNonZeroCol = k
                    break
        # if there is no such position, stop
        if leftmostNonZeroRow == m:
            break
        # If the leftmostNonZeroCol in row i is zero, swap this row 
        # with a row below it
        # to make that position nonzero. This creates a pivot in that position.
        if (leftmostNonZeroRow > i):
            swapRows(A, leftmostNonZeroRow, i)
        # Use row reduction operations to create zeros in all positions 
        # below the pivot.
        for h in range(i+1,m):
            rowReduce(A, i, h, leftmostNonZeroCol)
    return A

#################### 
# If any operation creates a row that is all zeros except the last element,
# the system is inconsistent; stop.
def inconsistentSystem(A):
    """
    B is assumed to be in echelon form; return True if it represents
    an inconsistent system, and False otherwise
    """
    m, n = np.shape(A)
    #loops through each row of the matrix
    for i in range(m):
        #gives us the indices of non-zero numbers in row i
        x = np.nonzero(A[i])
        #checks to seee if i isn't an empty list
        if x[0].size>0:
            #if not an empty list, then checks to see if the last element is the only non-zero
            if x[0][0] == n-1:
                return True
    return False

def backsubstitution(B):
    """
    return the reduced row echelon form matrix of B
    """
    A = B.copy().astype(float)
    m, n = np.shape(A)
    #loops from the bottom of the matrix
    for i in reversed(range(m)):
        x = np.nonzero(A[i])
        if x[0].size>0:
            colOfPivot = x[0][0]
            factor = A[i][colOfPivot]
            A[i] = A[i]/ factor
            for k in range(i):
                rowReduce(A, i, k, colOfPivot)
    return A
        

#####################
def test():
    A = np.loadtxt('h2m1.txt')
    B = np.loadtxt('h2m2.txt')
    C = np.loadtxt('h2m3.txt')
    D = np.loadtxt('h2m4.txt')
    E = np.loadtxt('h2m5.txt')
    F = np.loadtxt('h2m6.txt')
    
    AEchelon = forwardElimination(A)
    BEchelon = forwardElimination(B)
    CEchelon = forwardElimination(C)
    DEchelon = forwardElimination(D)
    EEchelon = forwardElimination(E)
    FEchelon = forwardElimination(F)
    
    if (not inconsistentSystem(AEchelon)):
        AReducedEchelon = backsubstitution(AEchelon)
        print(AReducedEchelon)

    if (not inconsistentSystem(BEchelon)):
        BReducedEchelon = backsubstitution(BEchelon)
        print(BReducedEchelon)
        
    if (not inconsistentSystem(CEchelon)):
        CReducedEchelon = backsubstitution(CEchelon)
        print(CReducedEchelon)
        
    if (not inconsistentSystem(DEchelon)):
        DReducedEchelon = backsubstitution(DEchelon)
        print(DReducedEchelon)
        
    if (not inconsistentSystem(EEchelon)):
        EReducedEchelon = backsubstitution(EEchelon)
        print(EReducedEchelon)
        
    if (not inconsistentSystem(FEchelon)):
        FReducedEchelon = backsubstitution(FEchelon)
        print(FReducedEchelon)
    
    
   

