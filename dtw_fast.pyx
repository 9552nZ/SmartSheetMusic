import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY

'''
Cython implementation of the Dynamic Time Warping.
Allows for subsequence alignment and maximum step constraint.

Run the below commands to compile:
cd C://Users\Alexis\Source\TestPython2\AutomaticAudioTranscript 
python setup.py build_ext --inplace

'''

DTYPE = np.double
ctypedef np.double_t DTYPE_t

ITYPE = np.int
ctypedef np.int_t ITYPE_t

cdef double min3(double a, double b, double c):
    '''
    Get the min across three numbers
    '''
    cdef double m = a
    if b < m:
        m = b
    if c < m:
        m = c
    return m

cdef unsigned int argmin3(double a, double b, double c):
    '''
    Get the index of the min across three numbers.
    '''
    cdef double m = a
    cdef unsigned int arg_m = 0
    if b < m:
        m = b
        arg_m = 1
    if c < m:
        m = c
        arg_m = 2
    return arg_m

    
def dtw_fast(np.ndarray[DTYPE_t,ndim=2] C, np.ndarray[DTYPE_t,ndim=1] weights_mul, unsigned int subseq, unsigned int max_consecutive_steps):
    '''
    Perform Dynamic Time Warping and return the cumulative distance matrix.
    
    Parameters
    ----------        
    C : np.ndarray [shape=(r,c)]
        The pre-computed cost matrix.
        
    weights_mul : np.ndarray [shape=(3,)]
        The multiplicative weights for the steps penalty. 
        The first item is the diagonal penalty.
        
    subseq : int > 0
        subseq > 0 : allow for subsequence alignment.
        subseq == 0 : full alignment.
        
    max_consecutive_steps : int > 0
        The maximum number of consecutive steps allowed (in either the row or columns directions).
        Should we reach the maximum number of steps, we make a diagonal move.
        May be used as  a local direction constraint.
        Set to the max(r, c) or more so as it does not bite.
        
    Returns :
    ----------
    D : np.ndarray [shape=(r,c)]
        The cumulative costs matrix. The minimum over the last column is the (local) DTW distance.
        It may contains inf values, should there be items that could now be accessed (because of local constraints).
        
    
    '''        
    
    if weights_mul.shape[0] != 3:
        raise ValueError("weights_mul need to have three entries.")
    
    cdef unsigned int r = C.shape[0]
    cdef unsigned int c = C.shape[1]
    
    # Set up placeholders for the cumulative distance matrix, the number of 
    # consecutive steps down and number of consecutive steps to the right.
    cdef np.ndarray[DTYPE_t,ndim=2] D = np.zeros((r,c), dtype = DTYPE)
    cdef np.ndarray[ITYPE_t,ndim=2] steps_down = np.zeros((r,c), dtype = ITYPE)
    cdef np.ndarray[ITYPE_t,ndim=2] steps_right = np.zeros((r,c), dtype = ITYPE)

    # Initialise the first row, start from the origin.     
    D[0, 0:max_consecutive_steps] = np.cumsum(C[0, 0:max_consecutive_steps])
    D[0, max_consecutive_steps:c] = INFINITY

    # Initialise the first column. If we allow for subsequence, do 
    # not start from the origin.         
    if subseq > 0:        
        D[:, 0] = C[:, 0]     
    else:
        D[0:max_consecutive_steps, 0] = np.cumsum(C[0:max_consecutive_steps, 0])
        D[max_consecutive_steps:r, 0] = INFINITY
    
    cdef unsigned int i,j
    cdef DTYPE_t dist_step_diag
    cdef DTYPE_t dist_down
    cdef DTYPE_t dist_right
    cdef unsigned int best_step
    
    # Loop over rows / cols (span the entire matrix)
    for i in range(r-1):
        for j in range(c-1):
            
            # Get the (adjusted) costs for the three possible steps  
            dist_step_diag = D[i, j] + C[i+1, j+1] * weights_mul[0]
            dist_down = D[i, j+1] + C[i+1, j+1] * weights_mul[1]
            dist_right = D[i+1, j] + C[i+1, j+1] * weights_mul[2]
            
            # Find the step with the minimum cost
            best_step = argmin3(dist_step_diag, dist_down, dist_right)
            
            # If it a steps down or to the right, make sure we have not reached the max possible
            if best_step == 1:
                # If we reached the max, force to take the diag step
                if steps_down[i,j+1] == max_consecutive_steps:  
                    D[i+1,j+1] = dist_step_diag
                else:
                    steps_down[i+1,j+1] = steps_down[i,j+1] + 1
                    D[i+1,j+1] = dist_down
                                
            elif best_step == 2:
                if steps_right[i+1,j] == max_consecutive_steps:                    
                    D[i+1,j+1] = dist_step_diag
                else:
                    steps_right[i+1,j+1] = steps_right[i+1,j] + 1    
                    D[i+1,j+1] = dist_right                
            
            # Diag steps are ok
            else:                        
                D[i+1,j+1] = dist_step_diag
    
    # Return the entire matrix
    return D        
