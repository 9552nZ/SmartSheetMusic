import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY

DTYPE = np.double
ctypedef np.double_t DTYPE_t

ITYPE = np.int
ctypedef np.int_t ITYPE_t

cdef double min3(double a, double b, double c):
    cdef double m = a
    if b < m:
        m = b
    if c < m:
        m = c
    return m

cdef unsigned int argmin3(double a, double b, double c):
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
    
    if weights_mul.shape[0] != 3:
        raise ValueError("weights_mul need to have three entries.")
    
    cdef unsigned int r = C.shape[0]
    cdef unsigned int c = C.shape[1]
    
    cdef np.ndarray[DTYPE_t,ndim=2] D = np.zeros((r,c), dtype = DTYPE)
    cdef np.ndarray[ITYPE_t,ndim=2] steps_down = np.zeros((r,c), dtype = ITYPE)
    cdef np.ndarray[ITYPE_t,ndim=2] steps_right = np.zeros((r,c), dtype = ITYPE)

    # Set starting point to C[0, 0]    
#     D[0, :] = np.cumsum(C[0,:])
    D[0, 0:max_consecutive_steps] = np.cumsum(C[0, 0:max_consecutive_steps])
    D[0, max_consecutive_steps:c] = INFINITY
        
    if subseq > 0:
#         D[:, 0] = C[:, 0]        
        D[0:max_consecutive_steps, 0] = C[0:max_consecutive_steps, 0]
        D[max_consecutive_steps:r, 0] = INFINITY        
    else:
        D[0:max_consecutive_steps, 0] = np.cumsum(C[0:max_consecutive_steps, 0])
        D[max_consecutive_steps:r, 0] = INFINITY
#         D[:, 0] = np.cumsum(C[:, 0])
    
    print D    

    
    cdef unsigned int i,j
    cdef DTYPE_t dist_step_diag
    cdef DTYPE_t dist_down
    cdef DTYPE_t dist_right
    cdef unsigned int best_step
    
    for i in range(r-1):
        for j in range(c-1):
            
            dist_step_diag = D[i, j] + C[i+1, j+1] * weights_mul[0]
            dist_down = D[i, j+1] + C[i+1, j+1] * weights_mul[1]
            dist_right = D[i+1, j] + C[i+1, j+1] * weights_mul[2]
            
            best_step = argmin3(dist_step_diag, dist_down, dist_right)
            
            if best_step == 1:
                if steps_down[i,j+1] == max_consecutive_steps:
                    print i, j+1
                    print steps_down
                    D[i+1,j+1] = dist_step_diag
                else:
                    steps_down[i+1,j+1] = steps_down[i,j+1] + 1
                    D[i+1,j+1] = dist_down
                                
            elif best_step == 2:
                if steps_right[i+1,j] == max_consecutive_steps:                    
                    print i+1, j
                    print steps_right
                    D[i+1,j+1] = dist_step_diag
                else:
                    steps_right[i+1,j+1] = steps_right[i+1,j] + 1    
                    D[i+1,j+1] = dist_right                
                    
            else:                        
                D[i+1,j+1] = dist_step_diag
    
    return D
        

# cd C:\Users\Alexis\Source\TestPython2\AutomaticAudioTranscript 
# python setup.py build_ext --inplace
