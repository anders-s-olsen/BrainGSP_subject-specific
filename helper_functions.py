import numpy as np
# import numba as nb

def calc_lm_weights(X,y):
    # Fit the linear model Y=XB to the task data.
    # note in this case y will always be a vector. X can be ND where the last dimension is the number of vertices
    # B = np.linalg.lstsq(X, Y, rcond=None)[0]
    B = np.linalg.inv(X.T@X)@X.T@y
    return B
def calc_corr(a,b):
    # Calculate the correlation between vectors a and b.
    # only for vectors!!
    corr = np.corrcoef(a, b,rowvar=0)[0, 1]
    return corr
def calc_corr_nd(A,B):
    """
    Calculate the correlation between two ND arrays, only corr of the LAST dimension is calculated
    The two arrays should be completely compatible, e.g., sizes (...,N) and (...,N)"""
    # if A.shape!=B.shape:
    #     raise ValueError('A and B must have the same shape')
    cov = np.sum((A-np.mean(A,axis=-1)[...,np.newaxis])*(B-np.mean(B,axis=-1)[...,np.newaxis]),axis=-1)/(A.shape[-1]-1)
    corr = cov/np.sqrt(np.var(A,axis=-1,ddof=1)*np.var(B,axis=-1,ddof=1))
    return corr
def calc_acc_nd(A,B):
    """
    Calculate the accuracy between two ND arrays, only acc of the LAST dimension is calculated
    The two arrays should be completely compatible, e.g., sizes (...,N) and (...,N)"""
    if A.shape!=B.shape:
        raise ValueError('A and B must have the same shape')
    A = A/np.linalg.norm(A,axis=-2,keepdims=True)
    B = B/np.linalg.norm(B,axis=-2,keepdims=True)
    
    acc = np.linalg.norm(A-B,'fro')
    return acc

def parcellate_data_nd(parc,data):
    # parc must be a vector of parcel indices, data can be ND where the last dimension is the number of vertices
    if data.ndim==1:
        data = data[np.newaxis,:]
    if parc.shape[0]!=data.shape[-1]:
        raise ValueError('parc and data must have same number of vertices')
    # Extract the unique parcels
    parcels = np.unique(parc[parc>0])
    # Initialize the parcellated data
    parcel_data_shape = list(data.shape[:-1])+[len(parcels)]
    parcellated_data = np.zeros(parcel_data_shape,dtype=data.dtype)
    # Loop over the parcels
    for parcel_ind, parcel in enumerate(parcels):
        # Find the vertices in the parcel
        ind_parcel = parc == parcel
        # Average the data in the parcel
        parcellated_data[...,parcel_ind] = np.mean(data[...,ind_parcel],axis=-1)
    return parcellated_data 

def calc_lm_weights_nd(X,y):
    # Fit the linear model Y=XB to the task data.
    # note in this case y will always be a vector. X can be ND where the last dimension is the number of vertices
    # X should ALWAYS be a matrix in the last two dimensions. So if a vector, ensure that the last dimension is 1
    # y should be a vector and thus the last dimension should be 1 (in the code below, we add a new axis to y to ensure this)
    # otherwise, the extra dimensions of X and y should be compatible, e.g., X is (B,1,P,1) and y is (1,N,P) extended in the code to (1,N,P,1)
    Xt = np.transpose(X,(0,1,3,2))
    XtX = Xt@X
    # if XtX.shape[-1]==1:
    #     if np.all(np.isclose(XtX,1)):
    #         X_hat = X.swapaxes(-2,-1)
    #     else:
    #         X_hat = np.linalg.inv(XtX)@X.swapaxes(-2,-1)
    # else:
    if np.all(np.isclose(np.linalg.det(XtX),1)): #don't calculate the inverse if it's not necessary
        X_hat = Xt
    else:
        X_hat = np.linalg.inv(XtX)@Xt
    b = X_hat@y[...,np.newaxis]
    return b[...,0]

def data_reconstruction(data,bases,num_modes_recon):
    B,P1,M = bases.shape
    N,P2 = data.shape
    # loop over modes, extracting the basis and calculating the weights of the linear model 
    recon = np.zeros((num_modes_recon,B,N,P1))
    for mode in range(num_modes_recon):
        # time1 = time.time()
        weights = calc_lm_weights_nd(bases[:,np.newaxis,:,:mode+1],data[np.newaxis,...])
        # time2 = time.time()
        # print('Completed weights in '+str(time2-time1)+' seconds')
        recon[mode] = (bases[:,np.newaxis,:,:mode+1]@weights[:,:,np.newaxis].swapaxes(-2,-1))[:,:,:,0]
        # time3 = time.time()
        # print('Completed reconstruction in '+str(time3-time2)+' seconds')
        print('Completed mode '+str(mode+1)+' of '+str(num_modes_recon))
    return recon

# @nb.njit(nb.float32[:](nb.float32[:,:],nb.float32[:,:]))
# def calc_corr_nd_numba(A,B):
#     """
#     Calculate the correlation between two ND arrays, only corr of the LAST dimension is calculated
#     The two arrays should be completely compatible, e.g., sizes (...,N) and (...,N)"""
#     if A.shape!=B.shape:
#         raise ValueError('A and B must have the same shape')
#     corr = np.zeros(A.shape[0],dtype=np.float32)
#     for i in range(A.shape[0]):
#         corr[i] = np.corrcoef(A[i], B[i])[0, 1]
#         # cov = np.sum((A[i]-np.mean(A[i]))*(B[i]-np.mean(B[i])))/(A.shape[-1]-1)
#         # corr[i] = cov/np.sqrt(np.var(A[i],ddof=1)*np.var(B[i],ddof=1))
#     return corr

# @nb.njit(nb.float32[:,:](nb.int64[:],nb.float32[:,:]))
# def parcellate_data_nd_numba(parc,data):
#     N,P = data.shape
#     parcels = np.unique(parc[parc>0])
#     parcellated_data = np.zeros((len(parcels),N),dtype=np.float32)
#     for parcel_ind, parcel in enumerate(parcels):
#         ind_parcel = parc == parcel
#         parcellated_data[parcel_ind] = np.mean(data[:,ind_parcel])
#     return parcellated_data.T

# @nb.njit(nb.types.UniTuple(nb.float32[:,:,:],2)\
#          (nb.float32[:,:],nb.float32[:,:,:],nb.int64,nb.int64[:]),parallel=True)
# def data_reconstruction_numba(data,bases,num_modes_recon,parc):
#     B,P1,M = bases.shape
#     N,P2 = data.shape
#     data_parc = parcellate_data_nd_numba(parc,data)
#     recon_accuracy = np.zeros((num_modes_recon,B,N),dtype=np.float32)
#     recon_accuracy_parc = np.zeros((num_modes_recon,B,N),dtype=np.float32)
#     for mode in nb.prange(num_modes_recon):
#         for b,basis in enumerate(bases):
#             weights = basis[:,:mode+1].T@data.T
#             recon = (basis[:,:mode+1]@weights).T
            
#             recon_parc = parcellate_data_nd_numba(parc,recon)
#             recon_accuracy[mode,b] =calc_corr_nd_numba(data,recon)
#             recon_accuracy_parc[mode,b] =calc_corr_nd_numba(data_parc,recon_parc)
#         print('Completed mode '+str(mode+1)+' of '+str(num_modes_recon))
#     return recon_accuracy,recon_accuracy_parc

# @nb.njit(nb.float32[:](nb.float32[:,:]))
# def construct_correlation_matrix_upper_triangle_numba(data):
#     # Calculate the correlation matrix for a (NxP) dataset.
#     conn_matrix_tmp = np.corrcoef(data,rowvar=False)
#     rest_data_parc_conn_matrix = conn_matrix_tmp[np.triu_indices(data.shape[1], k=1)]
#     return rest_data_parc_conn_matrix

# @nb.njit(nb.types.UniTuple(nb.float32[:,:,:],2)\
#          (nb.float32[:,:],nb.float32[:,:,:],nb.int64,nb.int64[:]),parallel=True)
# def data_reconstruction_numba2(data,bases,num_modes_recon,parc):
#     B,P1,M = bases.shape
#     N,P2 = data.shape
#     data_parc = parcellate_data_nd_numba(parc,data)
#     conn_matrix_data = construct_correlation_matrix_upper_triangle_numba(data_parc)
#     recon_accuracy = np.zeros((num_modes_recon,B,N),dtype=np.float32)
#     recon_accuracy_parc = np.zeros((num_modes_recon,B,N),dtype=np.float32)
#     for mode in nb.prange(num_modes_recon):
#         for b,basis in enumerate(bases):
#             weights = basis[:,:mode+1].T@data.T #MxN
#             recon = (basis[:,:mode+1]@weights).T #NxP1
#             recon_parc = parcellate_data_nd_numba(parc,recon)
#             conn_matrix_recon = construct_correlation_matrix_upper_triangle_numba(recon_parc)
#             recon_accuracy_parc[mode,b] =calc_corr_nd_numba(conn_matrix_data,conn_matrix_recon)
#         print('Completed mode '+str(mode+1)+' of '+str(num_modes_recon))
#     return recon_accuracy,recon_accuracy_parc


def GrDist(X,Y,p=None,metric='DGEO'):
    """
    Distance between two points on the Grassmann manifold. X and Y must be matrices of same size
    p is the number of angles to be used for the distance calculation
    'dist_type' is a string that specifies which distance measure
    you would like to use.  Options are:
        'DGEO'  for the Geodesic distance
        'DPF'   for the Projection F-norm
        'DC'    for the Chordal distance
        'DCF'   for the Chordal Frobenius
        'DFS'   for the Fubini-Studi
        'MAX'   for the maximum angle
        'MIN'   for the minimum angle
        'ALL'   for all the angles
        'EUC'   for the Frobenius norm of P-Q
        Default value is 'DGEO'
    """
    if p is None:
        p = X.shape[1]
    if X.shape[0]!=Y.shape[0]:
        raise ValueError('X and Y must have the same number of rows')

    q = np.min([X.shape[1],Y.shape[1]])
    p = np.min([p,q])


    if metric=='EUC':
        return np.linalg.norm(X-Y,'fro')
    else:
        S = np.linalg.svd(X.T@Y,compute_uv=False,full_matrices=False)
        principal_angles = np.arccos(np.clip(S,-1,1))
        # if np.any(S<1e-8):
        #     principal_angles[S<1e-8] = np.sqrt(2*(1-S[S<1e-8]))

    if metric=='DGEO': #geodesic distance
        return np.linalg.norm(principal_angles[:p])
    elif metric=='DPF': #projection F-norm / chordal distance
        return np.linalg.norm(np.sin(principal_angles[:p]))
    elif metric=='DCF': #chordal Frobenius
        return np.linalg.norm(2*np.sin(principal_angles[:p]/2))
    elif metric=='DFS': #Fubini-Studi
        return np.arccos(np.prod(S[:p]))
    elif metric=='MAX':
        return np.max(principal_angles[:p]) 
    elif metric=='MIN':
        return np.min(principal_angles[:p])
    elif metric=='ALL':
        return principal_angles[:p]
    else:
        raise ValueError('Unknown metric')
    
def GrDist2(x,Y,p=None,metric='DGEO'):
    X = x.reshape(29696,200)
    if p is None:
        p = X.shape[1]

    q = np.min([X.shape[1],Y.shape[1]])
    p = np.min([p,q])


    if metric=='EUC':
        return np.linalg.norm(X-Y,'fro')
    else:
        S = np.linalg.svd(X.T@Y,compute_uv=False,full_matrices=False)
        principal_angles = np.arccos(np.clip(S,-1,1))
        # if np.any(S<1e-8):
        #     principal_angles[S<1e-8] = np.sqrt(2*(1-S[S<1e-8]))

    if metric=='DGEO': #geodesic distance
        return np.linalg.norm(principal_angles[:p])
    elif metric=='DPF': #projection F-norm / chordal distance
        return np.linalg.norm(np.sin(principal_angles[:p]))
    elif metric=='DCF': #chordal Frobenius
        return np.linalg.norm(2*np.sin(principal_angles[:p]/2))
    elif metric=='DFS': #Fubini-Studi
        return np.arccos(np.prod(S[:p]))
    elif metric=='MAX':
        return np.max(principal_angles[:p]) 
    elif metric=='MIN':
        return np.min(principal_angles[:p])
    elif metric=='ALL':
        return principal_angles[:p]
    else:
        raise ValueError('Unknown metric')