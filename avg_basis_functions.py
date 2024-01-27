import numpy as np
import scipy
from helper_functions import GrDist,GrDist2



def GrExp(X,H):
    """
    Exponential map on the Grassmann manifold
    """
    U,S,Vt = np.linalg.svd(H,full_matrices=False)
    return X@Vt.T@np.diag(np.cos(S)) + U@np.diag(np.sin(S))

def GrLog(X,Y):
    """
    Logarithmic map on the Grassmann manifold, for nd arrays
    """
    XtY_inv = np.linalg.inv(X.swapaxes(-2,-1)@Y)
    # temp = np.eye(m)@Y@XtY_inv - X@XtY@XtY_inv #from marrinan etc
    temp = Y@XtY_inv - X 
    U,S,Vt = np.linalg.svd(temp,full_matrices=False)
    if X.ndim ==2 and Y.ndim == 2:
        theta = np.diag(np.arctan(S))
    else:
        theta = np.zeros((Y.shape[0],Y.shape[2],Y.shape[2]))
        for i in range(Y.shape[0]):
            theta[i] = np.diag(np.arctan(S[i]))
    
    return U@theta@Vt

def karcher_mean_Begelfor2006(data,start_pt,tol,max_iter,check=False):
    
    err = 999999
    iters = 1

    nr,nc = start_pt.shape
    M = start_pt.T@start_pt
    II = np.eye(nc)
    if np.linalg.norm(M-II,'fro')>tol:
        print('Start point is not orthogonal')
        U,_,Vt = np.linalg.svd(start_pt,full_matrices=False)
        start_pt = U@Vt
    
    num_pts,_,ncz = data.shape

    if check:
        for i in range(num_pts):
            Y = data[i]
            M = Y.T@Y
            II = np.eye(ncz)
        
            if np.linalg.norm(M-II,'fro') >= tol:
                print('Data point '+str(i)+' is not orthogonal')
                U,_,Vt = np.linalg.svd(Y,full_matrices=False)
                data[i] = U@Vt

    mu = start_pt
    while err>tol and iters<max_iter:
        # sum_T = np.zeros((nr,nc))
        # for i in range(num_pts):
        #     sum_T += GrLog(mu,data[i])
        # sum_T = sum_T/num_pts
        sum_T = np.mean(GrLog(mu,data),axis=0)
        
        mu_new = GrExp(mu,sum_T)
        err = GrDist(mu_new,mu, mu.shape[1],'DGEO')

        mu = mu_new
        iters += 1
        print('Iteration '+str(iters)+'/'+str(max_iter)+', error = '+str(err))

    return mu

def karcher_mean_opt(data):
    import pymanopt
    man = pymanopt.manifolds.Grassmann(n=29696, p=200)
    X = pymanopt.optimizers.nelder_mead.compute_centroid(man,data)
    # scipy.optimize.minimize(fun=GrDist2,x0=start_pt.flatten(),args=(data,None,'DGEO'),tol=tol,options={'maxiter':max_iter,'disp':True})
    return X

def sort_zero_crossings(X):
    ZC = np.zeros(X.shape[1])
    for i,x in enumerate(X.T):
        numpos = np.sum(x>0)
        numneg = np.sum(x<0)
        ZC[i] = numpos*numneg

    # sort X according to ZC, low first
    
    idx = np.argsort(ZC)
    X = X[:,idx]

    return X
def total_variation(x,L):
    """
    Compute the total variation of graph signal x on the sym. normalized Laplacian L"""

    TV = x.T@L@x

    return TV
def sort_total_variation(U,L):

    TV = np.zeros((len(L),U.shape[-1]))
    for i,x in enumerate(L):
        for j,u in enumerate(U.T):
            TV[i,j] = total_variation(u,L[i])
    

    # sort X according to TV, low first
    idx = np.argsort(TV.mean(axis=0))
    U = U[:,idx]

    return U

def flag_mean(X,k=200):
    # reoder X from (N,P,Q) to (P,N*Q) where each Xn is stacked horisontally
    X = X.swapaxes(0,1)
    X = X.reshape(X.shape[0],-1)
    # compute flag mean
    U,_,_ = scipy.sparse.linalg.svds(X,k,which='LM',return_singular_vectors='u')
    return U

def procrustes(X,X_template,tol=1e-6,max_iter=100):
    n = []

    for j in range(max_iter):
        U,S,Vt = np.linalg.svd(X_template.T@X)
        R3 = Vt.swapaxes(-2,-1)@U.swapaxes(-2,-1)
        X = X@R3
        U_mean = np.mean(X,axis=0)
        # U_mean_orth,_,_ = np.linalg.svd(U_mean,full_matrices=False)
        # compare before and after
        n.append(np.linalg.norm(X_template-U_mean))
        print('Iteration '+str(j)+'/'+str(max_iter)+', error = '+str(n[-1]))
        if n[-1]<tol:
            print('Converged')
            break
        else:
            X_template = U_mean
    U_mean,_,_ = np.linalg.svd(U_mean,full_matrices=False)
    return U_mean