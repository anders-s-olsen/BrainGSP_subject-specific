import numpy as np
import scipy
from lapy import TriaMesh


def construct_A_local(surface_file,mask_file = None):
    tria = TriaMesh.read_vtk(surface_file)
    num_vertices = tria.v.shape[0]
    # do the above more efficiently
    i = np.concatenate([tria.t[:,0], tria.t[:,0], tria.t[:,1]])
    j = np.concatenate([tria.t[:,1], tria.t[:,2], tria.t[:,2]])
    data = np.ones_like(i)
    surface_connectome = scipy.sparse.csr_matrix((data, (i, j)), shape=(num_vertices, num_vertices))
    
    surface_connectome = surface_connectome + surface_connectome.T
    surface_connectome[surface_connectome>0] = 1

    if mask_file is not None:
        mask = np.loadtxt(mask_file, dtype=bool)
        surface_connectome = surface_connectome[mask, :][:, mask]
    return surface_connectome

def compute_normalized_laplacian(W):
    """
    W is a scipy.sparse csr array. 
    Note, the scipy.sparse representation handles the diagonal differently than the numpy array representation.
    This means that zeros in the diagonal of D are not ignored, but are represented as zeros in the sparse matrix, 
    similarly to how one would normally treat zeros in the diagonal of D.
    """
    # print('Computing normalized Laplacian')
    # compute the normalized laplacian
    d = np.array(W.sum(axis=1))
    D = scipy.sparse.diags(d[:,0],format='csr')
    L = D - W
    D_half_inv = scipy.sparse.diags(1/np.sqrt(d[:,0]),format='csr') #pseudoinverse of D_half
    L_norm = D_half_inv @ (L @ D_half_inv)
    return L_norm

def compute_eigenmodes(L_norm, num_eigenmodes=200):
    # Compute the first 200 eigenmodes of the Laplacian
    print('Computing eigenmodes')
    evals, emodes = scipy.sparse.linalg.eigsh(L_norm, k=num_eigenmodes, which='SM')

    return evals,emodes

if __name__=='__main__':

    # load surface file and construct local neighborhood binary matrix
    surface_file = 'BrainEigenmodes/data/template_surfaces_volumes/fsLR_32k_midthickness-lh.vtk'
    mask_file = 'BrainEigenmodes/data/template_surfaces_volumes/fsLR_32k_cortex-lh_mask.txt'
    A_local = construct_A_local(surface_file,mask_file)
    L_local = compute_normalized_laplacian(A_local)
    evals,emodes = compute_eigenmodes(L_local, num_eigenmodes=200)
    np.savetxt('results/A_local-lh_emode_200.txt', emodes)
    