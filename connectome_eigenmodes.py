import numpy as np
import scipy
from lapy import TriaMesh
import os 
import argparse
import pandas as pd

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

def compute_density(W):
    # input W is a scipy sparse CSR matrix
    N = W.shape[0]
    W = scipy.sparse.triu(W,k=1) #even if it is not symmetric/zero diagonal, evaluate only the upper triangular part
    density_W = W.nnz*2/(N*(N-1))
    return density_W

def connectome_checks(W):
    # Check if the matrix is symmetric using relative norm
    if scipy.sparse.linalg.norm(W - W.T)>1e-10:
        raise ValueError('Matrix is not symmetric')

    # if diagonal is not zero, set to zero:
    if np.linalg.norm(W.diagonal())>1e-10:
        W.setdiag(0)
        W.eliminate_zeros()

    return W

def threshold_edges_density(W, desired_density, weight_to_remove='weakest'):
    """
    Thresholds the input sparse graph W to achieve the desired density.

    Parameters:
    W (scipy.sparse.csr_matrix): Input sparse symmetric graph.
    desired_density (float): Desired density of the output graph.
    weight_to_remove (str): Specifies whether to remove the weakest or strongest edges. Default is 'weakest'.

    Returns:
    scipy.sparse.csr_matrix: Thresholded sparse graph.
    """
    current_density = compute_density(W)

    if float(desired_density) > current_density:
        raise ValueError('Desired density is larger than the current density')

    # Compute the threshold for the matrix W
    threshold = np.percentile(W.data, 100*(1-(float(desired_density)/current_density)))

    # Apply the threshold to the upper triangular matrix W
    
    if weight_to_remove == 'weakest':
        keep_these_elements = W >= threshold
    elif weight_to_remove == 'strongest':
        keep_these_elements = W < threshold
    W = W.multiply(keep_these_elements)

    return W

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

def run_connectome_decomposition(A_local, connectome_filename,e_local=1,desired_density=0.1,binary=True,output_file=None):

    try:
        W = scipy.sparse.load_npz(connectome_filename)
    except:
        raise ValueError('Could not load connectome for' + connectome_filename)
    
    # extract only left hemisphere
    W = W[:29696, :29696]
    connectome_checks(W)

    if desired_density!='0.0':
        try:
            W = threshold_edges_density(W, desired_density=desired_density, weight_to_remove='weakest') 
        except:
            print('Could not threshold connectome for subject ' + connectome_filename)
            return

    # binarize and combine with A_local
    A_combined = W + float(e_local)*A_local
    if binary:
        A_combined[A_combined>0] = 1
    L_norm = compute_normalized_laplacian(A_combined)
    try:
        evals,emodes=compute_eigenmodes(L_norm, num_eigenmodes=200)
    except:
        print('Could not compute eigenmodes for subject ' + connectome_filename)
        return

    # Save the eigenvalues and eigenvectors to txt files
    print('Saving eigenmodes')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savetxt(output_file, emodes)
    np.savetxt(output_file[:-8]+'_evals.txt', evals)

    return compute_density(A_combined)


def main(A_local,num_subs=98,subject_specific = True,e_local=1,desired_density=0.1,binary=True,npz_file = None,output_file = None):
    

    if desired_density == '4xA_local':
        A_local_density = compute_density(A_local)
        desired_density = 4*A_local_density

    if subject_specific:
        # Load subject list
        subjectlist=np.loadtxt('subjectlists/subject_list_HCP_'+str(num_subs)+'.txt',dtype=str)
        for sub in subjectlist:
            output_file2 = 'data/'+sub+'/T1w/tractography_eigenmodes/'+sub+output_file
            if os.path.exists(output_file2):
                print('Found connectome decomposition for subject '+sub)
                continue
            filename = 'data/'+sub+'/T1w/tractography/'+sub+npz_file
            print('Running connectome decomposition for subject '+sub)
            run_connectome_decomposition(A_local=A_local, connectome_filename=filename,e_local=e_local,desired_density=desired_density,binary=binary,output_file=output_file2)
    else:
        output_file2 = 'results/avg_connectomes/'+output_file[:-4]+'_'+str(num_subs)+'.txt'
        if os.path.exists(output_file2):
            print('Found connectome decomposition for avg connectome')
            return
        filename = 'results/avg_connectomes/'+npz_file
        run_connectome_decomposition(A_local=A_local, connectome_filename=filename,e_local=e_local,desired_density=desired_density,binary=binary,output_file=output_file2)


if __name__ == '__main__':
    # load surface file and construct local neighborhood binary matrix
    surface_file = 'BrainEigenmodes/data/template_surfaces_volumes/fsLR_32k_midthickness-lh.vtk'
    mask_file = 'BrainEigenmodes/data/template_surfaces_volumes/fsLR_32k_cortex-lh_mask.txt'
    A_local = construct_A_local(surface_file,mask_file)
    npz_file = 'avg_structural_connectome_20M_fwhm0.0_100.npz'
    output_file = 'avg_structural_connectome_20M_fwhm0.0_100_emodes.txt'
    # npz_file = '100307_unsmoothed_high_resolution_volumetric_probabilistic_track_endpoints_20M.tck_structural_connectivity.npz'
    # output_file = '100307_structural_connectome_20M_fwhm0.0_emodes.txt'
    main(A_local=A_local,num_subs=100,subject_specific=False,e_local='1.0',desired_density='0.001',binary=False,npz_file=npz_file,output_file=output_file)
    # True
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--num_subs', type=int, default=100)
    # parser.add_argument('--reverse_order', type=bool, default=False)
    # args = parser.parse_args()
    # num_subs = args.num_subs
    # reverse_order = args.reverse_order

    # df = pd.read_csv('basis_experiments_table.csv')
    # df = df[df['basis'].isin(['avg connectome basis'])]#'subject-specific',
    # df = df[df['e_local'].isin(['1.0'])]
    # df = df[df['streamlines'].isin(['20M'])]#'5M','10M',




    # # reverse order of df
    # if reverse_order:
    #     df = df.iloc[::-1]
    #     print('Reversed order of df')

    # for index, row in df.iterrows():
    #     print('Running connectome decomposition for num_sub'+str(num_subs)+' fwhm '+row['fwhm']+' binarization '+row['binarization']+' density '+row['density']+' elocal '+row['e_local']+' basis '+row['basis'])
    #     main(A_local=A_local,num_subs=num_subs,subject_specific = row['basis']=='subject-specific',e_local=row['e_local'],desired_density=row['density'],binary=row['binarization']=='binary',npz_file=row['connectome_file'],output_file=row['basis_file'])
