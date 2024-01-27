import numpy as np
import os 
import argparse
import pandas as pd
import scipy
from avg_basis_functions import flag_mean, sort_total_variation,karcher_mean_Begelfor2006
from connectome_eigenmodes import construct_A_local, threshold_edges_density
from helper_functions import compute_normalized_laplacian

def main(A_local,row,num_subs):
    subjectlist=np.loadtxt('subjectlists/subject_list_HCP_'+str(num_subs)+'.txt',dtype=str)
    mask = np.loadtxt('BrainEigenmodes/data/template_surfaces_volumes/fsLR_32k_cortex-lh_mask.txt',dtype='bool')
    for sub in subjectlist:
        output_file2 = 'data/'+str(sub)+'/T1w/'+str(sub)+row.filename
        if os.path.exists(output_file2):
            print('Found connectome decomposition for subject '+sub)
            continue
        X = np.zeros((2,29696,200))
        connectome_basis = np.loadtxt('data/'+str(sub)+'/T1w/'+str(sub)+'_structural'+row.filename[8:])
        geometry_basis = np.loadtxt('data/'+str(sub)+'/T1w/fsaverage_LR32k/'+str(sub)+'.L.midthickness_MSMAll.32k_fs_LR_emodes.txt')
        X[0] = connectome_basis
        X[1] = geometry_basis[mask]

        W = scipy.sparse.load_npz('data/'+str(sub)+'/T1w/'+str(sub)+row.npz_file)[:29696, :29696]
        if row.density!='0.0':
            W = threshold_edges_density(W, desired_density=float(row.density), weight_to_remove='weakest') 

        # binarize and combine with A_local
        A_combined = W + float(row.e_local)*A_local
        if row.binarization=='binary':
            A_combined[A_combined>0] = 1
        L_norm = compute_normalized_laplacian(A_combined)
        print('Running connectome geom/conn avg for subject '+sub)
        U_flag = flag_mean(X,k=200)
        U_flag_sorted = sort_total_variation(U_flag,[L_norm])
        np.savetxt(output_file2, U_flag_sorted)
        U_karc = karcher_mean_Begelfor2006(X,start_pt=X[0],tol=1e-06,max_iter=1000)
        U_karc_sorted = sort_total_variation(U_karc,[L_norm])
        np.savetxt(output_file2, U_karc_sorted)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_subs', type=int, default=20)
    parser.add_argument('--reverse_order', type=bool, default=False)
    args = parser.parse_args()
    num_subs = args.num_subs
    reverse_order = args.reverse_order

    df = pd.read_csv('basis_experiments_table.csv')
    df = df[df['basis'].isin(['ind conn/surf avg'])]
    # load surface file and construct local neighborhood binary matrix
    surface_file = 'BrainEigenmodes/data/template_surfaces_volumes/fsLR_32k_midthickness-lh.vtk'
    mask_file = 'BrainEigenmodes/data/template_surfaces_volumes/fsLR_32k_cortex-lh_mask.txt'
    # print('Constructing local neighborhood matrix')
    A_local = construct_A_local(surface_file,mask_file)

    # reverse order of df
    if reverse_order:
        df = df.iloc[::-1]

    for index, row in df.iterrows():
        if row.e_local=='1e-05':
            continue
        print('Running connectome decomposition for num_sub'+str(num_subs)+' fwhm'+row['fwhm']+' bin'+row['binarization']+' den'+row['density']+' eloc'+row['e_local']+' bas'+row['basis'])
        main(A_local,row,num_subs=num_subs)

    # main(load_smoothed_connectome = True,num_subs=98,subject_specific = True)
    #170GB FOR BINARY, 682GB for weighted, 8,5 for avg