import numpy as np
import os
import pandas as pd
import scipy
import argparse
import matplotlib.pyplot as plt
from helper_functions import parcellate_data_nd, compute_normalized_laplacian,GrDist
from avg_basis_functions import karcher_mean_Begelfor2006, karcher_mean_opt, flag_mean, procrustes, sort_total_variation
from connectome_eigenmodes import construct_A_local, threshold_edges_density

def plot_avg_bases(U):
# parcellate the results...
    parc = np.loadtxt('BrainEigenmodes/data/parcellations/fsLR_32k_Glasser360-lh.txt',dtype=int)
    mask = np.loadtxt('BrainEigenmodes/data/template_surfaces_volumes/fsLR_32k_cortex-lh_mask.txt',dtype=bool)
    parc = parc[mask]
    for i in len(U):
        U[i] = parcellate_data_nd(parc,U[i].T).T
    # plot them next to each other using barh
    fix,axs = plt.subplots(5,5,figsize=(10,10))
    for i1,i in enumerate([0,49,99,149,199]):
    # for i1,i in enumerate([0,1,2,3,4]):
        axs[0,i1].barh(np.arange(180),U[0][:,i])
        axs[0,i1].set_title('Flag mean, mode '+str(i))
        axs[1,i1].barh(np.arange(180),U[1][:,i])
        axs[1,i1].set_title('Procrustes, mode '+str(i))
        axs[2,i1].barh(np.arange(180),U[2][:,i])
        axs[2,i1].set_title('Karcher mean, mode '+str(i))
        axs[3,i1].barh(np.arange(180),U[3][:,i])
        axs[3,i1].set_title('Karcher mean opt, mode '+str(i))
        axs[4,i1].barh(np.arange(180),U[4][:,i])
        axs[4,i1].set_title('Extrinsic mean, mode '+str(i))
    plt.tight_layout()
    plt.savefig('tmp.png')
    for i1,U1 in enumerate(U):
        for i2,U2 in enumerate(U):
            if i1>=i2:
                continue
            print('Correlation between '+str(i1)+' and '+str(i2)+': '+str((np.diag(U1.T@U2)**2).mean()))

def compute_avg_basis(subject_list,df_sub,df_save,A_local):
    num_modes = 200

    L_norm = [] #list of graphs
    U = np.zeros((len(subject_list),29696,num_modes))
    
    # Define path to save the average connectome
    for i,subject in enumerate(subject_list):
        txt_file = df_sub.basis_location.iloc[0]+subject+'/T1w/'+subject+df_sub.filename.iloc[0]
        npz_file = df_sub.basis_location.iloc[0]+subject+'/T1w/'+subject+df_sub.npz_file.iloc[0]
        
        if not os.path.exists(txt_file) or not os.path.exists(npz_file):
            print('Could not load connectome basis for subject ' + subject)
            return
        
        # load basis
        U[i] = np.loadtxt(txt_file)
        # load connectome4
        W=scipy.sparse.load_npz(npz_file)[:29696, :29696]
        if df_sub.density.iloc[0]!=0:
            W = threshold_edges_density(W, desired_density=float(df_sub.density.iloc[0]), weight_to_remove='weakest') 

        # binarize and combine with A_local
        A_combined = W + float(df_sub.e_local.iloc[0])*A_local
        if df_sub.binarization.iloc[0]=='binary':
            A_combined[A_combined>0] = 1
        L_norm.append(compute_normalized_laplacian(A_combined))
        
        print('Loaded connectome basis for subject ' + subject)

    U_all = []
    print('Computing average basis: flag mean')
    U_flag = flag_mean(U)
    U_all.append(sort_total_variation(U_flag,L_norm))
    # print('Computing average basis: extrinsic mean')
    # _,U_extr = scipy.sparse.linalg.eigsh(np.sum(U@U.swapaxes(-2,-1),axis=0),k=200,which='LM',return_eigenvectors=True)
    # U_all.append(sort_total_variation(U_extr,L_norm))
    print('Computing average basis: karcher mean opt')
    U_karc1 = karcher_mean_opt(U) #always only 15 iterations using pymanopt
    U_all.append(sort_total_variation(U_karc1,L_norm))
    print('Computing average basis: karcher mean Begelfor')
    U_karc2 = karcher_mean_Begelfor2006(U,start_pt=U[0],tol=1e-6,max_iter=100,check=False)
    U_all.append(sort_total_variation(U_karc2,L_norm))
    print('Computing average basis: procrustes')
    U_proc = procrustes(U,X_template=U[0],tol=1e-6,max_iter=100)
    U_all.append(sort_total_variation(U_proc,L_norm))

    # Save the average connectome as a txt file
    basis_values = ["avg basis flag", "avg basis karcher opt", "avg basis karcher Begelfor","avg basis procrustes"]#, "avg basis extr"
    print('Saving average basis')
    for u,U in enumerate(U_all):
        np.savetxt(df_save[df_save['basis']==basis_values[u]].basis_location.iloc[0]+df_save[df_save['basis']==basis_values[u]].filename.iloc[0], U)


def main(num_subs):
    # load experiments table
    df = pd.read_csv('basis_experiments_table.csv')
    df = df[df['num_subs'].isin([str(num_subs),'ind'])]

    # load subject list
    subject_list=np.loadtxt('subjectlists/subject_list_HCP_'+str(num_subs)+'.txt',dtype=str)

    # construct A_local
    surface_file = 'BrainEigenmodes/data/template_surfaces_volumes/fsLR_32k_midthickness-lh.vtk'
    mask_file = 'BrainEigenmodes/data/template_surfaces_volumes/fsLR_32k_cortex-lh_mask.txt'
    A_local = construct_A_local(surface_file,mask_file)

    # loop over density, binary, e_local, smooth
    for smooth in ['smoothed','unsmoothed']:
        for binary in ['binary','weighted']:
            for density in ['1e-05', '0.0001','0.0005', '0.001', '0.0']:
                for e_local in ['1e-05','1.0']:
                    if binary=='binary' and e_local=='1e-05':
                        continue
                    
                    print('Running connectome basis avg for density '+str(density)+', '+binary+', e_local '+str(e_local)+', '+smooth)
                    df2 = df.copy()
                    df2 = df2[df2['density']==density]
                    df2 = df2[df2['e_local']==e_local]
                    df2 = df2[df2['binarization']==binary]
                    df2 = df2[df2['smoothing']==smooth]

                    # keep only basis corresponding to avg basis for filenames etc
                    df_save = df2[df2['num_subs']==str(num_subs)]
                    df_save = df_save[df_save['basis'].isin(['avg basis flag','avg basis extr','avg basis karcher opt','avg basis karcher Begelfor','avg basis procrustes'])]

                    df_sub = df2[df2['num_subs']=='ind']
                    if os.path.exists(df_save[df_save['basis']=='avg basis procrustes'].basis_location.iloc[0]+df_save[df_save['basis']=='avg basis procrustes'].filename.iloc[0]):
                        print('Already computed avg basis for density '+str(density)+', binary '+binary+', e_local '+str(e_local)+', smooth '+smooth)
                        continue

                    compute_avg_basis(subject_list=subject_list,df_sub=df_sub,df_save=df_save,A_local=A_local)


if __name__ == '__main__':
    # add parser to get the number of subjects
    parser = argparse.ArgumentParser()
    # parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--num_subs', type=int, default=20)
    args = parser.parse_args()
    main(args.num_subs)