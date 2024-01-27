import pandas as pd
import numpy as np
import os

# def reconstruction_accuracies_table(num_subs):
#     # df = pd.read_pickle("reconstruction_accuracies"+str(num_subs)+".pkl")
#     subject_list = np.loadtxt('subjectlists/subject_list_HCP_'+str(num_subs)+'.txt',dtype=str)

#     # contrasts = np.loadtxt('contrast_list.txt',dtype=str)
#     # contrasts = np.append(contrasts,'rest')
#     contrasts = ['SSBCAP']
#     # Define the possible values for each column
#     density_values = ['1e-05', '0.0001','0.001','0.005','0.01', '0.1']
#     e_local_values = ['1e-05', '1.0']
#     binarization_values = ["binary", "weighted"]
#     fwhms = ['0.0','2.0','4.0','6.0','8.0','10.0']
#     streamliness = ['5M','10M','20M']
#     basis_values = ["subject-specific","avg connectome basis", 
#                     "avg basis flag", "avg basis karcher opt", 
#                     "avg basis karcher Begelfor","avg basis procrustes","ind conn/surf avg"]

#     # Create a list of all possible combinations of values
#     combinations = []
#     for sub in subject_list:
#         for contrast in contrasts:
#             for streamlines in streamliness:
#                 for density in density_values:
#                     for e_local in e_local_values:
#                         for binarization in binarization_values:
#                             if binarization == "binary" and e_local != '1.0':
#                                 continue
#                             for fwhm in fwhms:
#                                 for basis in basis_values:
#                                     combinations.append([sub,streamlines,density, e_local, binarization, fwhm, basis,contrast,np.zeros(200),np.zeros(200)])
    
#             combinations.append([sub,'none','none', 'none', 'none', 'none', "ind. surface",contrast,np.zeros(200),np.zeros(200)])
#             combinations.append([sub,'none','none', 'none', 'none', 'none', "template surface",contrast,np.zeros(200),np.zeros(200)])
#             combinations.append([sub,'none','none', 'none', 'none', 'none', "A_local",contrast,np.zeros(200),np.zeros(200)])

#     # Create the dataframe
#     df_new = pd.DataFrame(combinations, columns=["subject","streamlines","density", "e_local", "binarization", "fwhm", "basis","contrast","accuracy","accuracy_parc"])

#     # reorder the columns
#     df_new = df_new[["subject","streamlines", "fwhm", "binarization", "density", "e_local", "basis","contrast","accuracy","accuracy_parc"]]

#     # find matching rows between df and df_new, add the non-matching rows to df

#     # df = pd.concat([df,df_new],ignore_index=True)
#     df = df_new

#     # Save the dataframe as a csv file
#     df.to_pickle("reconstruction_accuracies_SSBCAP"+str(num_subs)+".pkl")

def bases_table(num_subs):
    # Define the possible values for each column
    ind_avg = ['ind',str(num_subs)] #can be extended with 100,255, etc
    density_values = ['1e-05', '0.0001','0.001','0.005','0.01','0.05', '0.1']
    e_local_values = ['1e-05', '1.0']
    binarization_values = ["binary", "weighted"]
    fwhms = ['0.0','2.0','4.0','6.0','8.0','10.0']
    streamliness = ['5M','10M','20M']
    # post_smoothing_threshold = '0.1'

    basis_values = ["subject-specific", "avg connectome basis",
                     "avg basis flag", "avg basis karcher opt", 
                     "avg basis karcher Begelfor","avg basis procrustes","ind conn/surf avg"]

    # Create a list of all possible combinations of values
    combinations = []
    for num_sub in ind_avg:
        for streamlines in streamliness:
            for density in density_values:
                for e_local in e_local_values:
                    for binarization in binarization_values:
                        if binarization == "binary" and e_local !='1.0':
                            continue
                        for fwhm in fwhms:
                            if fwhm == '0.0':
                                connectome_file = '_unsmoothed_high_resolution_volumetric_probabilistic_track_endpoints_'+streamlines+'.tck_structural_connectivity.npz'
                            else:
                                connectome_file = '_smoothed_structural_connectome_'+streamlines+'_fwhm'+fwhm+'.npz'

                            for basis in basis_values:
                                if basis == "subject-specific":
                                    if num_sub!="ind":
                                        continue
                                    basis_file = '_structural_connectome_'+streamlines+'_fwhm'+fwhm+'_elocal'+str(e_local)+'_density'+str(density)+'_'+binarization+'_emodes.txt'
                                elif basis == "ind conn/surf avg":
                                    if num_sub!="ind":
                                        continue
                                    basis_file = '_surface_connectome_'+streamlines+'_fwhm'+fwhm+'_elocal'+str(e_local)+'_density'+str(density)+'_'+binarization+'_emodes.txt'
                            
                                else:
                                    if num_sub=="ind":
                                        continue
                                    if basis== "avg connectome basis":
                                        basis_file = 'avg_structural_connectome_'+streamlines+'_fwhm'+fwhm+'_elocal'+str(e_local)+'_density'+str(density)+'_'+binarization+'_emodes.txt'
                                        connectome_file = 'avg_structural_connectome_'+streamlines+'_fwhm'+fwhm+'_'+num_sub+'.npz'
                                    else:
                                        basis_file = 'avg_basis_structural_connectome_'+streamlines+'_fwhm'+fwhm+'_elocal'+str(e_local)+'_density'+str(density)+'_'+binarization+'_emodes.txt'
                                combinations.append([num_sub,streamlines,density, e_local, binarization, fwhm, basis,basis_file,connectome_file])

    combinations.append(['0','none','none', 'none', 'none', 'none', "template surface",'fsLR_32k_midthickness-lh_emode_200.txt',''])
    combinations.append(['0','none','none', 'none', 'none', 'none', "A_local",'A_local-lh_emode_200.txt',''])
    combinations.append(['ind','none','none', 'none', 'none', 'none', "ind. surface",'.L.midthickness_MSMAll.32k_fs_LR_emodes.txt',''])

    # Create the dataframe
    df = pd.DataFrame(combinations, columns=["num_subs","streamlines","density", "e_local", "binarization", "fwhm", "basis","basis_file","connectome_file"])

    # reorder the columns
    df = df[["num_subs","streamlines","fwhm", "binarization", "density", "e_local", "basis","basis_file","connectome_file"]]
    df.reset_index(drop=True, inplace=True)

    # Save the dataframe as a csv file
    df.to_csv('basis_experiments_table'+str(num_subs)+'.csv', index=False)


if __name__ == "__main__":
    bases_table(num_subs=100)
    
    # prompt are you sure you want to do this, it overwrite the previous table

    # reconstruction_accuracies_table(20)