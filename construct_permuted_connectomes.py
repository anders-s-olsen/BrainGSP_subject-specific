import numpy as np
import os
import scipy.sparse

subject_list=np.loadtxt('subjectlists/subject_list_HCP_100.txt',dtype=str) #made from Pang-255 because no connectome for three subjects
for subject in subject_list:
    print('Running connectome permutation for subject '+subject)
    npz_file = 'data/'+str(subject)+'/T1w/tractography/' + subject + '_unsmoothed_high_resolution_volumetric_probabilistic_track_endpoints_20M.tck_structural_connectivity.npz'

    high_resolution_connectome = scipy.sparse.load_npz(npz_file)
    high_resolution_connectome = high_resolution_connectome[:59412, :59412]
    perm = np.random.permutation(59412)
    high_resolution_connectome_perm = high_resolution_connectome[perm,:]
    high_resolution_connectome_perm = high_resolution_connectome_perm[:,perm]

    scipy.sparse.save_npz('data/'+str(subject)+'/T1w/tractography/' + subject + '_unsmoothed_high_resolution_volumetric_probabilistic_track_endpoints_20M.tck_structural_connectivity_permuted.npz', high_resolution_connectome_perm)