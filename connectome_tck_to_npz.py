from Connectome_Spatial_Smoothing import CSS as css
import numpy as np
import scipy
import os

def tck_to_npz(subject, streamlines='5M'):
    if os.path.exists('data/'+str(subject)+'/T1w/tractography/' + subject + \
        '_unsmoothed_high_resolution_volumetric_probabilistic_track_endpoints_'\
            +streamlines+'.tck_structural_connectivity.npz'):
        print('Found unsmoothed connectome for subject ' + subject)
        return

    left_surface_file = 'data/'+str(subject)+'/T1w/fsaverage_LR32k/'+str(subject)+'.L.white_MSMAll.32k_fs_LR.surf.gii'
    right_surface_file = 'data/'+str(subject)+'/T1w/fsaverage_LR32k/'+str(subject)+'.R.white_MSMAll.32k_fs_LR.surf.gii'
    tractography_file = 'data/'+str(subject)+'/T1w/tractography/volumetric_probabilistic_tracks_'+streamlines+'.tck'
    tractography_endpoints_file = 'data/'+str(subject)+'/T1w/tractography/volumetric_probabilistic_track_endpoints_'+streamlines+'.tck'
    tractography_sift_file = 'data/'+str(subject)+'/T1w/tractography/volumetric_probabilistic_sift_weights_'+streamlines+'.txt'
    

    if not os.path.exists(tractography_file):
        print('Could not find tractography for subject ' + subject)
        return

    high_res_connectome = css.map_high_resolution_structural_connectivity(
        tractography_file, left_surface_file, right_surface_file, 
        warp_file=None,subcortex=False)

    # save the last variable as a scipy sparse matrix
    scipy.sparse.save_npz('data/'+str(subject)+'/T1w/tractography/' + subject + \
        '_unsmoothed_high_resolution_volumetric_probabilistic_track_endpoints_'\
            +streamlines+'.tck_structural_connectivity.npz',\
            high_res_connectome)
    os.remove(tractography_file)
    os.remove(tractography_endpoints_file)
    # os.remove(tractography_sift_file)

if __name__ == "__main__":
    # Load subject list
    subjectlist = np.loadtxt('subjectlists/subject_list_HCP_100.txt',dtype=str)
    # reverse order
    # subjectlist = subjectlist[::-1]
    streamliness = ['20M']
    for subject in subjectlist:
        print('Running connectome decomposition for subject '+subject)
        for streamlines in streamliness:
            tck_to_npz(subject, streamlines)