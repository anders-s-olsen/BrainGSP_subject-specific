import numpy as np
import os
import scipy
from Connectome_Spatial_Smoothing import CSS as css
import argparse

def smooth_connectome_and_save(npz_path,subject, fwhm, post_smoothing_threshold,streamlines='5M'):
    try:
        npz_file = os.path.join(npz_path, subject + '/T1w/tractography/' \
                        + subject + '_unsmoothed_high_resolution_volumetric_probabilistic_track_endpoints_'+streamlines+'.tck_structural_connectivity.npz')
        high_resolution_connectome = scipy.sparse.load_npz(npz_file)
        high_resolution_connectome = high_resolution_connectome[:59412, :59412]
    except:
        print('Could not load unsmoothed connectome for subject ' + subject)
        return
    left_surface_file = 'data/'+subject+'/T1w/fsaverage_LR32k/'+subject+'.L.white_MSMAll.32k_fs_LR.surf.gii'
    right_surface_file = 'data/'+subject+'/T1w/fsaverage_LR32k/'+subject+'.R.white_MSMAll.32k_fs_LR.surf.gii'

    smoothing_kernel = css.compute_smoothing_kernel(left_surface_file, right_surface_file, fwhm=float(fwhm), epsilon=0.01, subcortex=False)
    smoothed_high_resolution_connectome = css.smooth_high_resolution_connectome(high_resolution_connectome, smoothing_kernel)
    if float(post_smoothing_threshold)>0:
        thresholded_smoothed_high_resolution_connectome = \
            smoothed_high_resolution_connectome.multiply(smoothed_high_resolution_connectome > float(post_smoothing_threshold))
        
        # save the last variable as a scipy sparse matrix
        scipy.sparse.save_npz(npz_path + subject + '/T1w/tractography/' + subject + \
            '_smoothed_structural_connectome_'+streamlines+'_fwhm'+fwhm+'.npz',\
                thresholded_smoothed_high_resolution_connectome)
    else:
        # save the last variable as a scipy sparse matrix
        scipy.sparse.save_npz(npz_path + subject + '/T1w/' + subject + \
            '_smoothed_structural_connectome_'+streamlines+'_fwhm'+fwhm+'.npz',\
                smoothed_high_resolution_connectome)
    
    

def main(num_subs):

    # Load subject list
    subject_list=np.loadtxt('subjectlists/subject_list_HCP_'+str(num_subs)+'.txt',dtype=str) #made from Pang-255 because no connectome for three subjects
    subject_list = subject_list[::-1]
    # Define path to npz files
    npz_path = 'data/'

    fwhms = ['2.0','4.0','6.0','8.0','10.0']
    post_smoothing_threshold = '0.1'
    streamliness = ['20M']#['5M','10M','20M']

    # Define path to save the average connectome
    for subject in subject_list:
        for streamlines in streamliness:
            for fwhm in fwhms:
                npz_file = os.path.join(npz_path, subject + '/T1w/tractography/' \
                                        + subject + '_smoothed_structural_connectome_'+streamlines+'_fwhm'\
                                            +fwhm+'.npz')
                if os.path.exists(npz_file):
                    print('Found smoothed connectome for subject ' + subject, fwhm  + ' streamlines ' + streamlines)
                    continue
                else:
                    print('Could not load smoothed connectome for subject ' + subject+', will try to compute it')
                    smooth_connectome_and_save(npz_path,subject, fwhm, post_smoothing_threshold,streamlines=streamlines)

                print('Done smoothing connectome for subject ' + subject+' fwhm'+str(fwhm))
            
if __name__ == '__main__':
    # add parser to get the number of subjects
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_subs', type=int, default=100)
    args = parser.parse_args()
    main(args.num_subs)