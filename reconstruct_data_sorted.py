import numpy as np
import nibabel as nib
import argparse
import os
import pandas as pd
import numba as nb
from helper_functions import calc_corr_nd

def filter_df(df,exp_settings):
    df = df[df.basis.isin(exp_settings['basis'])]
    df = df[df.e_local.isin(exp_settings['e_local'])]
    df = df[df.density.isin(exp_settings['density'])]
    df = df[df.binarization.isin(exp_settings['binarization'])]
    df = df[df.fwhm.isin(exp_settings['fwhm'])]
    df = df[df.streamlines.isin(exp_settings['streamlines'])]
    return df

def load_bases(df,subject=None,mask=None,num_subs=None):
    """
    mask is a (P,) bool-array where P is number of vertices and each element is True if that vertex is in the mask
    """
    
    B = np.zeros((df.shape[0],mask.sum(),200),dtype=np.float64)
    locs = df.basis_file.tolist()
    for i1,loc in enumerate(locs):
        print('Loading basis '+str(i1+1)+' of '+str(df.shape[0])+'...')
        if subject is not None:
            filename = 'data/'+subject+'/T1w/tractography_eigenmodes/'+subject+loc
            if not os.path.exists(filename):
                filename = 'data/'+subject+'/T1w/fsaverage_LR32k/'+subject+loc
            if not os.path.exists(filename):
                raise ValueError('Could not find basis file for subject '+subject+loc)
        else:
            filename = 'results/avg_connectomes/'+loc[:-4]+'_'+str(num_subs)+'.txt'
            if not os.path.exists(filename):
                filename = 'results/'+loc
        tmp = np.loadtxt(filename,dtype=np.float64)
        if tmp.shape[0]!=mask.sum():
            tmp = tmp[mask]
        B[i1] = tmp
    return B

def check_exists(df_acc,df_basis,subject=None):
    # find corresponding row in df_acc
    if subject is None:
        df_acc_row = np.where([(df_acc.fwhm==df_basis.fwhm) & 
                            (df_acc.binarization==df_basis.binarization) & 
                            (df_acc.density==df_basis.density) & 
                            (df_acc.e_local==df_basis.e_local) & 
                            (df_acc.basis==df_basis.basis) & 
                            (df_acc.streamlines==df_basis.streamlines)])[1]
    else:
        df_acc_row = np.where([(df_acc.fwhm==df_basis.fwhm) & 
                            (df_acc.binarization==df_basis.binarization) & 
                            (df_acc.density==df_basis.density) & 
                            (df_acc.e_local==df_basis.e_local) & 
                            (df_acc.basis==df_basis.basis) & 
                            (df_acc.streamlines==df_basis.streamlines) &
                            (df_acc.subject==subject)])[1]
    return df_acc_row

def load_task_data(contrasts,subject,mask=None):
    print('Loading task data for subject '+subject+'...')
    task_data = np.zeros((len(contrasts),mask.sum()),dtype=np.float64)
    for j,contrast in enumerate(contrasts):
        tmp = np.loadtxt('data/' + subject + '/task_txtfiles/' + contrast + '.txt',dtype=np.float64)
        task_data[j] = tmp[mask]
    print('Task data loaded')
    return task_data

def load_SSBCAP_data(subject,mask=None):
    num_regions = 400
    print('Loading SSBCAP data for subject '+subject+'...')
    SSBCAP_data = np.zeros((num_regions,mask.sum()),dtype=np.float64)
    for j in range(num_regions):
        tmp = np.loadtxt('data/' + subject + '/T1w/Results/SBBCAPs/'+
                         'lh.rfMRI_REST1_LR.res1250.spaceT1w.detrend1_regMov1_zscore1.SSBCAPs_schaefer400yeo7_res1250_spaceT1w_reg'+
                            str(j+1)+'.txt',dtype=np.float64)
        SSBCAP_data[j] = tmp[mask]
    print('SSBCAP data loaded')
    return SSBCAP_data

def load_rest_data(subject,mask=None):
    print('Loading rest data for subject '+subject+'...')
    restdata_loc = '/media/miplab-nas2/Data2/Enrico_Projects/HCP_MIPLAB/HCP_data/'
    rest_data = nib.load(restdata_loc+subject+'/MNINonLinear/Results/rfMRI_REST1_LR/'+\
                        'rfMRI_REST1_LR_Atlas_MSMAll.dtseries.nii').get_fdata() #missing the 'clean' and 'hp2000 option
    rest_data = rest_data[:,:29696].astype(np.float64)
    print('Rest data loaded')
    return rest_data

def load_noise_data():
    print('Loading noise data...')
    noise_data = np.random.normal(size=(1,29696))
    print('Noise data loaded')
    return noise_data

# @nb.njit(nb.float64[:,:,:](nb.float64[:,::1],nb.float64[:,:,::1],nb.float64[:,:,:],nb.int64[:,:,:]),parallel=True)
def data_reconstruction_old(data,bases,recon_accuracy,sort_idx):
    B,M,P1 = bases.shape
    N,P2 = data.shape
    print('Starting reconstruction...')
    
    if np.any(np.isnan(data)):
        idx = np.where(np.isnan(data))
        for i in range(len(idx[0])):
            data[idx[0][i],idx[1][i]] = 0

    for mode in range(M):
        # weight = bases[:,:mode+1]@data.T
        weight = np.linalg.inv(bases[:,:mode+1]@np.transpose(bases[:,:mode+1],(0,2,1)))@bases[:,:mode+1]@data.T
        recon = np.transpose(np.transpose(bases[:,:mode+1],(0,2,1))@weight,(0,2,1))
        recon_accuracy[mode] = calc_corr_nd(data,recon)
        print('Completed mode '+str(mode+1)+' of '+str(M))

    return recon_accuracy

def sort_basis(data,bases):
    B,M,P1 = bases.shape
    N,P2 = data.shape

    # weight = bases@data.T
    weight = np.linalg.inv(bases@np.transpose(bases,(0,-1,-2)))@bases@data.T

    sort_idx = np.zeros_like(weight,dtype=int)
    for b in range(B):
        for n in range(N):
            sort_idx[b,:,n] = np.argsort(np.abs(weight[b,:,n]))[::-1]

    return sort_idx

def run_reconstruction(bases,num_modes_recon,subject,mask,parc,task_or_rest='task'):

    if task_or_rest=='rest':
        data = load_rest_data(subject,mask=mask)
    elif task_or_rest=='SSBCAP':
        data = load_SSBCAP_data(subject,mask=mask)
    elif task_or_rest=='noise':
        data = load_noise_data()
    else:
        data = load_task_data(contrasts=exp_settings['contrasts'],subject=subject,mask=mask)
    N = data.shape[0]
    B = bases.shape[0]
    bases = np.ascontiguousarray(np.transpose(bases,(0,2,1)))

    sort_idx = sort_basis(data=data,bases=bases)
    
    recon_accuracy = np.zeros((num_modes_recon,B,N),dtype=np.float64)
    recon_accuracy = data_reconstruction_old(data=data,bases=bases,recon_accuracy=recon_accuracy,sort_idx=sort_idx)

    if task_or_rest=='SSBCAP':
        recon_accuracy = np.mean(recon_accuracy,axis=2)[:,:,np.newaxis]

    return recon_accuracy


def main(num_subs,exp_settings):

    if exp_settings['contrasts'] not in ['47task','rest','SSBCAP','noise']:
        raise ValueError('exp_settings[\'contrast\'] must be either \'47task\', \'rest\', or \'SSBCAP\'')

    if exp_settings['contrasts']=='47task':
        exp_settings['contrasts'] = np.loadtxt('contrast_list.txt',dtype=str)
    else:
        exp_settings['contrasts'] = [exp_settings['contrasts']]
    
    num_subs = str(num_subs)

    # Load the subject list text file.
    subject_list = np.loadtxt('subjectlists/subject_list_HCP_'+num_subs+'.txt', dtype=str)

    mask = np.loadtxt('BrainEigenmodes/data/template_surfaces_volumes/fsLR_32k_cortex-lh_mask.txt',dtype=bool)
    parc = np.loadtxt('BrainEigenmodes/data/parcellations/fsLR_32k_Glasser360-lh.txt',dtype=int)
    parc = parc[mask]

    try:
        if len(exp_settings['contrasts'])==47:
            df_acc = pd.read_pickle('reconstruction_accuracies_sorted'+num_subs+'.pkl')
        else:
            df_acc = pd.read_pickle('reconstruction_accuracies_sorted_'+exp_settings['contrasts'][0]+num_subs+'.pkl')
    except:
        df_acc = pd.DataFrame(columns=["subject","streamlines", "density", "e_local", "binarization", "fwhm", "basis","contrast","accuracy","accuracy_parc"])
    # df_acc_copy = df_acc.copy()
    df_basis = pd.read_csv('basis_experiments_table'+str(num_subs)+'.csv')

    # filter df_basis based on exp_settings
    df_basis = filter_df(df_basis,exp_settings)
    df_basis = df_basis[df_basis.exists]
    # check if bases already done for all subjects
    is_done = np.zeros(df_basis.shape[0],dtype=bool)
    i=0
    for _,row in df_basis.iterrows():
        df_acc_row = check_exists(df_acc,row)
        if len(exp_settings['contrasts'])==47:
            if len(df_acc_row)==47*int(num_subs):
                is_done[i] = True
        else:
            if len(df_acc_row)==int(num_subs):
                is_done[i] = True
        i+=1
    df_basis = df_basis[~is_done]
    print('Already done with '+str(is_done.sum())+' rows')
    df_avg_basis_int = np.where(df_basis.num_subs.isin([num_subs,'0']))[0]
    df_avg_basis_idx = df_basis.index[df_avg_basis_int]
    bases_avg = load_bases(df=df_basis.iloc[df_avg_basis_int],subject=None,mask=mask,num_subs=num_subs)

    # Load the individual eigenmodes for each subject.
    for subject in subject_list:
        df_basis_sub = df_basis.copy()
        df_avg_basis_idx_sub = df_avg_basis_idx.copy()
        # check if this row already exists in df_acc
        is_done = np.zeros(df_basis_sub.shape[0],dtype=bool)
        i=0
        for _,row in df_basis_sub.iterrows():
            df_acc_row = check_exists(df_acc,row,subject=subject)
            if df_acc_row.size>0:
                print('Found reconstruction accuracies for subject '+subject)
                is_done[i] = True   
            i+=1
        df_basis_sub = df_basis_sub[~is_done]
        df_avg_basis_int_sub = np.where(df_basis_sub.num_subs.isin([num_subs,'0']))[0]
        df_avg_basis_idx_sub = df_basis_sub.index[df_avg_basis_int_sub]
        avg_include = np.in1d(df_avg_basis_idx,df_avg_basis_idx_sub)

        if df_basis_sub.shape[0]==0:
            print('Already completed '+subject)
            continue

        print('Loading bases for subject '+subject+'...')
        df_ind_basis_int_sub = np.where(df_basis_sub.num_subs=='ind')[0]
        bases = np.zeros((df_basis_sub.shape[0],mask.sum(),200),dtype=np.float64)
        bases_ind = load_bases(df=df_basis_sub.iloc[df_ind_basis_int_sub],subject=subject,mask=mask)
        bases[df_avg_basis_int_sub] = bases_avg[avg_include]
        bases[df_ind_basis_int_sub] = bases_ind

        if 'rest' in exp_settings['contrasts']:
            recon_accuracies = run_reconstruction(bases=bases,num_modes_recon=200,subject=subject,mask=mask,parc=parc,task_or_rest='rest')
        elif 'SSBCAP' in exp_settings['contrasts']:
            recon_accuracies = run_reconstruction(bases=bases,num_modes_recon=200,subject=subject,mask=mask,parc=parc,task_or_rest='SSBCAP')
        elif 'noise' in exp_settings['contrasts']:
            recon_accuracies = run_reconstruction(bases=bases,num_modes_recon=200,subject=subject,mask=mask,parc=parc,task_or_rest='noise')
        else:
            recon_accuracies = run_reconstruction(bases=bases,num_modes_recon=200,subject=subject,mask=mask,parc=parc,task_or_rest='task')
        
        b=0
        for _,row in df_basis_sub.iterrows():
            if len(exp_settings['contrasts'])==47:
                for i,task in enumerate(exp_settings['contrasts']):
                    # extend df_acc with new rows
                    combinations = [subject,row.streamlines,row.density,row.e_local,row.binarization,row.fwhm,row.basis,task,recon_accuracies[:,b,i],np.zeros_like(recon_accuracies[:,b,i])]
                    new_row = pd.DataFrame([combinations],columns=["subject","streamlines", "density", "e_local", "binarization", "fwhm", "basis","contrast","accuracy","accuracy_parc"])
                    df_acc = pd.concat([df_acc,new_row],ignore_index=True)
            else:
                combinations = [subject,row.streamlines,row.density,row.e_local,row.binarization,row.fwhm,row.basis,exp_settings['contrasts'][0],recon_accuracies[:,b,0],np.zeros_like(recon_accuracies[:,b,0])]
                new_row = pd.DataFrame([combinations],columns=["subject","streamlines", "density", "e_local", "binarization", "fwhm", "basis","contrast","accuracy","accuracy_parc"])
                df_acc = pd.concat([df_acc,new_row],ignore_index=True)
            b+=1
                
        
        if len(exp_settings['contrasts'])==47:
            df_acc.to_pickle('reconstruction_accuracies_sorted'+num_subs+'.pkl')
        else:
            df_acc.to_pickle('reconstruction_accuracies_sorted_'+exp_settings['contrasts'][0]+num_subs+'.pkl')
        print('Completed '+subject)
        

if __name__ == '__main__':
    # add parser
    parser = argparse.ArgumentParser(description='Reconstruct data using eigenmodes.')
    parser.add_argument('--num_subs', type=int, default=100, help='number of subjects')
    args = parser.parse_args()
    contrasts = np.loadtxt('contrast_list.txt',dtype=str)
    exp_settings = {'density':['1e-05', '0.0001', '0.001','0.005', '0.01','0.05','0.1','none'],#
                    'e_local':['1.0','none'],
                    'binarization':['binary','weighted','none'],
                    'fwhm':['0.0','2.0','4.0','6.0','8.0','10.0','none'],#,
                    'basis':['ind. surface','template surface'],#,,'A_local','subject-specific','avg connectome basis','ind. surface','template surface'
                    'contrasts':'SSBCAP',#SSBCAP,47task
                    'streamlines':['20M','none']}#,'5M','20M'
    main(num_subs=args.num_subs,exp_settings=exp_settings)