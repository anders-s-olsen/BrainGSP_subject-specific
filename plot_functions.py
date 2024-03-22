import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def filter_df(df,exp_settings):
    # loop through exp_settings and filter df
    for key in exp_settings.keys():
        df = df[df[key].isin(exp_settings[key])]
    return df

def make_cols_categorical(df):
    smooth_vals = ['Unsmoothed','2mm','4mm','6mm','8mm','10mm','12mm','14mm','16mm','18mm','20mm','22mm','24mm','26mm','28mm','30mm','none']
    Binarization_vals = ['Binary','Weighted','none']
    Streamlines_vals = ['5M','10M','20M','50M','none']
    Density_vals = ['0.001%', '0.01%', '0.1%','0.5%', '1%','5%', '10%','none']
    Basis_vals = ['Individual surface', 'Template surface','Individual connectome','Average connectome', 'Average basis: Flag mean','Average basis: Karcher mean (numerical gradient)', 'Average basis: Karcher mean','Average basis: Procrustes','none','A_local','Random smoothed basis']
    e_local_vals = ['1e-05','1.0','none']
       
    remove_vals = np.zeros(len(smooth_vals),dtype=bool)
    for id,d in enumerate(smooth_vals):
        if d not in df['Smoothing (FWHM)'].unique():
            remove_vals[id] = True
    smooth_vals = np.delete(smooth_vals,remove_vals)
    remove_vals = np.zeros(len(Streamlines_vals),dtype=bool)
    for id,d in enumerate(Streamlines_vals):
        if d not in df['Streamlines'].unique():
            remove_vals[id] = True
    Streamlines_vals = np.delete(Streamlines_vals,remove_vals)
    remove_vals = np.zeros(len(Density_vals),dtype=bool)
    for id,d in enumerate(Density_vals):
        if d not in df['Density (%)'].unique():
            remove_vals[id] = True
    Density_vals = np.delete(Density_vals,remove_vals)
    remove_vals = np.zeros(len(e_local_vals),dtype=bool)
    for id,d in enumerate(e_local_vals):
        if d not in df['e_local'].unique():
            remove_vals[id] = True
    e_local_vals = np.delete(e_local_vals,remove_vals)
    remove_vals = np.zeros(len(Basis_vals),dtype=bool)
    for id,d in enumerate(Basis_vals):
        if d not in df['Basis'].unique():
            remove_vals[id] = True
    Basis_vals = np.delete(Basis_vals,remove_vals)
    remove_vals = np.zeros(len(Binarization_vals),dtype=bool)
    for id,d in enumerate(Binarization_vals):
        if d not in df['Binarization'].unique():
            remove_vals[id] = True
    Binarization_vals = np.delete(Binarization_vals,remove_vals)
    
    df['Smoothing (FWHM)'] = pd.Categorical(df['Smoothing (FWHM)'],categories=smooth_vals,ordered=True)
    df['Basis'] = pd.Categorical(df['Basis'],categories=Basis_vals,ordered=True)
    df['Binarization'] = pd.Categorical(df['Binarization'],categories=Binarization_vals,ordered=True)
    df['Streamlines'] = pd.Categorical(df['Streamlines'],categories=Streamlines_vals,ordered=True)
    # df['Contrast'] = pd.Categorical(df['Contrast'],categories=np.unique(df['Contrast']),ordered=True)
    df['Density (%)'] = pd.Categorical(df['Density (%)'],categories=Density_vals,ordered=True)
    df['e_local'] = pd.Categorical(df['e_local'],categories=e_local_vals,ordered=True)
    df['subject'] = pd.Categorical(df['subject'],categories=np.unique(df['subject']),ordered=True)
    return df

def rename_df(df):
    df = df.rename(columns={'accuracy':'Reconstruction accuracy','accuracy_parc':'Reconstruction accuracy (parcellated)',
    'basis':'Basis','binarization':'Binarization','density':'Density (%)','contrast':'Contrast',
    'fwhm':'Smoothing (FWHM)','streamlines':'Streamlines'})

    old_density_names = ['1e-05', '0.0001','0.001','0.005','0.01','0.05', '0.1','none']#
    new_density_names = ['0.001%', '0.01%', '0.1%','0.5%', '1%','5%', '10%','none'] #
    # check if old_density_names has any elements in df['Density (%)'], remove them if not so
    remove_vals = np.zeros(len(old_density_names),dtype=bool)
    for id,d in enumerate(old_density_names):
        if d not in df['Density (%)'].unique():
            remove_vals[id] = True
    old_density_names = np.delete(old_density_names,remove_vals)
    new_density_names = np.delete(new_density_names,remove_vals)
    for id,d in enumerate(old_density_names):
        df['Density (%)'].replace(d, new_density_names[id], inplace=True)

    old_basis_names = ['ind. surface', 'template surface','subject-specific','avg connectome basis', 'avg basis flag','avg basis karcher opt', 'avg basis karcher Begelfor','avg basis procrustes','none','A_local','random_smoothed_basis']
    new_basis_names = ['Individual surface', 'Template surface','Individual connectome','Average connectome', 'Average basis: Flag mean','Average basis: Karcher mean (numerical gradient)', 'Average basis: Karcher mean','Average basis: Procrustes','none','A_local','Random smoothed basis']
    remove_vals = np.zeros(len(old_basis_names),dtype=bool)
    for id,d in enumerate(old_basis_names):
        if d not in df['Basis'].unique():
            remove_vals[id] = True
    old_basis_names = np.delete(old_basis_names,remove_vals)
    new_basis_names = np.delete(new_basis_names,remove_vals)
    for id,d in enumerate(old_basis_names):
        df['Basis'].replace(d, new_basis_names[id], inplace=True)
    
    df['Binarization'].replace('binary','Binary',inplace=True)
    df['Binarization'].replace('weighted','Weighted',inplace=True)

    old_smoothing_names = ['0.0','2.0','4.0','6.0','8.0','10.0','none','2','4','6','8','10','12','14','16','18','20','22','24','26','28','30']
    new_smoothing_names = ['Unsmoothed','2mm','4mm','6mm','8mm','10mm','none','2mm','4mm','6mm','8mm','10mm','12mm','14mm','16mm','18mm','20mm','22mm','24mm','26mm','28mm','30mm']
    remove_vals = np.zeros(len(old_smoothing_names),dtype=bool)
    for id,d in enumerate(old_smoothing_names):
        if d not in df['Smoothing (FWHM)'].unique():
            remove_vals[id] = True
    old_smoothing_names = np.delete(old_smoothing_names,remove_vals)
    new_smoothing_names = np.delete(new_smoothing_names,remove_vals)
    for id,d in enumerate(old_smoothing_names):
        df['Smoothing (FWHM)'].replace(d, new_smoothing_names[id], inplace=True)

    # df = make_cols_categorical(df)

    # df.sort_values(['Density (%)','Smoothing (FWHM)'],inplace=True)
    
    return df

def convert_df(df,do_df_plot=False):
    
    data_parc = np.concatenate(df['Reconstruction accuracy (parcellated)'].tolist())
    data = np.concatenate(df['Reconstruction accuracy'].tolist())
    if do_df_plot:
        df_plot = pd.DataFrame({'Reconstruction accuracy':data,'Reconstruction accuracy (parcellated)':data_parc})
        df2 = df.drop(['Reconstruction accuracy'],axis=1)
        df2 = df2.drop(['Reconstruction accuracy (parcellated)'],axis=1)
        df_new = pd.DataFrame(np.repeat(df2.values,200,axis=0),columns=df2.columns)
        df_plot = pd.concat([df_plot,df_new],axis=1)
        df_plot['Number of modes'] = np.tile(np.arange(1,201),(df.shape[0]))
    else:
        df_plot = []

    # AUC_parc = []
    AUC = []
    # average the first 200 elements of data, then 201..400 etc
    for i in range(0, len(data), 200):
        # AUC_parc.append(np.mean(data_parc[i:i+200]))
        AUC.append(np.mean(data[i:i+200]))

    df_AUC = df
    # remove accuracy_parc
    df_AUC['Reconstruction accuracy'] = AUC
    # df_AUC['Reconstruction accuracy (parcellated)'] = AUC_parc
    return df_plot, df_AUC

def avg_over_tasks(df_plot,df_AUC,which=0,do_df_plot=False):
    if which==0:
        ylab = 'Reconstruction accuracy'
    elif which==1:
        ylab = 'Reconstruction accuracy (parcellated)'
    # if np.any(df_plot[ylab]==0):
    #     idx = np.where(df_plot[ylab]==0)[0]
    #     raise ValueError(ylab+' cannot be 0, here''s the first:'+str(idx[0]))
   
    df_AUC2 = df_AUC.groupby(['subject','Streamlines','Smoothing (FWHM)','Binarization','Density (%)','e_local','Basis'],observed=True)[ylab].mean().reset_index()
    if do_df_plot:
        df_plot2 = df_plot.groupby(['subject','Streamlines','Smoothing (FWHM)','Binarization','Density (%)','e_local','Basis','Number of modes'],observed=True)[ylab].mean().reset_index()
    else:
        df_plot2 = []
    
    return df_plot2,df_AUC2

def plot_func(df_plot,df_AUC,group1,group2=None,title='',which=0):
    if which==0:
        ylab = 'Reconstruction accuracy'
        ylim = [0,1]
    elif which==1:
        ylab = 'Reconstruction accuracy (parcellated)'
        ylim = [0.5,1]
    
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    fig, axs = plt.subplots(1,2,figsize=(14, 6))
    plt.suptitle(title)
    sns.lineplot(x='Number of modes',y=ylab, data=df_plot[df_plot['Binarization']!='none'], ax=axs[0],hue=group1,style=group2)
    axs[0].set_xlabel('Method')
    axs[0].set_ylabel(ylab)
    axs[0].set_ylim([0,1])
    axs[0].set_title(ylab)
    handles, labels = axs[0].get_legend_handles_labels()
    leg1=axs[0].legend(handles, labels, loc='lower right',fontsize=10)
    axs[0].plot(df_plot[(df_plot['Binarization']=='none')&(df_plot['Basis']=='Individual surface')].groupby('Number of modes',observed=True)[ylab].mean(),color='black',linestyle='--',linewidth=1.5)
    axs[0].plot(df_plot[(df_plot['Binarization']=='none')&(df_plot['Basis']=='Template surface')].groupby('Number of modes',observed=True)[ylab].mean(),color='black',linestyle='-',linewidth=1.5)
    # Custom legend entries for the last two lines
    legend_new_labels = ['Template surface','Individual surface',]
    legend_lines = [plt.Line2D([0], [0], color='black', linestyle='-', linewidth=1.5),
                    plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5)]
    axs[0].add_artist(leg1)
    axs[0].legend(handles=legend_lines, labels=legend_new_labels, loc='upper left',fontsize=10)
    handles += legend_lines
    labels += legend_new_labels

    plt.tight_layout()
    sns.boxplot(x=group1,hue=group2,y=ylab,data=df_AUC[df_AUC['Binarization']!='none'],ax=axs[1])
    axs[1].set_ylabel('AUC')
    # axs[1].set_ylim(ylim)
    axs[1].set_title('AUC')
    axlims = axs[1].get_xlim()
    axs[1].hlines(y=np.array(df_AUC[(df_AUC['Binarization']=='none')&(df_AUC['Basis']=='Individual surface')][ylab]).mean(),xmin=axlims[0],xmax=axlims[1],color='black',linestyle='--',linewidth=1.5)
    axs[1].hlines(y=np.array(df_AUC[(df_AUC['Binarization']=='none')&(df_AUC['Basis']=='Template surface')][ylab]).mean(),xmin=axlims[0],xmax=axlims[1],color='black',linestyle='-',linewidth=1.5)
    handles, labels = axs[1].get_legend_handles_labels()
    handles += legend_lines
    labels += legend_new_labels
    axs[1].legend(handles, labels, loc='lower right',fontsize=10)
    plt.tight_layout()

def plot_func2(df_AUC,group1,group2=None,group3=None,group4=None,title='',which=0):
    if which==0:
        ylab = 'Reconstruction accuracy'
        ylim = [0.25,0.65]
    elif which==1:
        ylab = 'Reconstruction accuracy (parcellated)'
        ylim = [0.5,1]
    group3vals = df_AUC[group3].unique()
    group3vals = np.delete(group3vals,group3vals=='none')
    group4vals = df_AUC[group4].unique()
    group4vals = np.delete(group4vals,group4vals=='none')
    ind_surf_val = np.array(df_AUC[df_AUC['Basis']=='Individual surface'][ylab]).mean()
    template_surf_val = np.array(df_AUC[df_AUC['Basis']=='Template surface'][ylab]).mean()
    
    df_AUC = df_AUC[df_AUC['Binarization']!='none']
    df_AUC = make_cols_categorical(df_AUC)
    
    fig,axs = plt.subplots(2,3,figsize=(16, 8))
    plt.suptitle(title)
    for i in range(3):
        for j in range(2):
            dftmp = df_AUC[(df_AUC['Basis']!='Individual surface')&(df_AUC['Basis']!='Template surface')]
            dftmp = dftmp[dftmp[group3]==group3vals[i]]
            dftmp = dftmp[dftmp[group4]==group4vals[j]]
            sns.boxplot(x=group1,hue=group2,y=ylab,data=dftmp,ax=axs[j,i])
            axs[j,i].set_ylabel('AUC')
            axs[j,i].set_ylim(ylim)
            axs[j,i].set_title(group3vals[i]+' '+group4vals[j])
            axlims = axs[j,i].get_xlim()
            handles, labels = axs[j,i].get_legend_handles_labels()
            # keep only handles and labels that are actually shown
            unique_group2 = np.unique(dftmp[group2])
            # remove the labels that are not in the plot
            remove_vals = np.zeros(len(labels),dtype=bool)
            for id,l in enumerate(labels):
                if l not in unique_group2:
                    remove_vals[id] = True
            handles = np.delete(handles,remove_vals).tolist()
            labels = np.delete(labels,remove_vals).tolist()
            
            axs[j,i].hlines(y=ind_surf_val,xmin=axlims[0],xmax=axlims[1],color='black',linestyle='--',linewidth=1.5)
            axs[j,i].hlines(y=template_surf_val,xmin=axlims[0],xmax=axlims[1],color='black',linestyle='-',linewidth=1.5)
            legend_new_labels = ['Template surface','Individual surface',]
            legend_lines = [plt.Line2D([0], [0], color='black', linestyle='-', linewidth=1.5),
                            plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5)]
            handles += legend_lines
            labels += legend_new_labels
            axs[j,i].legend(handles, labels, loc='lower left',fontsize=10)
            plt.tight_layout()


def plot_func3(df_AUC,group1,group2=None,group3=None,group4=None,title='',which=0):
    if which==0:
        ylab = 'Reconstruction accuracy'
        ylim = [0.45,0.6]
    elif which==1:
        ylab = 'Reconstruction accuracy (parcellated)'
        ylim = [0.5,1]
    group3vals = df_AUC[group3].unique()
    group3vals = np.delete(group3vals,group3vals=='none')
    group4vals = df_AUC[group4].unique()
    group4vals = np.delete(group4vals,group4vals=='none')
    ind_surf_val = np.array(df_AUC[df_AUC['Basis']=='Individual surface'][ylab]).mean()
    template_surf_val = np.array(df_AUC[df_AUC['Basis']=='Template surface'][ylab]).mean()
    
    df_AUC = df_AUC[df_AUC['Binarization']!='none']
    df_AUC = make_cols_categorical(df_AUC)
    
    fig,axs = plt.subplots(2,3,figsize=(16, 8))
    plt.suptitle(title)
    for i in range(3):
        for j in range(2):
            dftmp = df_AUC[(df_AUC['Basis']!='Individual surface')&(df_AUC['Basis']!='Template surface')]
            dftmp = dftmp[dftmp[group3]==group3vals[i]]
            dftmp = dftmp[dftmp[group4]==group4vals[j]]
            
            sns.lineplot(x=group1,hue=group2,y=ylab,data=dftmp,ax=axs[j,i],markers=True,errorbar='se')
            axs[j,i].set_ylabel('AUC')
            axs[j,i].set_ylim(ylim)
            axs[j,i].set_title(group3vals[i]+' '+group4vals[j])
            axlims = axs[j,i].get_xlim()
            handles, labels = axs[j,i].get_legend_handles_labels()
            # keep only handles and labels that are actually shown
            unique_group2 = np.unique(dftmp[group2])
            # remove the labels that are not in the plot
            remove_vals = np.zeros(len(labels),dtype=bool)
            for id,l in enumerate(labels):
                if l not in unique_group2:
                    remove_vals[id] = True
            handles = np.delete(handles,remove_vals).tolist()
            labels = np.delete(labels,remove_vals).tolist()
            axs[j,i].hlines(y=ind_surf_val,xmin=axlims[0],xmax=axlims[1],color='black',linestyle='--',linewidth=1.5)
            axs[j,i].hlines(y=template_surf_val,xmin=axlims[0],xmax=axlims[1],color='black',linestyle='-',linewidth=1.5)
            legend_new_labels = ['Template surface','Individual surface',]
            legend_lines = [plt.Line2D([0], [0], color='black', linestyle='-', linewidth=1.5),
                            plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5)]
            handles += legend_lines
            labels += legend_new_labels
            axs[j,i].legend(handles, labels, loc='lower left',fontsize=10)
            plt.tight_layout()

def plot_func4(df_AUC,group1,group2=None,group3=None,group4=None,title='',which=0,leg=True,ylim=[0.15,0.7]):
    if which==0:
        ylab = 'Reconstruction accuracy'
    elif which==1:
        ylab = 'Reconstruction accuracy (parcellated)'
    ind_surf_val = np.array(df_AUC[df_AUC['Basis']=='Individual surface'][ylab]).mean()
    template_surf_val = np.array(df_AUC[df_AUC['Basis']=='Template surface'][ylab]).mean()
    df_AUC = df_AUC[df_AUC['Binarization']!='none']
    group3vals = df_AUC[group3].unique()
    group3vals = np.delete(group3vals,group3vals=='none')
    group4vals = df_AUC[group4].unique()
    group4vals = np.delete(group4vals,group4vals=='none')
    df_AUC = make_cols_categorical(df_AUC)
    
    fig,axs = plt.subplots(2,2,figsize=(9, 6))
    # plt.suptitle(title)
    for i in range(2):
        for j in range(2):
            dftmp = df_AUC[(df_AUC['Basis']!='Individual surface')&(df_AUC['Basis']!='Template surface')]
            dftmp = dftmp[dftmp[group3]==group3vals[i]]
            dftmp = dftmp[dftmp[group4]==group4vals[j]]
            sns.boxplot(x=group1,hue=group2,y=ylab,data=dftmp,ax=axs[j,i])

            if i==0:
                axs[j,i].set_ylabel('AUC')
                if ylim[0]<0.1:
                    axs[j,i].set_yticks([-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.1])
                elif ylim[0]<0.3:
                    axs[j,i].set_yticks([0.2,0.3,0.4,0.5,0.6,0.7])
                else:
                    axs[j,i].set_yticks([0.3,0.4,0.5,0.6,0.7])
            else:
                axs[j,i].set_ylabel('')
                axs[j,i].set_yticklabels([])
            
            if j==0:
                axs[j,i].set_xlabel('')
                axs[j,i].set_xticklabels([])
            else:
                axs[j,i].set_xticklabels(axs[j,i].get_xticklabels(),fontsize=9)
                
            axs[j,i].set_ylim(ylim)
            # axs[j,i].set_title(group3vals[i]+' '+group4vals[j])
            axlims = axs[j,i].get_xlim()
            handles, labels = axs[j,i].get_legend_handles_labels()
            # keep only handles and labels that are actually shown
            unique_group2 = np.unique(dftmp[group2])
            # remove the labels that are not in the plot
            remove_vals = np.zeros(len(labels),dtype=bool)
            for id,l in enumerate(labels):
                if l not in unique_group2:
                    remove_vals[id] = True
            handles = np.delete(handles,remove_vals).tolist()
            labels = np.delete(labels,remove_vals).tolist()
            
            axs[j,i].hlines(y=ind_surf_val,xmin=axlims[0],xmax=axlims[1],color='black',linestyle='--',linewidth=1.5)
            axs[j,i].hlines(y=template_surf_val,xmin=axlims[0],xmax=axlims[1],color='black',linestyle='-',linewidth=1.5)
            legend_new_labels = ['Template surface','Individual surface',]
            legend_lines = [plt.Line2D([0], [0], color='black', linestyle='-', linewidth=1.5),
                            plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5)]
            handles += legend_lines
            labels += legend_new_labels
            if leg:
                axs[j,i].set_ylim([-0.5,0.5])
                axs[j,i].legend(handles, labels, loc='lower left',fontsize=10,ncol=4,bbox_to_anchor=(0.5, 1.1))
            else:
                axs[j,i].legend([],[], frameon=False)
            plt.tight_layout()
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)


if __name__=="__main__":
    df = pd.read_pickle('reconstruction_accuracies20.pkl')
    # task, subject-specific
    exp_settings = {'basis':['avg connectome basis','ind. surface','template surface'],
                    'e_local':['1.0','none'],
                    'density':['1e-05', '0.0001','0.001','0.005','0.01', '0.1','none'],
                    'binarization':['binary','weighted','none'],
                    'fwhm':['0.0','2.0','4.0','6.0','8.0','10.0','none'],
                    'contrast':np.loadtxt('contrast_list.txt',dtype=str),
                    'streamlines':['5M','10M','20M','none']}

    df2 = filter_df(df,exp_settings)
    df2 = rename_df(df2)
    df_plot,df_AUC = convert_df(df2)
    df_plot,df_AUC = avg_over_tasks(df_plot,df_AUC,which=0)
    plot_func2(df_AUC,group1='Smoothing (FWHM)',group2='Density (%)',group3='Streamlines',group4='Binarization',title='Subject-specific connectomes',which=0)

# # counts, relies on previous cell
# fig,axs = plt.subplots(2,3,figsize=(16, 8))
# sns.countplot(x='Density (%)',hue='Smoothing (FWHM)',data=df_AUC[(df_AUC['Binarization']=='Binary')&(df_AUC['Streamlines']=='5M')],ax=axs[0,0]),axs[0,0].legend([],[], frameon=False)
# axs[0,0].set_title('5M streamlines, binary')
# sns.countplot(x='Density (%)',hue='Smoothing (FWHM)',data=df_AUC[(df_AUC['Binarization']=='Binary')&(df_AUC['Streamlines']=='10M')],ax=axs[0,1]),axs[0,1].legend([],[], frameon=False)
# axs[0,1].set_title('10M streamlines, binary')
# sns.countplot(x='Density (%)',hue='Smoothing (FWHM)',data=df_AUC[(df_AUC['Binarization']=='Binary')&(df_AUC['Streamlines']=='20M')],ax=axs[0,2]),axs[0,2].legend([],[], frameon=False)
# axs[0,2].set_title('20M streamlines, binary')
# sns.countplot(x='Density (%)',hue='Smoothing (FWHM)',data=df_AUC[(df_AUC['Binarization']=='Weighted')&(df_AUC['Streamlines']=='5M')],ax=axs[1,0]),axs[1,0].legend([],[], frameon=False)
# axs[1,0].set_title('5M streamlines, weighted')
# sns.countplot(x='Density (%)',hue='Smoothing (FWHM)',data=df_AUC[(df_AUC['Binarization']=='Weighted')&(df_AUC['Streamlines']=='10M')],ax=axs[1,1]),axs[1,1].legend([],[], frameon=False)
# axs[1,1].set_title('10M streamlines, weighted')
# sns.countplot(x='Density (%)',hue='Smoothing (FWHM)',data=df_AUC[(df_AUC['Binarization']=='Weighted')&(df_AUC['Streamlines']=='20M')],ax=axs[1,2])
# axs[1,2].set_title('20M streamlines, weighted')