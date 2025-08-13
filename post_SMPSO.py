# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:25:18 2025

@author: u226422
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pydaisy import Daisy
import numpy as np
import os
from scipy.stats import mannwhitneyu
from scipy.stats import median_test
from scipy.stats import ttest_ind
from scipy.stats import linregress
import datetime
os.chdir(r"C:\Users\u226422\OneDrive - Universite de Liege\Documents\daisy\CALIB\Outputs_multi")
import defs_calib

dataset = 'SMPSO_alloc2'
F = pd.read_csv("outF_"+dataset+".csv", usecols=[1, 2, 3])
X = pd.read_csv("outX_"+dataset+".csv")
numpF = F.to_numpy()
dfX = X.iloc[:, 1:]
names = ['fPSII_sah', 'fPSII_tob', 'fPSII_sma', 'fPSII_sky', 'Xn_sah', 'Xn_tob', 'Xn_sma', 'Xn_sky',
         'Do_sah', 'Do_tob', 'Do_sma', 'Do_sky', 'SpLAI_sah', 'SpLAI_tob', 'SpLAI_sma', 'SpLAI_sky',
         'm_sah', 'm_tob', 'm_sma', 'm_sky', 'A$_{brunt}$', 'B$_{brunt}$', '$\\varepsilon_{leaf}$',
         'b_sah', 'b_tob', 'b_sma', 'b_sky', '$\\sigma_{NIR}$', 'PenPar2', 'NNI$_{crit}$_sah',
         'NNI$_{crit}$_tob', 'NNI$_{crit}$_sma', 'NNI$_{crit}$_sky', 'Ksat$_1$', 'ShldResC_sah',
         'ShldResC_tob', 'ShldResC_sma', 'ShldResC_sky', '$\\delta_{leun}$_sah',
         '$\\delta_{leun}$_tob', '$\\delta_{leun}$_sma', '$\\delta_{leun}$_sky', 'k$_{Net}$',
         'SOrgPhotEff_sah', 'SOrgPhotEff_tob', 'SOrgPhotEff_sma', 'SOrgPhotEff_sky',
         'StemPhotEff_sah', 'StemPhotEff_tob', 'StemPhotEff_sma', 'StemPhotEff_sky',
         'Ke', 'Ksat$_{aquitard}$', 'Z$_{aquitard}$', 'Kc']

#%% GETTING 2-OBJ FRONT

def get_2d_pareto_front(df, obj_x, obj_y, minimize=True):
    """
    Returns the 2-D Pareto front from a DataFrame of multi-objective points.
    
    Args:
        df: pandas DataFrame containing the objective values.
        obj_x: str, column name for the first objective.
        obj_y: str, column name for the second objective.
        minimize: bool or tuple of bools. If True, assume minimization for both.
                  If tuple, specify (minimize_x, minimize_y).
                  
    Returns:
        pareto_df: subset of df corresponding to the 2-D Pareto front.
    """
    data = df[[obj_x, obj_y]].values
    if isinstance(minimize, bool):
        minimize = (minimize, minimize)
    if not minimize[0]:
        data[:, 0] *= -1
    if not minimize[1]:
        data[:, 1] *= -1
    pareto_mask = np.ones(data.shape[0], dtype=bool)
    for i in range(data.shape[0]):
        if pareto_mask[i]:
            pareto_mask[pareto_mask] = np.any(data[pareto_mask] < data[i], axis=1) | np.all(data[pareto_mask] == data[i], axis=1)
            pareto_mask[i] = True  # Keep self

    return df[pareto_mask]

pareto_01 = get_2d_pareto_front(F, '0', '1')
pareto_12 = get_2d_pareto_front(F, '1', '2')

#%% CHOSING THE BEST PARAMETER SET WITH SCALING -- RED POINT

obj_min = np.min(numpF, axis=0)
obj_max = np.max(numpF, axis=0)
# Normalize objectives using min-max scaling
obj_scaled = (numpF - obj_min) / (obj_max - obj_min)
ref_point = np.zeros(3)  
# Compute Euclidean distance to the reference point
distances_scaled = np.linalg.norm(obj_scaled - ref_point, axis=1)
scaled_index = np.argmin(distances_scaled)
scaled_params = dfX.iloc[scaled_index, :]

#%% PARETO FRONT (FIG 1)

plt.ioff()
couleur = 'grey'
X, Y, Z = numpF[:, 0], numpF[:, 2], numpF[:, 1]
fig = plt.figure(figsize=(8, 5), layout='constrained', dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
gs_right = gs[1].subgridspec(2, 1)
ax = fig.add_subplot(gs[0], projection='3d')
ax.scatter(X, Y, Z, color=couleur, s=15, linewidths=1)
ax.scatter(X[scaled_index], Y[scaled_index], Z[scaled_index], color='red')
ax.set(xlim=(0.2, 0.45), ylim=(0.48, 0.68), zlim=(0.35, 0.65))
ax.set_yticks([0.5, 0.55, 0.6, 0.65])
ax.set_zticks([0.4, 0.5, 0.6], fontname='monospace')
# ax.tick_params(axis='y', which='major', pad=-2, labelrotation=-15)
ax.tick_params(axis='both', labelsize=10, direction='inout', pad=2)
ax.set_yticklabels([0.5, "", 0.6, ""], rotation=0, horizontalalignment='left',
                   verticalalignment='bottom', fontname='monospace')
ax.set_xticklabels([0.2, "", 0.3, "", 0.4, ""], horizontalalignment='center',
                   verticalalignment='center', fontname='monospace')
ax.set_zticklabels([0.4, 0.5, 0.6], horizontalalignment='center',
                   verticalalignment='center', fontname='monospace')
ax.set_xlabel('F$_{DM}$', rotation=15, labelpad=4, fontname='monospace')
ax.set_ylabel('F$_{LE}$', rotation=50, labelpad=7, fontname='monospace')
ax.set_zlabel('F$_{NEE}$', rotation=90, labelpad=4, fontname='monospace')
ax.text(0.2, 0.5, 0.75, 'a.', fontname='monospace')
ax1 = fig.add_subplot(gs_right[0])
ax1.scatter(X, Z, c=couleur, alpha=0.8, s=15, linewidths=1)
ax1.scatter(pareto_01.iloc[:, 0], pareto_01.iloc[:, 1], c='black', s=15)
ax1.scatter(X[scaled_index], Z[scaled_index], color='red', s=18)
ax1.set_xlabel('F$_{DM}$', fontname='monospace')
ax1.set_ylabel('F$_{NEE}$', fontname='monospace')
ax1.yaxis.set_label_position("right")
ax1.tick_params(axis='both', labelsize=10, direction='inout', top=True, right=True, labelleft=False,
                labelright=True)
ax1.set_yticks([0.35, 0.4, 0.45, 0.5, 0.55, 0.6], fontname='monospace')
ax1.set_yticklabels(["", 0.4, "", 0.5, "", 0.6])
ax1.set_xticks([0.2, 0.25, 0.3, 0.35, 0.4], fontname='monospace')
ax1.set_xticklabels([0.2, "", 0.3, "", 0.4])
ax1.set_ylim([0.34, 0.61])
ax1.text(0.15, 0.58, 'b.', fontname='monospace')
for label in ax1.get_xticklabels() :
    label.set_fontproperties('monospace')
for label in ax1.get_yticklabels() :
    label.set_fontproperties('monospace')
ax2 = fig.add_subplot(gs_right[1])
ax2.scatter(Y, Z, c=couleur, alpha=0.8, s=15, linewidths=1)
ax2.scatter(pareto_12.iloc[:, 2], pareto_12.iloc[:, 1], c='black', s=15)
ax2.scatter(Y[scaled_index], Z[scaled_index], color='red', s=18)
ax2.set_xlabel('F$_{LE}$', fontname='monospace')
ax2.set_ylabel('F$_{NEE}$', fontname='monospace')
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis='both', labelsize=10, direction='inout', top=True, right=True, labelleft=False,
                labelright=True)
ax2.set_yticks([0.35, 0.4, 0.45, 0.5, 0.55, 0.6], fontname='monospace')
ax2.set_yticklabels(["", 0.4, "", 0.5, "", 0.6])
ax2.set_xticks([0.5, 0.55, 0.6, 0.65], fontname='monospace')
ax2.set_xticklabels([0.5, "", 0.6, ""])
ax2.set_ylim([0.34, 0.61])
ax2.set_xlim([0.49, 0.66])
ax1.text(0.15, 0.215, 'c.', fontname='monospace')
for label in ax2.get_xticklabels() :
    label.set_fontproperties('monospace')
for label in ax2.get_yticklabels() :
    label.set_fontproperties('monospace')
plt.subplots_adjust(left=0.1, right=0.9, 
                    top=0.9, bottom=0.1, 
                    wspace=0.4, hspace=0.4)
fig.savefig('Figure_'+dataset+'.png')

#%% RUNNING DAISY

def writing_nomore_common_dai(x, name):
    dai = Daisy.DaisyModel(os.getcwd()+r'\wheat_LTO.dai')
    if name == 'sah':
        defs_calib.photosynthesis(dai, 'alfa', x[0])
        defs_calib.photosynthesis(dai, 'Xn', x[4])
        defs_calib.stomatal(dai, 'Do_Leun', x[8])
        defs_calib.canopy(dai, 'SpLAI', x[12])
        defs_calib.stomatal(dai, 'm_Leun', x[16])
        defs_calib.stomatal(dai, 'b_Leun', x[23])
        defs_calib.ssoc(dai, 'b_Leun', x[23])
        defs_calib.partitioning(dai, 'NNI_crit', x[29])
        defs_calib.production(dai, 'ShldResC', x[34])
        defs_calib.stomatal(dai, 'delta_Leun', x[38])
        defs_calib.canopy(dai, 'SOrgPhotEff', x[43])
        defs_calib.canopy(dai, 'StemPhotEff', x[47])
    elif name == 'tob':
        defs_calib.photosynthesis(dai, 'alfa', x[1])
        defs_calib.photosynthesis(dai, 'Xn', x[5])
        defs_calib.stomatal(dai, 'Do_Leun', x[9])
        defs_calib.canopy(dai, 'SpLAI', x[13])
        defs_calib.stomatal(dai, 'm_Leun', x[17])
        defs_calib.stomatal(dai, 'b_Leun', x[24])
        defs_calib.ssoc(dai, 'b_Leun', x[24])
        defs_calib.partitioning(dai, 'NNI_crit', x[30])
        defs_calib.production(dai, 'ShldResC', x[35])
        defs_calib.stomatal(dai, 'delta_Leun', x[39])
        defs_calib.canopy(dai, 'SOrgPhotEff', x[44])
        defs_calib.canopy(dai, 'StemPhotEff', x[48])
    elif name == 'sma':
        defs_calib.photosynthesis(dai, 'alfa', x[2])
        defs_calib.photosynthesis(dai, 'Xn', x[6])
        defs_calib.stomatal(dai, 'Do_Leun', x[10])
        defs_calib.canopy(dai, 'SpLAI', x[14])
        defs_calib.stomatal(dai, 'm_Leun', x[18])
        defs_calib.stomatal(dai, 'b_Leun', x[25])
        defs_calib.ssoc(dai, 'b_Leun', x[25])
        defs_calib.partitioning(dai, 'NNI_crit', x[31])
        defs_calib.production(dai, 'ShldResC', x[36])
        defs_calib.stomatal(dai, 'delta_Leun', x[40])
        defs_calib.canopy(dai, 'SOrgPhotEff', x[45])
        defs_calib.canopy(dai, 'StemPhotEff', x[49])
    elif name == 'sky':
        defs_calib.photosynthesis(dai, 'alfa', x[3])
        defs_calib.photosynthesis(dai, 'Xn', x[7])
        defs_calib.stomatal(dai, 'Do_Leun', x[11])
        defs_calib.canopy(dai, 'SpLAI', x[15])
        defs_calib.stomatal(dai, 'm_Leun', x[19])
        defs_calib.stomatal(dai, 'b_Leun', x[26])
        defs_calib.ssoc(dai, 'b_Leun', x[26])
        defs_calib.partitioning(dai, 'NNI_crit', x[32])
        defs_calib.production(dai, 'ShldResC', x[37])
        defs_calib.stomatal(dai, 'delta_Leun', x[41])
        defs_calib.canopy(dai, 'SOrgPhotEff', x[46])
        defs_calib.canopy(dai, 'StemPhotEff', x[50])
    defs_calib.radiation(dai, 'A_brunt', x[20])  # A_brunt
    defs_calib.radiation(dai, 'B_brunt', x[21])  # B_brunt
    defs_calib.ssoc(dai, 'epsilon_leaf', x[22])  # epsilon_leaf
    defs_calib.raddist(dai, 'sigma_NIR', x[27])  # sigma_NIR
    defs_calib.root(dai, 'PenPar2', x[28])  # PenPar2
    defs_calib.hydraulic(dai, 'K_sat_1', x[33])  # K_sat 1
    defs_calib.canopy(dai, 'k_net', x[42])  # k_net (EPext)
    defs_calib.surface(dai, 'EpFactor', x[51])  # EpFactor (Ke)
    defs_calib.aquitard(dai, 'K_aquitard', x[52])  # K_aquitard
    defs_calib.aquitard(dai, 'Z_aquitard', x[53])  # Z_aquitard
    defs_calib.canopy(dai, 'EpFac', x[54])  # EpFac (Kc)
    dai.save_as(os.getcwd()+r'\wheat_LTO.dai')
    return

compromise = 'scaled'
Daisy.DaisyModel.path_to_daisy_executable = r'C:\Program Files\Daisy 2.11\bin\daisy.exe'
new_dir = '"C:/Users/u226422/OneDrive - Universite de Liege/Documents/daisy/CALIB/Outputs_multi"'
for crop in ['sah', 'tob', 'sma', 'sky']:
    writing_nomore_common_dai(scaled_params, crop)
    d = Daisy.DaisyModel(os.getcwd()+r'\Lonzee_'+crop+'.dai')
    d.Input['directory'].setvalue(new_dir)
    d.Input['output'][0]['where'].setvalue('"crop_'+crop+'_'+compromise+'.dlf"')
    d.Input['output'][1]['where'].setvalue('"bioclim_'+crop+'_'+compromise+'.dlf"')
    d.Input['output'][2]['where'].setvalue('"Cbal_'+crop+'_'+compromise+'.dlf"')
    d.Input['output'][3]['where'].setvalue('"bioclimate_'+crop+'_'+compromise+'.dlf"')
    d.Input['output'][4]['where'].setvalue('"ssoc_'+crop+'_'+compromise+'.dlf"')
    d.Input['output'][5]['where'].setvalue('"surfwat_'+crop+'_'+compromise+'.dlf"')
    d.Input['output'][6]['where'].setvalue('"stomsdw_'+crop+'_'+compromise+'.dlf"')
    d.Input['output'][7]['where'].setvalue('"stomsun_'+crop+'_'+compromise+'.dlf"')
    d.save_as(os.getcwd()+r'\setup_'+crop+'.dai')
    d.run()

#%% GETTING MEASUREMENTS AND MODEL OUTPUTS - FUNCTIONS


class Compromise:
    def __init__(self, compromise_name, crop_names=['sah', 'tob', 'sma', 'sky']):
        self.name = compromise_name
        self.crops = [CropRun(name, compromise_name) for name in crop_names]

    def load_all(self):
        for crop in self.crops:
            crop.get_biomass()
            crop.get_fluxes()
            crop.get_model_outputs()
            crop.get_stats()
            crop.get_other_outputs()

class CropRun:
    def __init__(self, name, compromise):
        self.name = name
        self.compromise = compromise
        self.time_offset = datetime.timedelta(minutes=30)

    def get_biomass(self):
        DM = pd.read_excel(os.getcwd()+r'\observations.xlsx', sheet_name='WShoot_'+self.name,
                           index_col='Date', date_parser=pd.to_datetime)
        self.DM = DM/100
        if self.name == 'sky':
            Root = pd.read_excel(os.getcwd()+r'\observations.xlsx', usecols=['Date', 'WRoot', 'STDRoot'],
                                 sheet_name='WRoot_sky', index_col='Date',
                                 date_parser=pd.to_datetime)
            self.Root = Root/100
        return

    def read_fluxes(self, sheetname, usecols, file_path=os.getcwd()+r'\observations.xlsx'):
        data = pd.read_excel(file_path, sheet_name=sheetname, index_col='Date',
                             date_parser=pd.to_datetime, usecols=usecols)
        data.index = data.index + self.time_offset
        self.starting_date = data.index[0]
        self.ending_date = data.index[-1]
        difference = self.ending_date - self.starting_date
        self.duration_hour = (difference.days * 24) + (difference.seconds // 3600)
        data = data.fillna(np.inf).groupby(pd.Grouper(freq='H')).mean().replace(np.inf, np.nan)
        if sheetname == 'LE_'+self.name:
            daily = data.fillna(np.inf).groupby(pd.Grouper(freq='D')).mean().replace(np.inf, np.nan)
        else:
            daily = data.fillna(np.inf).groupby(pd.Grouper(freq='D')).sum().replace(np.inf, np.nan)*3600*12e-6
        self.starting_day = daily.index[0]
        self.ending_day = daily.index[-1]
        data.dropna(inplace=True)
        return data, daily

    def get_fluxes(self):
        self.NEE, self.dailyNEE = self.read_fluxes('NEE_'+self.name, ['Date', 'NEE'])
        self.LE, self.dailyLE = self.read_fluxes('LE_'+self.name, ['Date', 'LE_CORR'])
        return

    def get_model_outputs(self):
        dlf = pd.read_csv(os.getcwd()+r'\crop_'+self.name+'_'+self.compromise+'.dlf', sep='\t',
                          skiprows=16, header=[0, 1])
        dlf.columns = dlf.columns.droplevel(1)
        dlf.rename(columns={"mday": "day"}, inplace=True)
        dlf.set_index(pd.to_datetime(dlf[['year', 'month', 'day', 'hour']]), inplace=True)
        self.simWSOrg = dlf['WSOrg']
        self.simWLeaf = dlf['WLeaf']
        self.simWStem = dlf['WStem']
        self.simWRoot = dlf['WRoot']
        self.DS = dlf['DS']
        self.LAI = dlf['LAI']
        # Reading NEE [µmol m^-2 s^-1] and LE [W m^-2]
        dlf2 = pd.read_csv(os.getcwd()+r'\bioclim_'+self.name+'_'+self.compromise+'.dlf', sep='\t',
                           skiprows=15, header=[0, 1])
        dlf2.columns = dlf2.columns.droplevel(1)
        dlf2.rename(columns={"mday": "day"}, inplace=True)
        dlf2.set_index(pd.to_datetime(dlf2[['year', 'month', 'day', 'hour']]), inplace=True)
        self.simNEE = dlf2['Bioinc_CO2'] + dlf2['OM_CO2'].values - dlf2['NetPhotosynthesis'].values
        self.simLE = dlf2['total_ea']
        self.simH = dlf2['H_c_a']
        self.simdailyNEE = self.simNEE.groupby(pd.Grouper(freq='D')).mean()
        self.simdNEE = self.simNEE.groupby(pd.Grouper(freq='D')).sum()*3600*12e-6
        self.simdailyLE = self.simLE.groupby(pd.Grouper(freq='D')).mean()
        self.simHR = dlf2['Bioinc_CO2'] + dlf2['OM_CO2'].values
        self.simAR = dlf2['MaintRespiration'] + dlf2['GrowthRespiration'].values
        self.simMR = dlf2['MaintRespiration']
        self.simGR = dlf2['GrowthRespiration']
        self.simOMR = dlf2['OM_CO2']
        self.simGPP = dlf2['CanopyAss']
        self.simdailyHR = self.simHR.groupby(pd.Grouper(freq='D')).mean()
        self.simdHR = self.simHR.groupby(pd.Grouper(freq='D')).sum()*3600*12e-6
        self.simdailyAR = self.simAR.groupby(pd.Grouper(freq='D')).mean()
        self.simdailyGPP = self.simGPP.groupby(pd.Grouper(freq='D')).mean()
        self.starting_sim = dlf2.index[0]
        return

    def get_stats(self):
        e = self.NEE.stack() - self.simNEE[self.NEE.index]
        self.RMSE_NEE = np.sqrt((e**2).mean())
        s, i, self.r_value_NEE, p_val, std_err = linregress(self.NEE.stack(), self.simNEE[self.NEE.index])
        e = self.LE.stack() - self.simLE[self.LE.index]
        self.RMSE_LE = np.sqrt((e**2).mean())
        s, i, self.r_value_LE, p_val, std_err = linregress(self.LE.stack(), self.simLE[self.LE.index])
        

    def read_dlf_files(self, file_name, skip_lines):
        dlf = pd.read_csv(os.getcwd()+'/'+file_name+'_'+self.name+'_'+self.compromise+'.dlf', sep='\t',
                          skiprows=skip_lines, header=[0, 1])
        dlf.columns = dlf.columns.droplevel(1)
        dlf.rename(columns={"mday": "day"}, inplace=True)
        dlf.set_index(pd.to_datetime(dlf[['year', 'month', 'day', 'hour']]), inplace=True)
        return dlf

    def get_other_outputs(self):
        self.BCLM = self.read_dlf_files('bioclimate', 15).iloc[:, 4:]
        self.SSOC = self.read_dlf_files('ssoc', 15).iloc[:, 4:]
        self.SURF = self.read_dlf_files('surfwat', 15).iloc[:, 4:]
        self.SHDW = self.read_dlf_files('stomsdw', 16).iloc[:, 4:]
        self.SUN = self.read_dlf_files('stomsun', 16).iloc[:, 4:]
        self.avgBCLM = self.BCLM.groupby(pd.Grouper(freq='D')).mean()
        self.avgSSOC = self.SSOC.groupby(pd.Grouper(freq='D')).mean()
        self.avgSURF = self.SURF.groupby(pd.Grouper(freq='D')).mean()
        self.avgSHDW = self.SHDW.groupby(pd.Grouper(freq='D')).mean()
        self.avgSUN = self.SUN.groupby(pd.Grouper(freq='D')).mean()
        self.sumBCLM = self.BCLM.groupby(pd.Grouper(freq='D')).sum()
        self.sumSSOC = self.SSOC.groupby(pd.Grouper(freq='D')).sum()
        self.sumSURF = self.SURF.groupby(pd.Grouper(freq='D')).sum()


#%% GETTING MEASUREMENTS AND MODEL OUTPUTS - CALLING

compromise = Compromise('scaled')
compromise.load_all()
saving = True

#%% PLOTTING BIOMASS (FIG 2)

name_crop = ['SAH', 'TOB', 'SMA', 'SKY']
plt.rcParams["font.family"] = "monospace"
fig, ax = plt.subplots(4, 1, sharey=True, figsize=(7, 10), dpi=300)
for i in range(len(compromise.crops)):
    ax[i].plot(compromise.crops[i].simWLeaf.loc[compromise.crops[i].starting_date:],
               color='firebrick', label='Leaf')
    ax[i].plot(compromise.crops[i].simWStem.loc[compromise.crops[i].starting_date:],
               color='mediumseagreen', label='Stem')
    ax[i].plot(compromise.crops[i].simWSOrg.loc[compromise.crops[i].starting_date:],
               color='peru', label='SOrg')
    ax[i].plot(compromise.crops[i].simWRoot.loc[compromise.crops[i].starting_date:],
               color='cornflowerblue', label='Root')
    ax[i].plot(compromise.crops[i].simWLeaf.loc[compromise.crops[i].starting_date:] +
               compromise.crops[i].simWStem.loc[compromise.crops[i].starting_date:],
               color='grey', label='Leaf+Stem')
    ax[i].errorbar(compromise.crops[i].DM.index, compromise.crops[i].DM.WSOrg,
                   yerr=compromise.crops[i].DM.STDSOrg, color='peru', fmt='o', capsize=2,
                   markersize=5)
    ax[i].errorbar(compromise.crops[i].DM.index, compromise.crops[i].DM.WLeaf_Stem,
                   yerr=compromise.crops[i].DM.STDLeaf_Stem, color='grey', fmt='o', capsize=2,
                   markersize=5)
    ax[i].errorbar(compromise.crops[i].DM.index, compromise.crops[i].DM.WLeaf,
                   yerr=compromise.crops[i].DM.STDLeaf, color='firebrick', fmt='o', capsize=2,
                   markersize=5)
    ax[i].errorbar(compromise.crops[i].DM.index, compromise.crops[i].DM.WStem,
                   yerr=compromise.crops[i].DM.STDStem, color='mediumseagreen', fmt='o',
                   capsize=2, markersize=5)
    if i == 3:
        ax[i].errorbar(compromise.crops[i].Root.index, compromise.crops[i].Root.WRoot,
                       yerr=compromise.crops[i].Root.STDRoot, color='cornflowerblue', fmt='o',
                       capsize=2, markersize=5)
    ax[i].grid(True)
    ax[i].set_yticks([0, 5, 10, 15], fontname='monospace')
    ax[i].text(compromise.crops[i].starting_date + datetime.timedelta(days=76), 15.2,
               name_crop[i], fontname='monospace')
    plot_year = compromise.crops[i].starting_date.year
    ax[i].set_xlim([datetime.date(plot_year, 2, 24), datetime.date(plot_year, 8, 7)])
ax[0].legend(prop={'family': 'monospace'})
fig.tight_layout()
if saving == True:
    fig.savefig('Fig_biomass_'+dataset+'_'+compromise.name+'.png')


#%% GETTING GAPFILLED DATA

file_dir = r'C:\Users\u226422\OneDrive - Universite de Liege\Documents\Measurements\Lonzee_data\L2'
data = pd.read_csv(file_dir+r'\FLX_BE-Lon_FLUXNET_2004-2020.csv', index_col='TIMESTAMP_END',
                   date_parser=pd.to_datetime, na_values="-9999").loc['2014-01-01 00:00:00':]
data2 = pd.read_csv(file_dir+r'\FLX_BE-Lon_FLUXNET_2021-2022.csv', index_col='TIMESTAMP_END',
                    date_parser=pd.to_datetime, na_values="-9999")
data3 = pd.read_csv(file_dir+r'\ONEFLUX\FLX_BE-Lon_FLUXNET2015_FULLSET_HH_2019-2024_1-5.csv',
                    index_col='TIMESTAMP_END', date_parser=pd.to_datetime, na_values="-9999")

dNEE1 = data['NEE_VUT_REF'].groupby(pd.Grouper(freq='D')).sum()*1800*12e-6
dNEE2 = data2['NEE_VUT_REF'].groupby(pd.Grouper(freq='D')).sum()*1800*12e-6
dLE1 = data['LE_CORR'].groupby(pd.Grouper(freq='D')).mean()
day = pd.read_csv(file_dir+r'\Warm_winter\FLX_BE-Lon_FLUXNET2015_FULLSET_DD_2004-2020_beta-3.csv',
                  index_col='TIMESTAMP', date_parser=pd.to_datetime,
                  na_values="-9999").loc['2014-01-01 00:00:00':]
day2 = pd.read_csv(file_dir+r'\Warm_winter_addon\FLX_BE-Lon_FLUXNET2015_FULLSET_DD_2021-2022_beta-3.csv',
                   index_col='TIMESTAMP', date_parser=pd.to_datetime, na_values="-9999")
DD_EBC = pd.read_csv(file_dir+r'\DD_EBC.csv', index_col='TIMESTAMP', date_parser=pd.to_datetime)
DD_EBC.loc[DD_EBC['LE_JOINTUNC_LD'].isna(), 'LE_CORR_LD'] = np.nan

#%% PLOTTING WITH GAPFILLED DATA (FIG 3 AND 4)

name_crop = ['SAH', 'TOB', 'SMA', 'SKY']
plt.rcParams["font.family"] = "monospace"
fig, axs = plt.subplots(4, 2, gridspec_kw={'width_ratios': [3, 1]}, dpi=300, figsize=(10, 10))
for i in range(len(compromise.crops)):
    axs[i, 0].plot(compromise.crops[i].simdailyLE.loc[compromise.crops[i].starting_day:],
                   color='grey', lw=1.5, label='model')
    axs[i, 0].plot(DD_EBC.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date,
                              'LE_CORR_LD'], color='red', lw=1, label='data')
    axs[i, 0].fill_between(DD_EBC.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date].index,
                          DD_EBC.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date, 'LE_CORR_LD'] -
                          DD_EBC.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date, 'LE_JOINTUNC_LD'],
                          DD_EBC.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date, 'LE_CORR_LD'] +
                          DD_EBC.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date, 'LE_JOINTUNC_LD'],
                          color='red', alpha=0.3)
    axs[i, 0].grid(True)
    axs[i, 0].set_ylim([-5, 305])
    plot_year = compromise.crops[i].starting_date.year
    axs[i, 0].set_xlim([datetime.date(plot_year, 2, 24), datetime.date(plot_year, 8, 7)])
    axs[i, 1].text(10, 720, 'RMSE = '+str(round(compromise.crops[i].RMSE_LE, 2)))
    axs[i, 1].text(10, 620, 'R$^2$ = '+str(round(compromise.crops[i].r_value_LE, 2)))
    axs[i, 0].set_ylabel(name_crop[i])
    axs[i, 1].plot([0, 800], [0, 800], color='black')
    axs[i, 1].set_ylim([-20, 820])
    axs[i, 1].scatter(compromise.crops[i].LE,
                      compromise.crops[i].simLE[compromise.crops[i].LE.index], color='grey', s=8)
    axs[i, 1].grid(True)
axs[0, 0].legend(prop={'family': 'monospace'})
fig.tight_layout()
if saving == True:
    fig.savefig('Fig_dailyLE_MM_'+dataset+'_'+compromise.name+'.png')

plt.rcParams["font.family"] = "monospace"
fig, axs = plt.subplots(4, 2, gridspec_kw={'width_ratios': [3, 1]}, dpi=300, figsize=(10, 10))
for i in range(len(compromise.crops)):
    axs[i, 0].plot(compromise.crops[i].simdailyNEE.loc[compromise.crops[i].starting_day:],
                   color='grey', lw=1.5, label='model')
    if i < 3:
        axs[i, 0].plot(dNEE1.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date],
                       color='red', lw=1, label='data')
        axs[i, 0].fill_between(dNEE1.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date].index,
                               dNEE1.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date] -
                               day.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date, 'NEE_VUT_REF_JOINTUNC'],
                               dNEE1.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date] +
                               day.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date, 'NEE_VUT_REF_JOINTUNC'],
                               color='red', alpha=0.3)
    else:
        axs[i, 0].plot(dNEE2.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date],
                       color='red', lw=1, label='data')
        axs[i, 0].fill_between(dNEE2.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date].index,
                               dNEE2.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date] -
                               day2.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date, 'NEE_VUT_REF_JOINTUNC'],
                               dNEE2.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date] +
                               day2.loc[compromise.crops[i].starting_date:compromise.crops[i].ending_date, 'NEE_VUT_REF_JOINTUNC'],
                               color='red', alpha=0.3)
    axs[i, 0].grid(True)
    axs[i, 0].set_ylim([-16, 6])
    plot_year = compromise.crops[i].starting_date.year
    axs[i, 0].set_xlim([datetime.date(plot_year, 2, 24), datetime.date(plot_year, 8, 7)])
    axs[i, 0].set_ylabel(name_crop[i])
    axs[i, 1].text(-43, 10, 'RMSE = '+str(round(compromise.crops[i].RMSE_NEE, 2)))
    axs[i, 1].text(-43, 2, 'R$^2$ = '+str(round(compromise.crops[i].r_value_NEE, 2)))
    axs[i, 1].plot([-45, 15], [-45, 15], color='black')
    axs[i, 1].scatter(compromise.crops[i].NEE,
                      compromise.crops[i].simNEE[compromise.crops[i].NEE.index], color='grey', s=8)
    axs[i, 1].set_yticks([-45, -30, -15, 0, 15], fontname='monospace')
    axs[i, 1].set_xticks([-45, -30, -15, 0, 15], fontname='monospace')
    axs[i, 1].grid(True)
axs[0, 0].legend(prop={'family': 'monospace'})
fig.tight_layout()
if saving == True:
    fig.savefig('Fig_dailyNEE_MM_'+dataset+'_'+compromise.name+'.png')

#%% HETEROTROPHIC RESPIRATION (FIG S1)

data3 = pd.read_csv(file_dir+r'\ONEFLUX\FLX_BE-Lon_FLUXNET2015_FULLSET_HH_2019-2024_1-5.csv',
                    index_col='TIMESTAMP_END', date_parser=pd.to_datetime, na_values="-9999")
Tair = pd.concat([data['TA_F'], data2['TA_F']]).groupby(pd.Grouper(freq='H')).mean()
Tsol1 = data['TS_F_MDS_1'].groupby(pd.Grouper(freq='H')).mean()
Tsol3 = data3['TS_F_MDS_1'].groupby(pd.Grouper(freq='H')).mean()
Tsol = pd.concat([Tsol1[:-1], Tsol3])
Tsol_dropped = Tsol.reset_index().drop_duplicates(subset='TIMESTAMP_END', keep='last').set_index('TIMESTAMP_END')

Q10_mean = 2.11
Q10_ci = (1.78, 2.44)
Q10_std = (Q10_ci[1] - Q10_ci[0]) / 4

R10_mean = 0.74
R10_ci = (0.59, 0.88)
R10_std = (R10_ci[1] - R10_ci[0]) / 4

n_samples = 1000

plt.rcParams["font.family"] = "monospace"
fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
ax = axs.ravel()

for i in range(len(compromise.crops)):
    year = compromise.crops[i].ending_date.year
    start = f'{year}-04-01 00:00:00'
    end = f'{year}-07-20 00:00:00'
    T_daily = Tsol_dropped.loc[start:end].resample('D').mean().squeeze()
    q10_samples = np.random.normal(loc=Q10_mean, scale=Q10_std, size=n_samples)
    r10_samples = np.random.normal(loc=R10_mean, scale=R10_std, size=n_samples)
    fT_samples = []
    for T in T_daily:
        ft_day = r10_samples * q10_samples ** ((T - 10) / 10)
        fT_samples.append(ft_day)

    fT_samples = np.array(fT_samples)  # shape: [n_days, n_samples]
    fT_mean = fT_samples.mean(axis=1)
    fT_std = fT_samples.std(axis=1)
    fT_mean_series = pd.Series(fT_mean, index=T_daily.index)
    fT_std_series = pd.Series(fT_std, index=T_daily.index)
    sim = compromise.crops[i].simdHR.loc[start:end]
    ax[i].plot(fT_mean_series, label='Suleau et al.', color='firebrick')
    ax[i].fill_between(fT_mean_series.index,
                       fT_mean_series - fT_std_series,
                       fT_mean_series + fT_std_series,
                       color='firebrick', alpha=0.3)
    ax[i].plot(sim, label='Model', color='grey')
    ax[i].xaxis.set_major_locator(mdates.MonthLocator())
    ax[i].grid(True)
    ax[i].text(compromise.crops[i].starting_date + datetime.timedelta(days=36), 2.6,
               name_crop[i], fontname='monospace')
ax[0].legend()
fig.tight_layout()
fig.savefig('Fig_Suleau_comparaison.png')


#%% COMPUTATION OF 1/WUE (HEATWAVES)

data3 = pd.read_csv(file_dir+r'\ONEFLUX\FLX_BE-Lon_FLUXNET2015_FULLSET_HH_2019-2024_1-5.csv',
                    index_col='TIMESTAMP_END', date_parser=pd.to_datetime, na_values="-9999")
flux1 = data[['NEE_VUT_REF', 'GPP_NT_VUT_REF', 'LE_CORR', 'H_CORR', 'WS', 'USTAR', 'TA_F_MDS',
              'VPD_F_MDS', 'SW_IN_F', 'LW_IN_F', 'SW_OUT', 'LW_OUT', 'G_F_MDS', 'PA_F', 'CO2_F_MDS']]
flux2 = data3[['NEE_VUT_REF', 'GPP_NT_VUT_REF', 'LE_CORR', 'H_CORR', 'WS', 'USTAR', 'TA_F_MDS',
               'VPD_F_MDS', 'SW_IN_F', 'LW_IN_F', 'SW_OUT', 'LW_OUT', 'G_F_MDS', 'PA_F', 'CO2_F_MDS']]
fluxes = pd.concat([flux1, flux2])

fluxes['VPD'] = fluxes['VPD_F_MDS']*100  # Pa
fluxes.loc['2015-06-30 00:00:00':'2015-07-01 23:30:00', 'flag'] = 2  # PRECIP 23-06
fluxes.loc['2017-06-19 00:00:00':'2017-06-22 23:30:00', 'flag'] = 3  # PRECIP 09-06
fluxes.loc['2019-04-18 00:00:00':'2019-04-22 23:30:00', 'flag'] = 4  # PRECIP 09-04
fluxes.loc['2022-05-17 00:00:00':'2022-05-18 23:30:00', 'flag'] = 8  # PRECIP 08-04
# Periods that could be suitable but it rained previously :
# 2015-06-04--05, 2019-06-01--02, 2019-06-17--18, 2022-04-17--19

fluxes.loc['2015-06-27 00:00:00':'2015-06-29 23:30:00', 'flag'] = 12
fluxes.loc['2015-07-03 00:00:00':'2015-07-05 23:30:00', 'flag'] = 12
fluxes.loc['2017-06-16 00:00:00':'2017-06-17 23:30:00', 'flag'] = 13
fluxes.loc['2017-06-23 00:00:00':'2017-06-24 23:30:00', 'flag'] = 13
fluxes.loc['2019-04-15 00:00:00':'2019-04-16 23:30:00', 'flag'] = 14
fluxes.loc['2019-04-24 00:00:00':'2019-04-25 23:30:00', 'flag'] = 14
fluxes.loc['2022-05-13 00:00:00':'2022-05-14 23:30:00', 'flag'] = 18
fluxes.loc['2022-05-20 00:00:00':'2022-05-21 23:30:00', 'flag'] = 18

fluxes.loc[(fluxes.index.hour > 14) | (fluxes.index.hour < 11), 'LE_CORR'] = np.nan
fluxes.loc[(fluxes.index.hour > 14) | (fluxes.index.hour < 11), 'GPP_NT_VUT_REF'] = np.nan
fluxes.loc[(fluxes.index.hour > 14) | (fluxes.index.hour < 11), 'VPD'] = np.nan
fluxes['1/WUE'] = fluxes['LE_CORR']/fluxes['GPP_NT_VUT_REF']

#%% BOXPLOTS (FIG 5)

vpd_bins = [0, 1500, 2000, np.inf]
vpd_labels = ['<1500', '1500-2000', '>2000']
fluxes['VPD_group'] = pd.cut(fluxes['VPD'], bins=vpd_bins, labels=vpd_labels)

data_box = []
positions = []
group_centers = []
width = 0.3  # spacing between boxplots within a group
group_gap = 1
pos = 1

for vpd_group in vpd_labels:
    group_data = []
    for flag_cond, flag_label in [(lambda x: x <= 10, 'HW'), (lambda x: x > 10, 'CT')]:
        values = fluxes.loc[(fluxes['VPD_group'] == vpd_group) & flag_cond(fluxes['flag']), '1/WUE']
        data_box.append(values)
        positions.append(pos)
        pos += width
    group_centers.append(np.mean(positions[-2:]))
    pos += group_gap

# plt.ioff()
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
box = ax.boxplot(data_box, positions=positions, patch_artist=True, widths=0.25,
                 medianprops=dict(color='red'))
colors = ['darksalmon', 'darkgray'] * len(vpd_labels)
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xticks(group_centers)
ax.set_xticklabels(vpd_labels)
ax.set_ylabel('LE / GPP')
ax.grid(axis='y')
ax.set_xlabel('VPD (Pa)')
legend_patches = [
    plt.Line2D([0], [0], color='darksalmon', lw=8, label='HW'),
    plt.Line2D([0], [0], color='darkgray', lw=8, label='CT')
]
ax.legend(handles=legend_patches, loc='upper left')
fig.savefig('Fig_boxplot3.png')

for vpd_group in vpd_labels:
    group_data = fluxes[fluxes['VPD_group'] == vpd_group]
    data_flag_low = group_data[group_data['flag'] <= 10]['1/WUE']
    data_flag_high = group_data[group_data['flag'] > 10]['1/WUE']
    stat, p = mannwhitneyu(data_flag_low, data_flag_high, alternative='two-sided')
    print(f"VPD Group {vpd_group}: Mann-Whitney U test p-value = {p:.4f}")
    stat, p, med, tbl = median_test(data_flag_low, data_flag_high)
    print(f"VPD Group {vpd_group}: Median test p-value = {p:.4f}")
    stat, p = ttest_ind(data_flag_low, data_flag_high, equal_var=False)
    print(f"VPD Group {vpd_group}: t-test p-value = {p:.4f}")


#%% ENERGY (FIG S2)

data_energy = data.loc['2019-04-18 00:00:00':'2019-04-23 00:00:00', ['LE_CORR', 'LE_F_MDS_QC',
                                                                     'H_CORR', 'H_F_MDS_QC']]

i=2
plt.rcParams["font.family"] = "monospace"
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
axs[0].plot(compromise.crops[i].simLE.loc['2019-04-18 00:00:00':'2019-04-23 00:00:00'],
            color='grey', label='model')
axs[0].scatter(data_energy.loc[data_energy['LE_F_MDS_QC'] == 0].index,
               data_energy.loc[data_energy['LE_F_MDS_QC'] == 0, 'LE_CORR'], color='firebrick',
               label='data', s=8)
axs[1].plot(compromise.crops[i].simLE.loc['2019-04-18 00:00:00':'2019-04-23 00:00:00'],
            color='grey')
axs[1].scatter(data_energy.loc[data_energy['H_F_MDS_QC'] == 0].index,
               data_energy.loc[data_energy['H_F_MDS_QC'] == 0, 'H_CORR'], color='firebrick', s=8)
axs[0].legend(loc='upper left')
fig.tight_layout()
fig.savefig('Fig_energy_fluxes.png')