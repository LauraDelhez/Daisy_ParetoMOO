# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:12:35 2025

@author: u226422
"""

from pydaisy import Daisy
import defs_calib
import os
import warnings
import pandas as pd
import numpy as np
import datetime


def CalcRMSE(e):
    e_without_nan = e[~np.isnan(e)]
    return np.sqrt((e_without_nan**2).mean())


class Crop:
    def __init__(self, name):
        self.name = name
        self.time_offset = datetime.timedelta(minutes=30)

    def get_biomass(self):
        DM = pd.read_excel(os.getcwd()+r'\observations.xlsx',
                           usecols=['Date', 'WLeaf_Stem', 'WLeaf', 'WStem', 'WSOrg'],
                           sheet_name='WShoot_'+self.name, index_col='Date',
                           date_parser=pd.to_datetime)
        self.DM = DM/100
        if self.name == 'sky':
            Root = pd.read_excel(os.getcwd()+r'\observations.xlsx', usecols=['Date', 'WRoot'],
                                 sheet_name='WRoot_sky', index_col='Date',
                                 date_parser=pd.to_datetime)
            self.Root = Root/100
        return

    def read_fluxes(self, sheetname, usecols, file_path=os.getcwd()+r'\observations.xlsx'):
        data = pd.read_excel(file_path, sheet_name=sheetname, index_col='Date',
                             date_parser=pd.to_datetime, usecols=usecols)
        data.index = data.index + self.time_offset
        data = data.fillna(np.inf).groupby(pd.Grouper(freq='H')).mean().replace(np.inf, np.nan)
        data.dropna(inplace=True)
        return data

    def get_fluxes(self):
        self.NEE = self.read_fluxes('NEE_'+self.name, ['Date', 'NEE'])
        self.LE = self.read_fluxes('LE_'+self.name, ['Date', 'LE_CORR'])
        return


def measured_data():
    # DON'T FORGET TO DIVIDE BY 100 !
    crops = [Crop(i) for i in iter(['sah', 'tob', 'sma', 'sky'])]
    for n in range(len(crops)):
        crops[n].get_biomass()
        crops[n].get_fluxes()
    return crops


def writing_nomore_common_dai(param_names, x, main_dir, model_dir, name):
    dai = Daisy.DaisyModel(main_dir+r'\wheat_LTO.dai')
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
    defs_calib.radiation(dai, param_names[20], x[20])  # A_brunt
    defs_calib.radiation(dai, param_names[21], x[21])  # B_brunt
    defs_calib.ssoc(dai, param_names[22], x[22])  # epsilon_leaf
    defs_calib.raddist(dai, param_names[27], x[27])  # sigma_NIR
    defs_calib.root(dai, param_names[28], x[28])  # PenPar2
    defs_calib.hydraulic(dai, param_names[33], x[33])  # K_sat 1
    defs_calib.canopy(dai, param_names[42], x[42])  # k_net (EPext)
    defs_calib.surface(dai, param_names[51], x[51])  # EpFactor (Ke)
    defs_calib.aquitard(dai, param_names[52], x[52])  # K_aquitard
    defs_calib.aquitard(dai, param_names[53], x[53])  # Z_aquitard
    defs_calib.canopy(dai, param_names[54], x[54])  # EpFac (Kc)
    dai.save_as(model_dir+r'\wheat_LTO.dai')
    return

def running_daisy(param_names, crops, x, thread_id):
    main_dir = os.getcwd()
    model_dir = main_dir+r'\runs\setup_'+str(thread_id)
    diff_DM = None
    for n in range(len(crops)):
        name = crops[n].name
        writing_nomore_common_dai(param_names, x, main_dir, model_dir, name)
        d = Daisy.DaisyModel(main_dir+r'\Lonzee_'+name+'.dai')
        new_dir = '"C:/Users/u226422/Documents/CALIB/jMetal/runs/setup_'+str(thread_id)+'"'
        d.Input['directory'].setvalue(new_dir)
        d.save_as(model_dir+r'\setup_test.dai')
        Daisy.DaisyModel.path_to_daisy_executable = r'C:\Program Files\Daisy 2.11\bin\daisy.exe'
        d.run()
        max_retries = 4
        retries = 0
        success = False
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            while not success and retries < max_retries:
                try:
                    dlf2 = pd.read_csv(model_dir+r'\bioclim_'+name+'.dlf', sep='\t', skiprows=15,
                                       header=[0, 1])
                    dlf2.columns = dlf2.columns.droplevel(1)
                    dlf2.rename(columns={"mday": "day"}, inplace=True)
                    dlf2.set_index(pd.to_datetime(dlf2[['year', 'month', 'day', 'hour']]), inplace=True)
                    NEE = dlf2['Bioinc_CO2'] + dlf2['OM_CO2'].values - dlf2['NetPhotosynthesis'].values
                    simNEE = NEE.loc[crops[n].NEE.index]
                    success = True
                except KeyError:
                    retries += 1
                    d.run()
                    # print(f"Erreur de convergence (KeyError) - Tentative {retries}/{max_retries}")
            # Reading crop yield [t ha^-1]
            dlf = pd.read_csv(model_dir+r'\crop_'+name+'.dlf', sep='\t', skiprows=16, header=[0, 1])
            dlf.columns = dlf.columns.droplevel(1)
            dlf.rename(columns={"mday": "day"}, inplace=True)
            dlf.set_index(pd.to_datetime(dlf[['year', 'month', 'day', 'hour']]), inplace=True)
            simWSOrg = dlf.loc[crops[n].DM.index, 'WSOrg']
            simWLeaf = dlf.loc[crops[n].DM.index, 'WLeaf']
            simWStem = dlf.loc[crops[n].DM.index, 'WStem']
            if name == 'sky':
                simWRoot = dlf.loc[crops[n].Root.index, 'WRoot']
            # Reading NEE [µmol m^-2 s^-1] and LE [W m^-2]
            dlf2 = pd.read_csv(model_dir+r'\bioclim_'+name+'.dlf', sep='\t', skiprows=15, header=[0, 1])
            dlf2.columns = dlf2.columns.droplevel(1)
            dlf2.rename(columns={"mday": "day"}, inplace=True)
            dlf2.set_index(pd.to_datetime(dlf2[['year', 'month', 'day', 'hour']]), inplace=True)
            NEE = dlf2['Bioinc_CO2'] + dlf2['OM_CO2'].values - dlf2['NetPhotosynthesis'].values
            simNEE = NEE.loc[crops[n].NEE.index]
            simLE = dlf2.loc[crops[n].LE.index, 'total_ea']
            # Computing the difference between measurements and outputs
            e_WSOrg = simWSOrg.to_numpy() - crops[n].DM.WSOrg.to_numpy()
            e_WLeaf = simWLeaf.to_numpy() - crops[n].DM.WLeaf.to_numpy()
            e_WStem = simWStem.to_numpy() - crops[n].DM.WStem.to_numpy()
            e_WLeaf_Stem = (simWLeaf.to_numpy() + simWStem.to_numpy()) - crops[n].DM.WLeaf_Stem.to_numpy()
            if name == 'sky':
                e_WRoot = simWRoot.to_numpy() - crops[n].Root.WRoot.to_numpy()
                e_DM = np.concatenate((e_WSOrg, e_WLeaf, e_WStem, e_WLeaf_Stem, e_WRoot))
            else:
                e_DM = np.concatenate((e_WSOrg, e_WLeaf, e_WStem, e_WLeaf_Stem))
            e_NEE = simNEE.to_numpy() - crops[n].NEE.to_numpy().flatten()
            e_LE = simLE.to_numpy() - crops[n].LE.to_numpy().flatten()
            if diff_DM is None:
                diff_DM = e_DM
                diff_NEE = e_NEE
                diff_LE = e_LE
            else:
                diff_DM = np.concatenate((diff_DM, e_DM))
                diff_NEE = np.concatenate((diff_NEE, e_NEE))
                diff_LE = np.concatenate((diff_LE, e_LE))
    RMSE1 = CalcRMSE(diff_DM)
    RMSE2 = CalcRMSE(diff_NEE)
    RMSE3 = CalcRMSE(diff_LE)
    AGB_all = pd.concat([crop.DM.add_suffix(f'_{crop.name}') for crop in crops], axis=1)
    DM_all = pd.concat([AGB_all, crops[-1].Root.add_suffix('_sky')], axis=1)
    DM_mean = DM_all.mean().mean()
    NEE_all = pd.concat([crop.NEE.rename(columns={'NEE': crop.name}) for crop in crops], axis=1)
    NEE_mean = np.abs(NEE_all.mean().mean())
    LE_all = pd.concat([crop.LE.rename(columns={'LE': crop.name}) for crop in crops], axis=1)
    LE_mean = LE_all.mean().mean()
    rRMSE1 = RMSE1/DM_mean
    rRMSE2 = RMSE2/NEE_mean
    rRMSE3 = RMSE3/LE_mean
    return [rRMSE1, rRMSE2, rRMSE3]
