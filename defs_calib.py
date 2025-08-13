# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:23:05 2023

@author: u226422
"""

from pydaisy import Daisy
import os


def photosynthesis(d, name, param):
    d.Input['defphotosynthesis'][name].setvalue(param)
    return


def stomatal(d, name, param):
    var = name.rsplit(sep='_', maxsplit=1)[0]
    d.Input['defphotosynthesis']['Stomatacon'][var].setvalue(param)
    return

def ssoc(d, name, param):
    if name == 'b_Leun':
        d.Input['defcolumn']['Bioclimate']['svat']['b_leuning'].setvalue(param)
    else:
        d.Input['defcolumn']['Bioclimate']['svat'][name].setvalue(param)
    return

def development(d, name, param):
    d.Input['defcrop']['Devel'][name].setvalue(param)
    return


def radiation(d, name, param):
    var = name.rsplit(sep='_', maxsplit=1)[0]
    d.Input['defcolumn']['Bioclimate']['net_radiation'][var].setvalue(param)
    return


def difrad(d, name, param):
    d.Input['defcolumn']['Bioclimate']['difrad'][name].setvalue(param)
    return


def surface(d, name, param):
    d.Input['defcolumn']['Surface'][name].setvalue(param)
    return


def aquitard(d, name, param):
    d.Input['defcolumn']['Groundwater'][name].setvalue(param)
    return


def canopy(d, name, param):
    if name == 'k_net':
        var = 'EPext'
        d.Input['defcrop']['Canopy'][var].setvalue(param)
    else:
        d.Input['defcrop']['Canopy'][name].setvalue(param)
    return


def raddist(d, name, param):
    d.Input['defcolumn']['Bioclimate']['raddist'][name].setvalue(param)
    return


def partitioning(d, name, param):
    d.Input['defcrop']['Partit'][name].setvalue(param)
    return


def root(d, name, param):
    d.Input['defcrop']['Root'][name].setvalue(param)
    return


def rubisco(d, name, param):
    d.Input['defcrop']['RubiscoN']['fraction'].setvalue(param)
    return


def hydraulic(d, name, param):
    var = name.rsplit(sep='_', maxsplit=1)[0]
    z = int(name.rsplit(sep='_', maxsplit=1)[1])
    d.Input['defhorizon'][z]['hydraulic'][var].setvalue(param)
    return


def production(d, name, param):
    d.Input['defcrop']['Prod'][name].setvalue(param)
    return
