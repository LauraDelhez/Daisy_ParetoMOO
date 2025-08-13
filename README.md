[![DOI](https://zenodo.org/badge/1037199630.svg)](https://doi.org/10.5281/zenodo.16836123)
# Daisy_ParetoMOO
All python scripts and input files needed to run SMPSO on Daisy

## Running SMPSO
Python script run_SMPSO.py generates multiple parameters sets, passing them through funcs_multi.py which runs different cropping seasons using these parameters sets. Relative RMSE is returned to run_SMPSO, guiding the new generation of parameters sets.

## Input files
Daisy input files contain weather data (.dwf), groundwater table depth (.gwt) and parameterisation (.dai) for each cropping season.
