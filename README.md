# Daisy_ParetoMOO
All python scripts and input files needed to run SMPSO on Daisy

## Running SMPSO
Python script run_SMPSO.py generates multiple parameters sets, passing them through funcs_multi.py which runs different cropping seasons using these parameters sets. Relative RMSE is returned to run_SMPSO, guiding the new generation of parameters sets.
