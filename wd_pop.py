import kicker
import astropy.units as u
import numpy as np
from kicker import galaxy
import pandas as pd


def find_metallicity_weights(met_bins, gx_model, SAMPLESIZE = 1000000):
    
    initial_gx = gx_model(
    size=SAMPLESIZE, components=["low_alpha_disc", "high_alpha_disc"],
    component_masses=[2.585e10, 2.585e10]
    )

    # adjust the sample so everything falls inside the compas grid
    sampled_Z = initial_gx.Z
    sampled_Z[sampled_Z > np.max(met_bins)] = np.max(met_bins)
    sampled_Z[sampled_Z < np.min(met_bins)] = np.min(met_bins)
    
    # create a histogram on the grid and divide by the number of samples to find the weights
    h, _ = np.histogram(sampled_Z, bins=met_bins)
    w_Z = h / SAMPLESIZE
    return w_Z

def get_paths(met_grid, kstar1s, kstar2s, dat_store_path):
    paths = []
    for k1, k2 in zip(kstar1s, kstar2s):
        datfiles = []
        for m in met_grid:
            datfiles.append(dat_store_path+f"/dat_kstar1_{k1}_kstar2_{k2}_SFstart_13700.0_SFduration_0.0_metallicity_{m}.h5")
        paths.append(datfiles)
    return paths

def get_DWD_mass_num(met_grid, paths):
    m_formed = []
    n_formed = []
    for path in paths:
        m_formed.append(pd.read_hdf(path, key='mass_stars').iloc[-1])
        n_formed.append(len(pd.read_hdf(path, key='conv')))
    
    return m_formed, n_formed

def get_initGx(n_samp, met_grid, w_Z, gx_model):
    met_grid = np.append(0.0, met_grid)
    met_grid = np.append(met_grid, 0.1)
    met_lows = met_grid[:-1]
    met_his = met_grid[1:]
    init_gx_list = []
    for n, met_lo, met_hi, weight in zip(n_samp, met_lows, met_his, w_Z):
        ## need to get the relative mass fraction formed in each metallicity bin
        initial_gx = gx_model(
            size=int(n/weight), components=["low_alpha_disc", "high_alpha_disc"],
            component_masses=[2.585e10, 2.585e10]            
        )
        initial_gx = initial_gx[(initial_gx.Z <= met_hi) & (initial_gx.Z > met_lo)]
        init_gx_list.append(initial_gx)

    for initgx, n in zip(init_gx_list, n_samp):
        print(initgx.size, n)
    return init_gx_list

met_grid = np.logspace(np.log10(1e-4), np.log10(0.03), 15)
met_grid = np.round(met_grid, 8)

inner_bins = np.array([met_grid[i] + (met_grid[i+1] - met_grid[i]) / 2 for i in range(len(met_grid) - 1)])
bins = np.concatenate(([met_grid[0]], inner_bins, [met_grid[-1]]))

kstar1s = ['10', '11' ,'11' ,'12']
kstar2s = ['10', '11' ,'11' ,'10_12']

dat_store_path = '/mnt/home/kbreivik/ceph/DWD_Z_alpha25/'
paths_kstar = get_paths(met_grid, kstar1s, kstar2s, dat_store_path)

## initialize the metallicity weights
w_Z = find_metallicity_weights(bins, galaxy.Frankel2018)

## galactic disk components
mlow = 2.585e10
mhi = 2.585e10
m_tot_Z = (mlow + mhi)  * w_Z

m_DWD_form_kstar = []
n_DWD_form_kstar = []
n_samp_kstar = []
for paths in paths_kstar:
    m_DWD, n_DWD = get_DWD_mass_num(met_grid, paths)
    m_DWD_form_kstar.append(m_DWD)
    n_DWD_form_kstar.append(n_DWD)
    n_samp_kstar.append(n_DWD * m_tot_Z / m_DWD)


init_Gx_list = get_initGx(n_samp_kstar[0], met_grid, w_Z, galaxy.Frankel2018)            


#Z_ind_sort = np.digitize(initial_gx.Z, bins=met_grid, right=False)
#Z_frac = []
#for ii in range(len(met_grid)):
#    Z_frac.append(len(initial_gx.Z[Z_ind_sort == ii])/size)
#    print(met_grid[ii], np.sort(initial_gx.Z[Z_ind_sort == ii]))
