import kicker
import astropy.units as u
import numpy as np
from kicker import galaxy, grid
import pandas as pd
import gala.potential as gp
import gala.dynamics as gd

import matplotlib.pyplot as plt

met_grid = np.logspace(np.log10(1e-4), np.log10(0.03), 15)
met_grid = np.round(met_grid, 8)
met_grid = met_grid
grid_path = '/mnt/home/kbreivik/ceph/DWD_Z_alpha25/'
kstar1s = ['10', '11' ,'11' ,'12']
kstar2s = ['10', '10' ,'11' ,'10_12']
#

for kstar1, kstar2 in zip(kstar1s, kstar2s):
    p = kicker.pop.EvolvedPopulation(grid_path=grid_path, kstar1=kstar1, kstar2=kstar2, met_grid=met_grid, galaxy_model=galaxy.Frankel2018)

    p.create_population()
    
    plt.hist(p._mergers.porb_today)
    plt.show()
    #p.get_grid_pop()

#
#met_grid = met_grid[::6]
#inner_bins = np.array([met_grid[i] + (met_grid[i+1] - met_grid[i]) / 2 for i in range(len(met_grid) - 1)])
#bins = np.concatenate(([met_grid[0]], inner_bins, [met_grid[-1]]))
#
#
#dat_store_path = '/mnt/home/kbreivik/ceph/DWD_Z_alpha25/'
#paths_kstar = get_paths(met_grid, kstar1s, kstar2s, dat_store_path)
#
### initialize the metallicity weights
#w_Z = find_metallicity_weights(bins, galaxy.Frankel2018)
#
### galactic disk components
#mlow = 2.585e10
#mhi = 2.585e10
#m_tot_Z = (mlow + mhi) * w_Z
#
#m_DWD_form_kstar = []
#n_DWD_form_kstar = []
#n_samp_kstar = []
#
#gx_model = galaxy.Frankel2018
#gx_potential = gp.MilkyWayPotential()
#v_dispersion=5 * u.km / u.s
#
##iterate over DWD type to get the number of DWDs to 
##sample at each metallicity for each kstar type
#init_Gx_list = []
#for paths in paths_kstar:
#    m_DWD, n_DWD = get_DWD_mass_num(met_grid, paths)
#    m_DWD_form_kstar.append(m_DWD)
#    n_DWD_form_kstar.append(n_DWD)
#    n_DWD_samp = np.sum(n_DWD * m_tot_Z / m_DWD)
#    init_Gx = gx_model(size=int(n/weight), 
#                       components=["low_alpha_disc", "high_alpha_disc"],
#                       component_masses=[2.585e10, 2.585e10])
#    init_Gx["grid_met"] = init_Gx.Z
#    init_Gx.loc[init_Gx["grid_met"] < 1e-4, "grid_met"] = 1e-4
#    init_Gx.loc[init_Gx["grid_met"] > 0.03, "grid_met"] = 0.03
#
#    for met, path, ii in enumerate(bins[1:], paths):
#        init_Gx_select = init_Gx.loc[(grid_met.Z < met) & (init_Gx.Z >= bins[ii])]
#        
#        n_DWD_sample = init_Gx_select.size
#        DWDs = pd.read_hdf(path, key='conv')
#        DWD_sample = DWDs.sample(n_DWD_sample, replace=True)
#        
#        # Filter out any binaries which won't make a DWD by the present day
#        # based on the age from the Galactic sample
#        DWD_sample = DWD_sample.loc[DWD_sample.tphys.values < init_Gx_select.age]
#        init_Gx_select = init_Gx_select[DWD_sample.tphys.values < init_Gx_select.age]
#        
#        t_evol = init_Gx_select.age - DWD_sample.tphys
#        # Calculate the final separations and RLOF times for each binary
#        sep_f, t_RLOF = grid.WD_GW_evol(t_evol=t_evol, 
#                                        m1=DWD_sample.mass_1 * u.Msun, 
#                                        m2=DWD_sample.mass_2 * u.Msun, 
#                                        sep_i=DWD_sample.sep * u.Rsun)
#        
#        # work out the initial velocities of each binary
#        vel_units = u.km / u.s
#
#        # calculate the Galactic circular velocity at the initial positions
#        v_circ = galactic_potential.circular_velocity(q=[init_Gx_select.positions.x,
#                                                         init_Gx_select.positions.y,
#                                                         init_Gx_select.positions.z]).to(vel_units)
#
#        # add some velocity dispersion
#        v_R, v_T, v_z = np.random.normal([np.zeros_like(v_circ), v_circ, np.zeros_like(v_circ)],
#                                         v_dispersion.to(vel_units) / np.sqrt(3),
#                                         size=(3, init_Gx_select.size))
#        v_R, v_T, v_z = v_R * vel_units, v_T * vel_units, v_z * vel_units
#        
#        
#        
#    
##Z_ind_sort = np.digitize(initial_gx.Z, bins=met_grid, right=False)
##Z_frac = []
##for ii in range(len(met_grid)):
##    Z_frac.append(len(initial_gx.Z[Z_ind_sort == ii])/size)
##    print(met_grid[ii], np.sort(initial_gx.Z[Z_ind_sort == ii]))
#