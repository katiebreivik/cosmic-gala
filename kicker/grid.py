import numpy as np
import pandas as pd
from legwork import evol, utils
from astropy import units as u


__all__ = ["get_cosmic_datfiles", "WD_RLOF", "rad_WD", "t_of_a", 
           "WD_GW_evol", ]



def get_cosmic_datfiles(mets, k1, k2, dat_store_path):
    """Sets up COSMIC datafile names for kstar types and metallicity grid
    
    Parameters
    ----------
    mets : `float`
        Metallicity grid
    k1 : `str`
        primary kstar value
    k2 : `str`
        secondary kstar value(s)
    dat_store_path : `str`
        absolute path to data
        
    Returns
    -------
    datfiles : `list`
        datfiles for kstars and metallicities on grid
    """
    datfiles = []
    for m in mets: 
        datfiles.append(dat_store_path+f"dat_kstar1_{k1}_kstar2_{k2}_SFstart_13700.0_SFduration_0.0_metallicity_{m}.h5")
    return datfiles


def WD_RLOF(m1, m2, r1, r2):
    """
    Finds separation when lower mass WD overflows its
    Roche Lobe. Taken from Eq. 23 in "Binary evolution in a nutshell" 
    by Marc van der Sluys, which is an approximation of a fit
    done of Roche-lobe radius by Eggleton (1983).
    
    Parameters
    ----------
    m1 : numpy.array
        component mass
    m2 : numpy.array
        component mass
    r1 : numpy.array
        component radius
    r2 : numpy.array
        component radius

    Returns
    -------
    a : numpy.array
        separation
    """
    # calculat the mass ratio part
    m1 = m1.value
    m2 = m2.value
    primary_mass = np.where(m1>m2, m1, m2)
    secondary_mass = np.where(m1>m2, m2, m1)
    q = secondary_mass / primary_mass
    num = 0.49 * q ** (2/3)
    denom = 0.6 * q ** (2/3) + np.log(1 + q ** (1/3))
    
    # assign the secondary radius based on WD sizes
    # which go up with decreasing mass
    secondary_radius = np.where(m1>m2, r2, r1)
    R2 = secondary_radius
    
    # calculate a at RLOF
    a = denom * R2 / num
    
    return a

def rad_WD(M):
    """
    Calculates the radius of a WD as a function of mass M in solar masses.
    Taken from Eq. 91 in Hurley et al. (2000), from Eq. 17 in Tout et al. (1997)
    
    Parameters
    ----------
    M : numpy.array
        masses of the WDs
    
    Returns
    -------
    rad : numpy.array
        radii of the WDs
    """
    M = M.to(u.Msun).value
    M_ch = 1.44
    R_NS = 1.4e-5*np.ones(len(M))
    A = 0.0115 * np.sqrt((M_ch/M)**(2/3) - (M/M_ch)**(2/3))
    rad = np.max(np.array([R_NS, A]), axis=0)
    return rad * u.Rsun


def t_of_a(m1, m2, sep_i, sep_f):
    """
    Uses Peters(1964) equation (5.9) for circular binaries to find separation.
    as a function of time.

    Parameters
    ----------
    m1 : numpy.array
        component mass in astropy mass units
    m2 : numpy.array
        component mass in astropy mass units
    sep : numpy.array
        initial binary separation in astropy length units
    sep_f : numpy.array
        final binary separation astropy length units
    Returns
    -------
    t : numpy.array
        evolution time to sep_f in astropy time units
    """
    beta = utils.beta(m1, m2)
    a_i = sep_i
    a_f = sep_f
    t = (a_i ** 4 - a_f ** 4) / 4 / beta
    
    return t

def WD_GW_evol(t_evol, m1, m2, sep_i):
    """
    Evolve an initial population of binary WD's using
    GW radiation, enforcing circular binaries only
    
    Parameters
    ----------
    t_evol : numpy.array
        evolution time with astropy time unit
    m1 : numpy.array
        mass of initially more massive ZAMS star with astropy mass unit
    m2 : numpy.array
        mass of initially less massive ZAMS star with astropy mass unit
    sep_i : numpy.array
        initial separation with astropy length unit
                                
    RETURNS
    ----------------------
    sep_f : numpy.array
        final separation after t_evol with astropy length unit
        
    t_RLOF : numpy.array
        time for which binary will fill Roche lobe with astropy time unit
        
    RLOF_mask : numpy.array of bool
        mask for binaries which fill their Roche lobe's during t_evol
    """
    # First get WD radii
    r1 = rad_WD(M=m1)
    r2 = rad_WD(M=m2)
    
    # Evolve the WD binaries from Peters GW emission
    m_c = utils.chirp_mass(m1, m2)
    f_orb_i = utils.get_f_orb_from_a(sep_i, m1, m2)
    f_orb_final = evol.evolve_f_orb_circ(f_orb_i, m_c, t_evol)
    sep_f = utils.get_a_from_f_orb(f_orb_final, m1, m2)
    
    # check the RLOF separation
    sep_RLOF = WD_RLOF(m1, m2, r1, r2)
    
    # calculate the RLOF times
    t_RLOF = t_of_a(m1, m2, sep_i, sep_RLOF).to(u.Myr)

    return sep_f, f_orb_final, t_RLOF