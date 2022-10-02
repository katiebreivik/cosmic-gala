import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const
from dustmaps.bayestar import BayestarQuery
from isochrones.mist.bc import MISTBolometricCorrectionGrid


def get_log_g(mass, radius):
    """Computes log of the surface gravity in cgs

    Parameters
    ----------
    mass : `Astropy Quantity`
        Mass of the star
    radius : `Astropy Quantity`
        Radius of the star

    Returns
    -------
    log g : `Astropy Quantity`
        Log of the surface gravity in cgs
    """
    g = const.G * mass / radius**2

    # avoid division by zero errors (for massless remnants)
    with np.errstate(divide='ignore'):
        return np.log10(g.cgs.value)


def get_absolute_bol_lum(lum):
    """Computes the absolute bolometric luminosity

    Parameters
    ----------
    lum : `Astropy Quantity`
        Luminosity of the star

    Returns
    -------
    M_bol : `float/array`
        Absolute bolometric magnitude
    """
    log_lum = np.log10(lum.to(u.Lsun).value)
    M_bol = 4.75 - 2.7 * log_lum
    return M_bol


def get_apparent_mag(M_abs, distance):
    """Convert absolute magnitude to apparent magnitude

    Parameters
    ----------
    M_abs : `float/array`
        Absolute magnitude
    distance : `float/array`
        Distance

    Returns
    -------
    m_app : `float/array`
        Apparent magnitude
    """
    finite_distance = np.isfinite(distance)
    m_app = np.repeat(np.inf, len(distance))
    m_app[finite_distance] = M_abs[finite_distance] + 5 * np.log10(distance[finite_distance] / (10 * u.pc))
    return m_app


def get_absolute_mag(m_app, distance):
    """Convert apparent magnitude to absolute magnitude

    Parameters
    ----------
    M_abs : `float/array`
        Apparent magnitude
    distance : `float/array`
        Distance

    Returns
    -------
    m_app : `float/array`
        Absolute magnitude
    """
    M_abs = m_app - 5 * np.log10(distance / (10 * u.pc))
    return M_abs


def add_mags(*mags, remove_nans=True):
    """Add any number of magnitudes

    Parameters
    ----------
    *mags : `list/np.array/float/int`
        A series of magnitudes. If arrays are given then all must have the same length. If a mixture of single
        values and arrays are given then the single values will be added to each array element
    remove_nans : `bool`, optional
        Whether to remove NaNs from the total (if not the total will be NaN), by default True

    Returns
    -------
    total_mag : `float/array`
        Total magnitude

    Raises
    ------
    ValueError
        If any magnitude is not a list, float, int or numpy array
    AssertionError
        If any magnitude array has a different length from another
    """
    total_mag = 0

    # convert lists to numpy arrays
    mags = list(mags)
    for i in range(len(mags)):
        if isinstance(mags[i], list):
            mags[i] = np.array(mags[i])
        if isinstance(mags[i], int):
            mags[i] = float(mags[i])

    # check for dodgy input
    if isinstance(mags[0], (list, np.ndarray)):
        length = len(mags[0])
        for mag in mags[1:]:
            assert len(mag) == length, ("All magnitude arrays must have the same length - one array is of "
                                        f"length {length} but another is {len(mag)}")

    # go through each supplied magnitude
    for mag in mags:
        if not isinstance(mag, (np.ndarray, float, int)):
            raise ValueError(("All `mag`s must either be a list, numpy array, float or an int. Unfortunately "
                              f"for us both, what you have given me is of type `{type(mag).__name__}`..."))

        # compute the default additional magnitude
        additional = 10**(-mag * 0.4)

        # if you want to remove any NaNs then do so
        if remove_nans:
            if isinstance(mag, np.ndarray):
                additional[np.isnan(mag)] = 0.0
            elif np.isnan(mag):
                additional = 0.0
        total_mag += additional

    # hide divide by zero errors (since inf magnitude is correct when all mags are NaN)
    with np.errstate(divide='ignore'):
        return -2.5 * np.log10(total_mag)


def get_extinction(coords):
    """Calculates the visual extinction values for a set of coordinates using the dustmaps.bayestar query

    Parameters
    ----------
    coords : `Astropy.coordinates.SkyCoord`
        The coordinates at which you wish to calculate extinction values

    Returns
    -------
    Av : `float/array`
        Visual extinction values for each set of coordinates
    """
    bayestar = BayestarQuery(max_samples=2, version='bayestar2019')
    ebv = bayestar(coords, mode='random_sample')
    Av = 3.2 * ebv
    return Av


def get_phot(final_bpp, final_coords, filters):
    """Computes photometry subject to dust extinction using the MIST boloemtric correction grid

    Parameters
    ----------
    final_bpp : `pandas.DataFrame`
        A dataset of COSMIC binaries at present day - must include these columns [TODO]
    final_coords : `tuple Astropy.coordinates.SkyCoord`
        Final positions and velocities of the binaries at present day. First entry is for binaries or the
        primary in a disrupted system, second entry is for secondaries in a disrupted system.
    filters : `list of str`
        Which filters to compute photometry for

    Returns
    -------
    photometry : `pandas.DataFrame`
        Photometry and extinction information for supplied COSMIC binaries in desired `filters`
    """
    # set up empty photometry table
    photometry = pd.DataFrame()

    # get extinction for bound binaries and primary of disrupted binaries
    photometry['Av_1'] = get_extinction(final_coords[0])

    # get extinction for secondaries of disrupted binaries (leave as np.inf otherwise)
    photometry['Av_2'] = np.repeat(np.inf, len(final_coords[1]))
    photometry.loc[final_bpp["sep"] < 0, "Av_2"] = get_extinction(final_coords[1][final_bpp["sep"] < 0])

    # ensure extinction remains in MIST grid range (<= 6) and is not NaN
    print('pop size before extinction cut: {}'.format(len(photometry)))
    photometry.loc[photometry.Av_1 > 6, ['Av_1']] = 6
    photometry.loc[photometry.Av_2 > 6, ['Av_2']] = 6
    photometry = photometry.fillna(6)
    print('pop size after extinction cut: {}'.format(len(photometry)))

    # get Fe/H using e.g. Bertelli+1994 Eq. 10
    Z_sun = 0.0142
    FeH = np.log10(final_bpp["metallicity"].values / Z_sun) / 0.977

    # set up MIST bolometric correction grid
    bc_grid = MISTBolometricCorrectionGrid(bands=filters)
    bc = [None, None]

    # for each star in the (possibly disrupted/merged) binary
    for ind in [1, 2]:
        # calculate the surface gravity
        final_bpp.insert(len(final_bpp.columns), f"log_g_{ind}",
                         get_log_g(mass=final_bpp[f"mass_{ind}"].values * u.Msun,
                                   radius=final_bpp[f"rad_{ind}"].values * u.Rsun))

        # get the bolometric corrections from MIST isochrones
        bc[ind - 1] = bc_grid.interp([final_bpp[f"teff_{ind}"].values, final_bpp[f"log_g_{ind}"].values,
                                      FeH, photometry[f"Av_{ind}"]],
                                     filters)

        # calculate the absolute bolometric magnitude and set any BH or massless remnants to invisible
        photometry[f"M_abs_{ind}"] = get_absolute_bol_lum(lum=final_bpp[f"lum_{ind}"].values * u.Lsun)
        photometry.loc[final_bpp[f"kstar_{ind}"].isin([14, 15]), f"M_abs_{ind}"] = np.inf

        # work out the distance (if the system is bound always use the first `final_coords` SkyCoord)
        distance = np.repeat(np.inf, len(final_bpp)) * u.kpc
        distance[final_bpp["sep"] < 0] = final_coords[ind - 1][final_bpp["sep"] < 0].icrs.distance
        distance[final_bpp["sep"] >= 0] = final_coords[0][final_bpp["sep"] >= 0].icrs.distance

        # convert the absolute magnitude to an apparent magnitude
        photometry[f"m_app_{ind}"] = get_apparent_mag(M_abs=photometry[f"M_abs_{ind}"].values,
                                                      distance=distance)

    # go through each filter
    for i, filter in enumerate(filters):
        # apply the bolometric corrections to the apparent magnitude of each star
        filter_mags = [photometry[f"m_app_{ind}"].values - bc[ind - 1][:, i] for ind in [1, 2]]

        # total the magnitudes (removing any NaNs)
        total_filter_mag = add_mags(*filter_mags, remove_nans=True)

        # default to assuming all systems are bound - in this case total magnitude is listed
        # in primary filter apparent mag, and secondary is non-existent
        photometry[f"{filter}_app_1"] = total_filter_mag
        photometry[f"{filter}_app_2"] = np.repeat(np.inf, len(photometry))

        # for disrupted systems, change the filter apparent magnitudes to the values for each individual star
        disrupted = final_bpp["sep"] < 0.0
        for ind in [1, 2]:
            photometry.loc[disrupted, f"{filter}_app_{ind}"] = filter_mags[ind - 1][disrupted]

        # for the G filter in particular, see which temperature/log-g is getting measured
        if filter == "G":
            # by default assume the primary is dominant
            photometry["teff_obs"] = final_bpp["teff_1"].values
            photometry["log_g_obs"] = final_bpp["log_g_1"].values

            # overwrite any values where the secondary is brighter
            secondary_brighter = (filter_mags[1] < filter_mags[0]) | (np.isnan(filter_mags[0])
                                                                      & ~np.isnan(filter_mags[1]))
            photometry["secondary_brighter"] = secondary_brighter
            photometry.loc[secondary_brighter, "teff_obs"] = final_bpp["teff_2"].values[secondary_brighter]
            photometry.loc[secondary_brighter, "log_g_obs"] = final_bpp["log_g_2"].values[secondary_brighter]

    return photometry
