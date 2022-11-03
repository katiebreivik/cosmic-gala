import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const
import astropy.coordinates as coords
from dustmaps.bayestar import BayestarQuery
from scipy.interpolate import interp1d

# HACK around the isochrone import to ignore warnings about Holoview and Multinest
import logging
logging.getLogger("isochrones").setLevel("ERROR")
from isochrones.mist.bc import MISTBolometricCorrectionGrid
logging.getLogger("isochrones").setLevel("WARNING")

__all__ = ["get_log_g", "get_absolute_bol_mag", "get_apparent_mag", "get_absolute_mag", "add_mags",
           "get_extinction", "get_photometry", "get_single_WD_absolute_mag",
           "get_binary_WD_absolute_mag", "get_WD_photometry"]


def get_log_g(mass, radius):
    """Computes log of the surface gravity in cgs

    Parameters
    ----------
    mass : :class:`~astropy.units.Quantity` [mass]
        Mass of the star
    radius : :class:`~astropy.units.Quantity` [radius]
        Radius of the star

    Returns
    -------
    log g : :class:`~numpy.ndarray`
        Log of the surface gravity in cgs
    """
    g = const.G * mass / radius**2

    # avoid division by zero errors (for massless remnants)
    with np.errstate(divide='ignore'):
        return np.log10(g.cgs.value)


def get_absolute_bol_mag(lum):
    """Computes the absolute bolometric magnitude following
    `IAU Resolution B2 <https://www.iau.org/news/announcements/detail/ann15023/>`_

    Parameters
    ----------
    lum : :class:`~astropy.units.Quantity` [luminosity]
        Luminosity of the star

    Returns
    -------
    M_bol : :class:`~numpy.ndarray`
        Absolute bolometric magnitude
    """
    zero_point_lum = 3.0128e28 * u.watt
    return -2.5 * np.log10(lum / zero_point_lum)


def get_apparent_mag(M_abs, distance):
    """Convert absolute magnitude to apparent magnitude

    Parameters
    ----------
    M_abs : :class:`~numpy.ndarray`
        Absolute magnitude
    distance : :class:`~astropy.units.Quantity` [length]
        Distance

    Returns
    -------
    m_app : :class:`~numpy.ndarray`
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
    M_abs : :class:`~numpy.ndarray`
        Apparent magnitude
    distance : :class:`~astropy.units.Quantity` [length]
        Distance

    Returns
    -------
    m_app : :class:`~numpy.ndarray`
        Absolute magnitude
    """
    M_abs = m_app - 5 * np.log10(distance / (10 * u.pc))
    return M_abs


def add_mags(*mags, remove_nans=True):
    """Add any number of magnitudes

    Parameters
    ----------
    *mags : `list` or `np.array` or `float` or `int`
        A series of magnitudes. If arrays are given then all must have the same length. If a mixture of single
        values and arrays are given then the single values will be added to each array element
    remove_nans : `bool`, optional
        Whether to remove NaNs from the total (if not the total will be NaN), by default True

    Returns
    -------
    total_mag : :class:`~numpy.ndarray`
        Total magnitude

    Raises
    ------
    ValueError
        If any magnitude is an invalid type
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


def get_extinction(coords):     # pragma: no cover
    """Calculates the visual extinction values for a set of coordinates

    Reddening due to dust is calculated using the Bayestar dustmap. Then the conversion from this to a visual
    extension is done assuming a total-to-selective extinction ratio of 3.3, as is used by
    `Green+2019 <https://iopscience.iop.org/article/10.3847/1538-4357/ab5362#apjab5362s2>`_

    .. warning::
        The dustmap used only covers declinations > -30 degrees, any supplied coordinates below this will be
        reflected around the galactic plane (any of [(-l, b), (l, -b), (-l, -b)]) and the dust at these
        locations will be used instead

    Parameters
    ----------
    coords : :class:`~astropy.coordinates.SkyCoord`
        The coordinates at which you wish to calculate extinction values

    Returns
    -------
    Av : :class:`~numpy.ndarray`
        Visual extinction values for each set of coordinates
    """

    # following section performs reflections for coordinates below -30 deg declination
    # convert to galactic coordinates
    galactic = coords.galactic

    # try all possible reflections about the galactic plane
    for ref_l, ref_b in [(-1, 1), (1, -1), (-1, -1)]:
        # check which things are too low for the dustmap
        too_low = galactic.icrs.dec < (-30 * u.deg)

        # if everything is fine now then stop
        if not too_low.any():
            break

        # apply the reflection to the subset that are too low
        reflected = galactic[too_low]
        reflected.data.lon[()] *= ref_l
        reflected.data.lat[()] *= ref_b
        reflected.cache.clear()

        # check which are fixed now, and reverse the reflection if not
        fixed = reflected.icrs.dec > (-30 * u.deg)
        reflected.data.lon[~fixed] *= ref_l
        reflected.data.lat[~fixed] *= ref_b

        # set the data back in the main coord object
        galactic.data.lon[too_low] = reflected.data.lon
        galactic.data.lat[too_low] = reflected.data.lat

        # clear the cache to ensure consistency
        galactic.cache.clear()

    bayestar = BayestarQuery(max_samples=2, version='bayestar2019')

    # calculate the reddening due to dust
    ebv = bayestar(galactic, mode='random_sample')

    # convert this to a visual extinction
    Av = 3.3 * ebv
    return Av


def get_photometry(final_bpp, final_coords, filters, ignore_extinction=False):
    """Computes photometry subject to dust extinction using the MIST boloemtric correction grid

    Parameters
    ----------
    final_bpp : :class:`~pandas.DataFrame`
        A dataset of COSMIC binaries at present day - must include these columns: ["sep", "metallicity"] and
        for each star it must have the columns ["teff", "lum", "mass", "rad", "kstar"]
    final_coords : `tuple` of :class:`~astropy.coordinates.SkyCoord`
        Final positions and velocities of the binaries at present day. First entry is for binaries or the
        primary in a disrupted system, second entry is for secondaries in a disrupted system.
    filters : `list` of `str`
        Which filters to compute photometry for (e.g. ['J', 'H', 'K', 'G', 'BP', 'RP'])
    ignore_extinction : `bool`
        Whether to ignore extinction

    Returns
    -------
    photometry : :class:`~pandas.DataFrame`
        Photometry and extinction information for supplied COSMIC binaries in desired `filters`
    """
    # set up empty photometry table
    photometry = pd.DataFrame()
    disrupted = final_bpp["sep"].values < 0.0

    if not ignore_extinction:       # pragma: no cover
        # get extinction for bound binaries and primary of disrupted binaries
        photometry['Av_1'] = get_extinction(final_coords[0])

        # get extinction for secondaries of disrupted binaries (leave as np.inf otherwise)
        photometry['Av_2'] = np.repeat(np.inf, len(final_coords[1]))
        photometry.loc[disrupted, "Av_2"] = get_extinction(final_coords[1][disrupted])

        # ensure extinction remains in MIST grid range (<= 6) and is not NaN
        photometry.loc[photometry.Av_1 > 6, ['Av_1']] = 6
        photometry.loc[photometry.Av_2 > 6, ['Av_2']] = 6
        photometry = photometry.fillna(6)
    else:
        photometry['Av_1'] = np.zeros(len(final_coords[0]))
        photometry['Av_2'] = np.zeros(len(final_coords[0]))

    # get Fe/H using e.g. Bertelli+1994 Eq. 10 (assuming all stars have the solar abundance pattern)
    Z_sun = 0.0142
    FeH = np.log10(final_bpp["metallicity"].values / Z_sun)

    # set up MIST bolometric correction grid
    bc_grid = MISTBolometricCorrectionGrid(bands=filters)
    bc = [None, None]

    # for each star in the (possibly disrupted/merged) binary
    for ind in [1, 2]:
        # calculate the surface gravity if necessary
        if f"log_g_{ind}" not in final_bpp:
            final_bpp.insert(len(final_bpp.columns), f"log_g_{ind}",
                             get_log_g(mass=final_bpp[f"mass_{ind}"].values * u.Msun,
                                       radius=final_bpp[f"rad_{ind}"].values * u.Rsun))

        # get the bolometric corrections from MIST isochrones
        # TODO: absolute mags should not include dust, give apparent both with and without
        bc[ind - 1] = bc_grid.interp([final_bpp[f"teff_{ind}"].values, final_bpp[f"log_g_{ind}"].values,
                                      FeH, photometry[f"Av_{ind}"]],
                                     filters)

        # calculate the absolute bolometric magnitude and set any BH or massless remnants to invisible
        photometry[f"M_abs_{ind}"] = get_absolute_bol_mag(lum=final_bpp[f"lum_{ind}"].values * u.Lsun)
        photometry.loc[np.isin(final_bpp[f"kstar_{ind}"].values, [13, 14, 15]), f"M_abs_{ind}"] = np.inf

        # work out the distance (if the system is bound always use the first `final_coords` SkyCoord)
        distance = np.repeat(np.inf, len(final_bpp)) * u.kpc
        distance[disrupted] = final_coords[ind - 1][disrupted].icrs.distance
        distance[~disrupted] = final_coords[0][~disrupted].icrs.distance

        # convert the absolute magnitude to an apparent magnitude
        photometry[f"m_app_{ind}"] = get_apparent_mag(M_abs=photometry[f"M_abs_{ind}"].values,
                                                      distance=distance)

    # go through each filter
    for i, filter in enumerate(filters):
        for prefix, mag_type in [("m", "app"), ("M", "abs")]:
            # apply the bolometric corrections to the apparent magnitude of each star
            filter_mags = [photometry[f"{prefix}_{mag_type}_{ind}"].values - bc[ind - 1][:, i]
                           for ind in [1, 2]]

            # total the magnitudes (removing any NaNs)
            total_filter_mag = add_mags(*filter_mags, remove_nans=True)

            # default to assuming all systems are bound - in this case total magnitude is listed
            # in primary filter mag, and secondary is non-existent
            photometry[f"{filter}_{mag_type}_1"] = total_filter_mag
            photometry[f"{filter}_{mag_type}_2"] = np.repeat(np.inf, len(photometry))

            # for disrupted systems, change filter apparent magnitudes to the values for each individual star
            for ind in [1, 2]:
                photometry.loc[disrupted, f"{filter}_{mag_type}_{ind}"] = filter_mags[ind - 1][disrupted]

            # for the G filter in particular, see which temperature/log-g is getting measured
            if filter == "G" and mag_type == "app":
                # by default assume the primary is dominant
                photometry["teff_obs"] = final_bpp["teff_1"].values
                photometry["log_g_obs"] = final_bpp["log_g_1"].values

                # overwrite any values where the secondary is brighter
                two_is_brighter = (filter_mags[1] < filter_mags[0]) | (np.isnan(filter_mags[0])
                                                                       & ~np.isnan(filter_mags[1]))
                photometry["secondary_brighter"] = two_is_brighter
                photometry.loc[two_is_brighter, "teff_obs"] = final_bpp["teff_2"].values[two_is_brighter]
                photometry.loc[two_is_brighter, "log_g_obs"] = final_bpp["log_g_2"].values[two_is_brighter]

    return photometry


def WD_photometry(age_dat, mass_dat, color_select):
    """Computes the photometry of a single white dwarf
    population based on age and mass for colors provided
    
    Uses the WD cooling models of Pierre Bergeron et al. 
    https://www.astro.umontreal.ca/~bergeron/CoolingModels/
    
    Parameters
    ----------
    age : :class:`~numpy.ndarray`
        White dwarf ages with astropy time unit
    mass : :class:`~numpy.ndarray`
        White dwarf masses with astropy mass unit
    color_select : `str`
        Specifies the colors to compute absolute brightnesses for
        choose from:
        ["Mbol", "UU", "BB", "VV", "RR", "II", 
         "JJ", "HH", "Ks", "Y", "J", "H", "K", "W1",
         "W2", "W3", "W4", "S3.6", "S4.5", "S5.8", "S8.0", 
         "u", "g", "r", "i", "z", "gg", "rr", "ii", "zz", "yy",
         "G2", "G2_BP", "G2_RP", "G3", "G3_BP", "G3_RP", 
         "FUV", "NUV"]
         
    Returns
    -------
    photometry : `list` of :class:`~numpy.ndarray`s 
        Absolute magnitudes for each WD in color_select color
    """
    # age is in yrs
    WD_models = pd.read_hdf("./data/Bergeron_WD_photometry.h5", key="dat")
    masses = np.round(np.arange(0.2, 1.4, 0.1), 1)
    mag = np.zeros(len(age_dat))
    for m in masses:
        ind, = np.where( (mass_dat < m + 0.05) & (mass_dat> m - 0.05) )
        WD_model = WD_models.loc[(WD_models.mass == m)]
        mag_interp = interp1d(WD_model.Age.values/1e6, WD_model[color_select].values.flatten(), fill_value="extrapolate")
    
        mag[ind] = mag_interp(age_dat[ind])
    
                              
    return mag


def get_single_WD_photometry(age, mass, color_select):
    """Computes the photometry of a single white dwarf
    population based on age and mass for colors provided
    
    Uses the WD cooling models of Pierre Bergeron et al. 
    https://www.astro.umontreal.ca/~bergeron/CoolingModels/
    
    Parameters
    ----------
    age : :class:`~numpy.ndarray`
        White dwarf ages with astropy time unit
    mass : :class:`~numpy.ndarray`
        White dwarf masses with astropy mass unit
    color_select : `list` of `str`
        Specifies the colors to compute absolute brightnesses for
        choose from:
        ["Mbol", "UU", "BB", "VV", "RR", "II", 
         "JJ", "HH", "Ks", "Y", "J", "H", "K", "W1",
         "W2", "W3", "W4", "S3.6", "S4.5", "S5.8", "S8.0", 
         "u", "g", "r", "i", "z", "gg", "rr", "ii", "zz", "yy",
         "G2", "G2_BP", "G2_RP", "G3", "G3_BP", "G3_RP", 
         "FUV", "NUV"]
         
    Returns
    -------
    photometry_abs : `list` of :class:`~numpy.ndarray`s 
        Absolute magnitudes for each WD and color
    """
    photometry_abs =  WD_photometry(age.to(u.Myr).value, mass.to(u.Msun).value, color_select)
    
    return photometry_abs
    

def get_binary_WD_photometry(age_1, age_2, mass_1, mass_2, color_select):
    """Computes the photometry of a single white dwarf
    population based on age and mass for colors provided
    
    Uses the WD cooling models of Pierre Bergeron et al. 
    https://www.astro.umontreal.ca/~bergeron/CoolingModels/
    
    Parameters
    ----------
    age_1 : :class:`~numpy.ndarray`
        White dwarf ages for component 1 with astropy time unit
    age_2 : :class:`~numpy.ndarray`
        White dwarf ages for component 2 with astropy time unit
    mass_1 : :class:`~numpy.ndarray`
        White dwarf masses for component 1 with astropy mass unit
    mass_2 : :class:`~numpy.ndarray`
        White dwarf masses for component 2 with astropy mass unit
    color_select : `list` of `str`
        Specifies the colors to compute absolute brightnesses for
        choose from:
        ["Mbol", "UU", "BB", "VV", "RR", "II", 
         "JJ", "HH", "Ks", "Y", "J", "H", "K", "W1",
         "W2", "W3", "W4", "S3.6", "S4.5", "S5.8", "S8.0", 
         "u", "g", "r", "i", "z", "gg", "rr", "ii", "zz", "yy",
         "G2", "G2_BP", "G2_RP", "G3", "G3_BP", "G3_RP", 
         "FUV", "NUV"]
         
    Returns
    -------
    photometry_1 : `list` of :class:`~numpy.ndarray`s 
        Absolute magnitudes for each WD and color for component 1
    photometry_2 : `list` of :class:`~numpy.ndarray`s 
        Absolute magnitudes for each WD and color for component 2
    photometry_tot : `list` of :class:`~numpy.ndarray`s 
        Absolute magnitudes for each WD and color for binary brightness
    """
    
    
    photometry_1_abs =  WD_photometry(age_1.to(u.Myr).value, mass_1.to(u.Msun).value, color_select)
    photometry_2_abs =  WD_photometry(age_2.to(u.Myr).value, mass_2.to(u.Msun).value, color_select)
    photometry_tot_abs = add_mags([photometry_1_abs, photometry_2_abs])
    
    return photometry_1, photometry_2, photometry_tot

def get_WD_photometry(WD_pop, final_coords, filters):
    """Computes photometry subject to dust extinction using the 
    Bergeron cooling models
    https://www.astro.umontreal.ca/~bergeron/CoolingModels/

    Parameters
    ----------
    WD pop : :class:`~pandas.DataFrame`
        A dataset of COSMIC binaries at present day - must include these columns: ["sep"] and
        for each star it must have the columns ["age_1", "mass_1", "age_2", "mass_2"]
    final_coords : `tuple` of :class:`~astropy.coordinates.SkyCoord`
        Final positions and velocities of the binaries at present day. First entry is for binaries or the
        primary in a disrupted system, second entry is for secondaries in a disrupted system.
    filters : `list` of `str`
        Which filters to compute photometry for choose from:
        ["Mbol", "UU", "BB", "VV", "RR", "II", 
         "JJ", "HH", "Ks", "Y", "J", "H", "K", "W1",
         "W2", "W3", "W4", "S3.6", "S4.5", "S5.8", "S8.0", 
         "u", "g", "r", "i", "z", "gg", "rr", "ii", "zz", "yy",
         "G2", "G2_BP", "G2_RP", "G3", "G3_BP", "G3_RP", 
         "FUV", "NUV"]
    
    
    Returns
    -------
    photometry : :class:`~pandas.DataFrame`
        Photometry and extinction information for supplied COSMIC binaries in desired `filters`
    """
    merger_mask = WD_pop.WD_merger_age.values > 0
    
    # First calculate the photometry of each
    # for each star in the (possibly disrupted/merged) binary
    # set up empty photometry table
    photometry = pd.DataFrame()
    
    distance = final_coords.transform_to(coords.ICRS).distance
    # go through each filter
    for i, filter in enumerate(filters):
        # apply the bolometric corrections to the apparent magnitude of each star
        #print("{prefix}_{mag_type}_{1}", f"{filter}_{mag_type}_1")
        filter_mags = [get_single_WD_photometry(WD_pop[f"WD_age_{ind}"].values * u.Myr, 
                                                WD_pop[f"mass_{ind}"].values * u.Msun, 
                                                [filter])
                       for ind in [1, 2]]

        ## total the magnitudes (removing any NaNs)
        total_filter_mag = add_mags(*filter_mags, remove_nans=True)

        # Assign the different filter magnitudes
        photometry[f"{filter}_abs_1"] = filter_mags[0]
        photometry[f"{filter}_abs_2"] = filter_mags[1]
        photometry[f"{filter}_abs_tot"] = total_filter_mag
        
        photometry[f"{filter}_app_1"] = get_apparent_mag(filter_mags[0], distance=distance)
        photometry[f"{filter}_app_2"] = get_apparent_mag(filter_mags[1], distance=distance)
        photometry[f"{filter}_app_tot"] = get_apparent_mag(total_filter_mag, distance=distance)

        for mag_type in ["abs", "app"]:
            for ind in [1, 2]:
                photometry.loc[merger_mask, f"{filter}_{mag_type}_{ind}"] = 9999.9
            photometry.loc[merger_mask, f"{filter}_{mag_type}_tot"] = 9999.9
        
        # Next do the mergers
        
        photometry[f"{filter}_abs_merger"] = np.ones_like(total_filter_mag) * 9999.9
        photometry[f"{filter}_app_merger"] = np.ones_like(total_filter_mag) * 9999.9
        photometry.loc[merger_mask, f"{filter}_abs_merger"] = get_single_WD_photometry(
            WD_pop.loc[merger_mask].WD_merger_age.values * u.Myr, 
            WD_pop.loc[merger_mask].WD_merger_mass.values * u.Msun, 
            [filter]
        )
        
        photometry.loc[merger_mask, f"{filter}_app_merger"] = get_apparent_mag(photometry.loc[merger_mask][f"{filter}_abs_merger"], distance=distance[merger_mask])
        
    return photometry