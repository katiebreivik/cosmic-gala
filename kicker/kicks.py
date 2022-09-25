import numpy as np
import gala.integrate as gi
import gala.dynamics as gd
import gala.potential as gp

import astropy.coordinates as coords
import astropy.units as u
import astropy.constants as const


def get_kick_differential(delta_v_sys_xyz, m_1, m_2, a):
    """Calculate the Differential from a combination of the natal kick, Blauuw kick and orbital motion.

    Parameters
    ----------
    delta_v_sys_xyz : `float array`
        Change in systemic velocity due to natal and Blauuw kicks in BSE (v_x, v_y, v_z) frame (see Fig A1 of
        Hurley+02)
    m_1 : `float`
        Primary mass
    m_2 : `float`
        Secondary Mass
    a : `float`
        Binary separation

    Returns
    -------
    kick_differential : `CylindricalDifferential`
        Kick differential
    """
    # calculate the orbital velocity ASSUMING A CIRCULAR ORBIT
    if a.value > 0.0:
        v_orb = np.sqrt(const.G * (m_1 + m_2) / a)

        # adjust change in velocity by orbital motion of supernova star
        delta_v_sys_xyz -= v_orb

    # orbital phase angle and inclination to Galactic plane
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, 2 * np.pi)

    # rotate BSE (v_x, v_y, v_z) into Galactocentric (v_X, v_Y, v_Z)
    v_X = delta_v_sys_xyz[0] * np.cos(theta) - delta_v_sys_xyz[1] * np.sin(theta) * np.cos(phi)\
        + delta_v_sys_xyz[2] * np.sin(theta) * np.sin(phi)
    v_Y = delta_v_sys_xyz[0] * np.sin(theta) + delta_v_sys_xyz[1] * np.cos(theta) * np.cos(phi)\
        - delta_v_sys_xyz[2] * np.cos(theta) * np.sin(phi)
    v_Z = delta_v_sys_xyz[1] * np.sin(phi) + delta_v_sys_xyz[2] * np.cos(phi)

    kick_differential = coords.CartesianDifferential(v_X, v_Y, v_Z)

    return kick_differential


def integrate_orbit_with_events(w0, potential=gp.MilkyWayPotential(), events=None,
                                t1=None, t2=None, dt=None):
    """Integrate PhaseSpacePosition in a potential with events that occur at certain times

    Parameters
    ----------
    w0 : `ga.dynamics.PhaseSpacePosition`
        Initial phase space position
    potential : `ga.potential.PotentialBase`, optional
        Potential in which you which to integrate the orbits, by default the MilkyWayPotential()
    events : `list of objects`
        Events that occur during the orbit evolution. Should contain the following parameters: `time`, `m_1`,
        `m_2`, `a`, `ecc`, `delta_v_sys_xyz`
    rng : `NumPy RandomNumberGenerator`
        Which random number generator to use

    Returns
    -------
    full_orbit : `ga.orbit.Orbit`
        Orbit that have been integrated
    """
    # if there are no events then just integrate the whole thing
    if events is None:
        return potential.integrate_orbit(w0, t1=t1, t2=t2, dt=dt)

    # if there are two lists (due to a disruption) then recursively call the function
    if isinstance(events[0], list):
        assert len(events) == 2
        return [integrate_orbit_with_events(w0=w0, potential=potential, events=events[i],
                                            t1=t1, t2=t2, dt=dt) for i in range(len(events))]

    # work out what the timesteps would be without kicks
    timesteps = gi.parse_time_specification(units=[u.s], t1=t1, t2=t2, dt=dt) * u.s

    # start the cursor at the smallest timestep
    time_cursor = timesteps[0]
    current_w0 = w0

    # keep track of the orbit data throughout
    orbit_data = []

    # loop over the events
    for event in events:
        # find the timesteps that occur before the kick
        timestep_mask = (timesteps >= time_cursor) & (timesteps < event["time"])

        # if any of them occur before the kick then do some integration
        if any(timestep_mask):
            matching_timesteps = timesteps[timestep_mask]

            # integrate the orbit over these timesteps
            orbit = potential.integrate_orbit(current_w0, t=matching_timesteps)

            # save the orbit data
            orbit_data.append(orbit.data)

            # get new PhaseSpacePosition(s)
            current_w0 = orbit[-1]

        # adjust the time
        time_cursor = event["time"]

        # calculate the kick differential
        kick_differential = get_kick_differential(delta_v_sys_xyz=event["delta_v_sys_xyz"],
                                                  m_1=event["m_1"], m_2=event["m_2"], a=event["a"])

        # update the velocity of the current PhaseSpacePosition
        current_w0 = gd.PhaseSpacePosition(pos=current_w0.pos,
                                           vel=current_w0.vel + kick_differential,
                                           frame=current_w0.frame)

    # if we still have time left after the last event (very likely)
    if time_cursor < timesteps[-1]:
        # evolve the rest of the orbit out
        matching_timesteps = timesteps[timesteps >= time_cursor]
        orbit = potential.integrate_orbit(current_w0, t=matching_timesteps)
        orbit_data.append(orbit.data)

    if len(orbit_data) > 1:
        orbit_data = coords.concatenate_representations(orbit_data)
    else:
        orbit_data = orbit_data[0]

    full_orbit = gd.orbit.Orbit(pos=orbit_data.without_differentials(),
                                vel=orbit_data.differentials["s"],
                                t=timesteps.to(u.Myr))

    return full_orbit

