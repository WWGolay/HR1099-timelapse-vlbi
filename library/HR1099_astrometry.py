""" 
18 Jan 2020 change pm unit to mas/yr to avoid small number problem in LMFIT 
26 Feb 2021 Add geo_to_helio option in helio_geo()
01 May 2023 add GAIA DR3 astrometry, Ransom binary params, incl = 40 deg
13 May 2023 add stellar params [masses, radii, etc]
03 Jun 2023 fix orbit_phase
21 Aug 2023 fix yr = 356.26*day typo
23 Aug 2023 fix plot_sky (r1 -> R1, etc, mas units problem)
"""

version = "23 Aug 2023"

import numpy as np
from scipy.optimize import root
from skyfield.api import Star, load
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

sec = 1
day = 86400 * sec
yr = 365.26 * day
arcsec = np.pi / (180.0 * 3600)
mas = arcsec / 1e3
meter = 1
km = 1e3 * meter
AU = 1.49e11 * meter
pc = AU / arcsec
R_sun = 7e5 * km
deg = np.pi / 180.0

# GAIA astrometry DR3
ts = load.timescale()
gaia_epoch = ts.utc(2016, 1, 1)
gaia_jd0 = gaia_epoch.tai
gaia_coord0 = (
    54.1969001299 * deg,
    0.5870418112 * deg,
)  # uncertainties +/-0.07 mas, +/-0.05 mas
gaia_parallax = 33.9783 * mas  # uncertainty 0.035 mas
gaia_pm = (
    -32.2464,
    -162.0732,
)  # N.B. units; are mas/yr uncertainties +/-0.036, +/-0.031 mas/yr
d = AU / gaia_parallax

# Binary params [Strassmeier et al. 2020 AA]
T = 2457729.7084
P = 2.837711 * day
ecc = 0.0
m1_sin3i = 0.2256
m2_sin3i = 0.1752
a1_sini = 1.8915e9 / (d)
a2_sini = 2.4355e9 / (d)

# BM392 JDs and dates
bm392_epochs = [
    2456609.83,
    2456615.80,
    2456622.77,
    2456658.69,
    2456659.68,
    2456674.63,
]  # Appx UT at middle of epoch
bm392_labels = ["Epoch A", "Epoch B", "Epoch C", "Epoch D", "Epoch E", "Epoch F"]
# For each epoch, UT hour ranges [i.e., J0340 has good solution]
ep_ymd = [
    [2013, 11, 13],
    [2013, 11, 19],
    [2013, 11, 26],
    [2014, 1, 1],
    [2014, 1, 2],
    [2014, 1, 17],
]
ep_hr = [[4, 8], [3, 11], [2, 9], [2, 8], [2, 7], [1, 5]]

R_pri = 3.7 * R_sun
R_sec = 1.1 * R_sun
R1 = R_pri / (d * mas)
R2 = R_sec / (d * mas)

# omega is indeterminant
omega = 0 * deg


class HR1099_astrometry:
    def __init__(self, incl, Omega):
        """
        Binary astrometry calculator: outputs celestial (ICRS) coordinates of binary components given input jd
        Astrometric parameters from Strassmeier et al. AA 2020.
        Inputs:
                incl = inclination angel [rad]
                Omega = longitude f ascending node (rads)

        """

        coord0 = gaia_coord0
        jd0 = gaia_jd0
        Par = gaia_parallax
        pm = gaia_pm
        m1 = m1_sin3i / np.sin(incl) ** 3
        m2 = m2_sin3i / np.sin(incl) ** 3
        a1 = a1_sini / np.sin(incl)
        a2 = a2_sini / np.sin(incl)
        a = a1 + a2

        self.coord0 = coord0
        self.jd0 = jd0
        self.pm = pm
        self.T = T
        self.P = P
        self.d = d
        self.Par = Par
        self.a = a
        self.m1 = m1
        self.m2 = m2
        self.R1 = R1
        self.R2 = R2
        self.incl = incl
        self.Omega = Omega
        self.ecc = ecc
        self.omega = omega
        self.bm392_epochs = bm392_epochs
        self.bm392_labels = bm392_labels
        self.ep_ymd = ep_ymd
        self.ep_hr = ep_hr
        self.version = version

    def get_version(self):
        return self.version

    def bm392_info(self):
        """Return BM392 epochs (JD, dates/times, and mean orbit phases)"""
        Phases = []
        for jd in self.bm392_epochs:
            Phases.append(self.orbit_phase(jd))
        return self.bm392_labels, self.bm392_epochs, Phases

    def hr1099_info(self):
        """Returns stellar parameters for HR1099, from Strassmeier et al. 2020 AA
        T = reference epoch = when K star is in front and radial velocity = 0 [day]
        P = orbital period [sec]
        d = distance [m]
        a = semi-major axis [mas]
        m1,m2 = masses of K, G star [Msun]
        r1,r2 = (angular) radii of K, G stars [mas]"""
        return self.T, self.P, self.d, self.a, self.m1, self.m2, self.R1, self.R2

    def coord_cm(self, jd):
        """return HR1099 c.m. ICRS coordinates (radians) at given jd using GAIA position,pm"""
        ra, dec = self.coord0
        pm_ra, pm_dec = [t * mas for t in self.pm]
        ra += pm_ra * (jd - self.jd0) * day / yr
        dec += pm_dec * (jd - self.jd0) * day / yr
        return SkyCoord(ra=ra * u.radian, dec=dec * u.radian)

    def coord_comps(self, jd):
        """Returns RA, Dec (Skycoord object) of both binary components (cm + offset) at given jd"""
        c = self.coord_cm(jd)
        ra_cm, dec_cm = (c.ra.rad, c.dec.rad)
        dra1, ddec1, dra2, ddec2, rho1, rho2, phi = self.binary_offsets(jd)
        ra1 = ra_cm + dra1
        dec1 = dec_cm + ddec1
        c_pri = SkyCoord(ra=(ra_cm + dra1) * u.radian, dec=(dec_cm + ddec1) * u.radian)
        c_sec = SkyCoord(ra=(ra_cm + dra2) * u.radian, dec=(dec_cm + ddec2) * u.radian)
        return (c_pri, c_sec)

    def Ecc(self, jd):
        """Calculate eccentric anomaly"""

        def f_ecc(E, *p):
            M, e = p
            return E - e * np.sin(E) - M

        t = np.array((jd - self.T) / (self.P / day))
        M = 2 * np.pi * np.modf(t)[0]
        res = root(f_ecc, M, args=(M, self.ecc))
        E = res.x[0]
        Success = res.success
        return E, Success

    def orbit_phase(self, jd):
        t = (jd - self.T) / (self.P / day)
        t = np.modf(t)[0]
        if t < 0:
            t += 1
        return t

    def r(self, jd):
        """semi-major axes, component distances from c.m."""
        R = self.m2 / (self.m1 + self.m2)
        a1 = self.a * R
        a2 = self.a * (1 - R)
        E, success = self.Ecc(jd)
        t = 1 - self.ecc * np.cos(E)
        r1 = a1 * t
        r2 = a2 * t
        return a1, a2, r1, r2

    def binary_offsets(self, jd):
        """RA,dec offsets, position angles w.r.t c.m. (mas)"""
        a1, a2, r1, r2 = self.r(jd)
        t = (1 + self.ecc) / (1 - self.ecc)
        E, success = self.Ecc(jd)
        if not np.all(success):
            print("Warning: Bad eccentric anomaly solution")
        nu = 2 * np.arctan(np.sqrt(t) * np.tan(E / 2))
        Theta = np.arctan2(
            np.sin(nu + self.omega) * np.cos(self.incl), np.cos(nu + self.omega)
        )
        phi = np.fmod(Theta + self.Omega, 2 * np.pi)
        zeta = np.sqrt(
            np.sin(nu + self.omega) ** 2 * np.cos(self.incl) ** 2
            + np.cos(nu + self.omega) ** 2
        )
        rho1 = r1 * zeta
        rho2 = -r2 * zeta
        ra1 = -rho1 * np.sin(phi)
        ra2 = -rho2 * np.sin(phi)
        dec1 = -rho1 * np.cos(phi)
        dec2 = -rho2 * np.cos(phi)
        return ra1, dec1, ra2, dec2, rho1, rho2, phi

    def helio_geo(self, c, jd, geo_to_helio=False):
        """Calculates geocentric coords given heliocentric coords at given jd, (or reverse if geo_to_helio = True)"""
        ra, dec = (c.ra.rad, c.dec.rad)
        parallax = self.Par
        n = jd - 2451545.0
        L = np.radians(280.466 + 0.9856474 * n)
        g = np.radians(357.528 + 0.9856003 * n)
        lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.02) * np.sin(2 * g)
        epsilon = np.radians(23.439 - 0.00000004 * n)
        R = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)
        X = -R * np.cos(lam)
        Y = -R * np.sin(lam) * np.cos(epsilon)
        Z = -R * np.sin(lam) * np.sin(epsilon)
        dra = parallax * (X * np.sin(ra) - Y * np.cos(ra))
        ddec = parallax * (
            X * np.cos(ra) * np.sin(dec)
            + Y * np.sin(ra) * np.sin(dec)
            - Z * np.cos(dec)
        )
        if geo_to_helio:
            ra -= dra
            dec -= ddec
        else:
            ra += dra
            dec += ddec
        return SkyCoord(ra=ra * u.radian, dec=dec * u.radian)

    def plot_sky(self, jd, vlb_xy=None, size=1.5, figsize=10, grid=True, title=""):
        """Plots components on sky at specified jd (returns plt object)"""
        ra1, dec1, ra2, dec2, rho1, rho2, phi = self.binary_offsets(jd)

        figure, axes = plt.subplots(figsize=(figsize, figsize))
        # K star
        draw_circle = plt.Circle((ra1 / mas, dec1 / mas), self.R1, color="red")
        axes.add_artist(draw_circle)
        # G star
        draw_circle = plt.Circle((ra2 / mas, dec2 / mas), self.R2, color="green")
        axes.add_artist(draw_circle)

        # Plot VLB position if given
        if vlb_xy != None:
            x, xerr, y, yerr = vlb_xy
            axes.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="b+")

        axes.set_aspect(1)
        plt.xlim(size, -size)
        plt.ylim(-size, size)
        if title == "":
            title = r"HR1099 JD = %.3f ($\phi$=%.2f)" % (jd, self.orbit_phase(jd))
        plt.title(title)
        plt.grid(grid)
        plt.xlabel(r"$\Delta$ RA")
        plt.ylabel(r"$\Delta$ Dec")
        return figure
