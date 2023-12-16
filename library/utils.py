import cmath
import os
import pickle
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib import offsetbox, patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy import ndimage

deg = np.pi / 180
mas = deg / (60 * 60 * 1000)
R_sun = 6.957e8
mJy = 1
Jy = 1e3 * mJy


def polar2rect(r, phi):
    """
    Convert polar to cartesian coordinates
    For difmap model comps, phi is measured from y CCW [so rotate, swap x sign
    Input: radius, angle [rad], Output: x,y
    """
    phi += np.pi / 2
    z = cmath.rect(r, phi)
    return -z.real, z.imag


def rect2polar(x, y):
    """
    Convert cartesian to polar coordinates
    For difmap model comps, phi is measured from y CCW [so rotate, swap x sign
    Input: x,y, Output: radius, angle [rad]
    """
    z = complex(x, y)
    r = abs(z)
    phi = cmath.phase(z)
    phi -= np.pi / 2
    return r, -phi


def f_rot1(x, y, phi):
    """2-d Cartesian rotation [phi in radians]"""
    xr = x * np.cos(phi) - y * np.sin(phi)
    yr = x * np.sin(phi) + y * np.cos(phi)
    return xr, yr


def get_beam(log):
    # Example: "! Restoring with beam: 0.3935 x 0.8931 at -4.334 degrees (North through East)"

    for line in reversed(open(log).readlines()):
        if "! Restoring with beam: " in line:
            x = line[23:].split(" x ")
            y = x[1].split(" at ")
            z = y[1].split(" degrees (North through East)")
            beam = [float(y[0]) * mas, float(x[0]) * mas, float(z[0]) * deg]
            break
    else:
        print("Beam not found in log file")
        beam = None
    return beam


def plot_binary(
    ax,
    binary,
    jd,
    corotate=False,
    r1=0.59,
    r2=0.18,
    plot_stars=True,
    centroid=None,
    color="black",
    label="",
    model=None,
    mapsize=6,
    cells=256,
    levs=[4, 8, 16, 32, 64],
    max_lev_scalar=1,
    cmap=cm.hawaii,
    bg=None,
    write_levs=True,
    beam=None,
    show_coord_axes=False,
    hour_range=None,
    fontsize=14,
    marker="o",
    d=None,
    bar_pos="lower right",
):
    if bg is not None:
        img = plt.imread(bg)
        ax.imshow(
            img,
            extent=[-mapsize / 2, mapsize / 2, -mapsize / 2, mapsize / 2],
            zorder=0,
            aspect="auto",
        )
        ax.tick_params(axis="both", which="major", color="darkgray")

    ra1, dec1, ra2, dec2, rho1, rho2, phi = binary.binary_offsets(jd)
    ra1 /= mas
    dec1 /= mas
    ra2 /= mas
    dec2 /= mas

    if corotate:
        ra1, dec1 = f_rot1(ra1, dec1, phi - np.pi / 2)
        ra2, dec2 = f_rot1(ra2, dec2, phi - np.pi / 2)

    if plot_stars:
        s1 = plt.Circle(
            (ra1, dec1), r1, color="red", zorder=1, label="K1 IV", alpha=0.2
        )
        s2 = plt.Circle(
            (ra2, dec2), r2, color="blue", zorder=1, label="G5 IV-V", alpha=0.2
        )
        ax.add_artist(s1)
        ax.add_artist(s2)

        ax.plot([ra1], [dec1], markersize=2, marker="o", color="red")
        ax.plot([ra2], [dec2], markersize=2, marker="o", color="blue")

    ax.plot([0], [0], markersize=5, marker="o", color="black")

    if hour_range is not None:
        for hr in range(-hour_range, hour_range + 1):
            if hr == 0:
                continue
            ra1, dec1, ra2, dec2, rho1, rho2, phi = binary.binary_offsets(jd + hr / 24)
            ra1 /= mas
            dec1 /= mas
            ra2 /= mas
            dec2 /= mas
            ax.plot([ra1], [dec1], markersize=1, marker="o", color="red")
            ax.plot([ra2], [dec2], markersize=1, marker="o", color="blue")

    if centroid is not None and model is not None:
        xx, yy = np.meshgrid(
            np.linspace(-mapsize * mas, mapsize * mas, cells, endpoint=True),
            np.linspace(-mapsize * mas, mapsize * mas, cells, endpoint=True),
        )
        im_model = model.get_image_plane_model(
            xx, yy, x_shift=centroid[0] * mas, y_shift=centroid[1] * mas
        )
        if corotate:
            im_model = ndimage.rotate(im_model, phi / deg - np.pi / 2, reshape=False)
        levels = [max_lev_scalar * np.max(im_model) * l / 100 for l in levs]
        cs = ax.contour(
            xx / mas, yy / mas, im_model, levels=levels, cmap=cmap, zorder=3
        )

    elif centroid is not None:
        if corotate:
            centroid[0], centroid[1] = f_rot1(centroid[0], centroid[1], phi - np.pi / 2)

            new_sigma_x = (
                np.abs(np.cos(phi - np.pi / 2)) * centroid[2]
                + np.abs(np.sin(phi - np.pi / 2)) * centroid[3]
            )
            new_sigma_y = (
                np.abs(np.sin(phi - np.pi / 2)) * centroid[2]
                + np.abs(np.cos(phi - np.pi / 2)) * centroid[3]
            )
            centroid[2] = new_sigma_x
            centroid[3] = new_sigma_y

        ax.errorbar(
            centroid[0],
            centroid[1],
            xerr=centroid[2],
            yerr=centroid[3],
            marker=marker,
            markersize=5,
            linestyle="None",
            color=color,
            zorder=2,
            label=label,
        )

    if beam is not None:
        fwhm_maj, fwhm_min, angle = beam
        if corotate:
            angle += phi - np.pi / 2
        aux_tr_box = offsetbox.AuxTransformBox(ax.transData)
        aux_tr_box.add_artist(
            patches.Ellipse(
                (0, 0),
                facecolor="gray",
                edgecolor=None,
                alpha=0.5,
                width=fwhm_min / mas,
                height=fwhm_maj / mas,
                angle=-angle / deg,
            )
        )
        box = offsetbox.AnchoredOffsetbox(
            child=aux_tr_box, loc="lower right", frameon=True
        )
        ax.add_artist(box)
        box.set_clip_box(ax.bbox)

    if d is not None:
        angle = np.arctan(2 * R_sun / d) / mas
        asb = AnchoredSizeBar(
            ax.transData,
            angle,
            r"2$R_{\odot}$",
            loc=bar_pos,
            frameon=False,
        )
        ax.add_artist(asb)

    ax.set_aspect("equal")

    ax.tick_params(axis="both", which="major", color="darkgray")

    ax.set_xlim(mapsize / 2, -mapsize / 2)
    ax.set_ylim(-mapsize / 2, mapsize / 2)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))

    if not corotate:
        if show_coord_axes:
            cm = binary.coord_cm(jd)
            ra = (
                cm.ra.to_string(unit="hourangle", sep=" ", precision=6, pad=True)
                .replace(".", " ")
                .split()
            )
            ra = r"$%s^{\rm h} %s^{\rm m} %s\overset{{\rm s}}{.}%s$" % (
                ra[0],
                ra[1],
                ra[2],
                ra[3],
            )
            dec = (
                cm.dec.to_string(
                    unit="degree", sep=" ", precision=5, pad=True, alwayssign=True
                )
                .replace(".", " ")
                .split()
            )
            dec = r"$%s^{\circ} %s^{\prime} %s\overset{\prime\prime}{.}$%s" % (
                dec[0],
                dec[1],
                dec[2],
                dec[3],
            )
            ax.set_xlabel(
                r"$\Delta \alpha$ (mas) from %s [GCRF3]" % ra, fontsize=fontsize
            )
            ax.set_ylabel(
                r"$\Delta \delta$ (mas) from %s [GCRF3]" % dec, fontsize=fontsize
            )
        else:
            ax.set_xlabel(r"$\Delta \alpha$ (mas)")
            ax.set_ylabel(r"$\Delta \delta$ (mas)")
    else:
        ax.set_xlabel(r"$\Delta \alpha^{\prime}$ (mas)")
        ax.set_ylabel(r"$\Delta \delta^{\prime}$ (mas)")


class vlb_model:
    def __init__(self, model_file, restoring_beam):
        self.model_file = model_file
        self.restoring_beam = restoring_beam
        self.fwhm_maj, self.fwhm_min, self.beam_phi = self.restoring_beam
        self.x_shift = 0
        self.y_shift = 0
        self.read_model(self.restoring_beam)

    def read_model(self, restoring_beam):
        self.restoring_beam = restoring_beam
        """ Read model (.mod) file from Difmap, convert to xy coordinates, 
        sorts by flux, returns (I, x, y)
        N.B. currectly ignores last 3 columns [FWHM, AR, theta] """
        header = []
        gauss_params = []
        if os.path.isfile(self.model_file):
            f = open(self.model_file, "r")
            lines = f.readlines()
            for line in lines:
                if line[0] != "#" and line[0] != "!":
                    x = [float(s) for s in line.strip().split()]
                    x[0] *= Jy  # Convert to mJy
                    x[1] *= mas  # Convert to radians
                    x[2] *= deg  # Convert to radians
                    gauss_params.append(x)
                else:
                    header.append(line)
            f.close()
        else:
            sys.exit("Cannot find file %s, try again, exiting" % fname)

        # Convert to x,y coordinates
        for j, p in enumerate(gauss_params):
            I, r, phi = p[0:3]
            x, y = [t for t in polar2rect(r, phi)]
            gauss_params[j][1:3] = x, y

        self.gauss_params = gauss_params
        self.header = header

    def excise_components(self, n):
        """Keep n highest flux components"""
        self.gauss_params = self.gauss_params[:n]

    def shift(self, x_shift, y_shift):
        """Shift model by x_shift, y_shift"""
        for i in range(len(self.gauss_params)):
            self.gauss_params[i][1] += x_shift
            self.gauss_params[i][2] += y_shift
        self.x_shift += x_shift
        self.y_shift += y_shift

    def reset_shift(self):
        """Reset shift to zero"""
        self.shift(-self.x_shift, -self.y_shift)

    def single_component_ad(self, x, y, I, x0, y0):
        phi = self.beam_phi + np.pi / 2  # Rotate by 90 deg to match Difmap convention
        sigma_x = self.fwhm_maj / (2 * np.sqrt(2 * np.log(2)))  # FWHM -> sigma
        sigma_y = self.fwhm_min / (2 * np.sqrt(2 * np.log(2)))  # FWHM -> sigma

        cos = np.cos(phi)
        sin = np.sin(phi)
        X = (((x - x0) * cos - (y - y0) * sin) / sigma_x) ** 2
        Y = (((x - x0) * sin + (y - y0) * cos) / sigma_y) ** 2

        return I * np.exp(-(X + Y))

    def get_image_plane_model(self, x, y, x_shift=0, y_shift=0):
        im_models = []
        for i in range(len(self.gauss_params)):
            I, x0, y0 = self.gauss_params[i]
            im_models.append(
                self.single_component_ad(x, y, I, x0 + x_shift, y0 + y_shift)
            )
        clean_model = np.sum(im_models, axis=0)
        clean_model *= np.sum([p[0] for p in self.gauss_params]) / np.sum(clean_model)
        return clean_model

    def plot_model(
        self,
        mapsize=None,
        cells=1024,
        levs=[-1, 1, 2, 4, 8, 16, 32, 64],
        cmap="viridis",
        x_shift=0,
        y_shift=0,
    ):
        if mapsize == None:
            mapsize = 2 * np.max(
                np.sqrt(np.array([p[1] ** 2 + p[2] ** 2 for p in self.gauss_params]))
            )
        x = np.linspace(-mapsize, mapsize, cells, endpoint=False)
        y = np.linspace(-mapsize, mapsize, cells, endpoint=False)
        x, y = np.meshgrid(x, y)
        m2 = mapsize / (2 * mas)
        im_model = self.get_image_plane_model(x, y, x_shift=x_shift, y_shift=y_shift)

        fig, ax = plt.subplots(figsize=(7, 7))
        levels = [np.max(im_model) * i / 100 for i in levs]
        ax.contour(x / mas, y / mas, im_model, levels=levels, cmap=cmap, linewidths=1)
        ax.set_xlabel(r"$\Delta\alpha$ [mas]")
        ax.set_ylabel(r"$\Delta\delta$ [mas]")
        ax.set_xlim(m2, -m2)
        ax.set_ylim(-m2, m2)
        ax.set_title("Clean model: %s" % self.model_file)
        s = "Levels = " + ", ".join(map(str, levs)) + " mJy"
        plt.text(0.02, 0.02, s, font="tahoma", transform=ax.transAxes)
        ax.grid()

        return fig, ax
