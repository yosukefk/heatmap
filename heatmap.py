# import pandas as pd
import numpy as np
import numpy.typing as npt

from scipy.stats import norm
import warnings


def cart2pol(x: npt.ArrayLike, y: npt.ArrayLike) -> (npt.ArrayLike, npt.ArrayLike,):
    """
    cartesian to polar coords

    :param x: Cartesian x coords
    :param y: Cartesian y coords
    :return: tuple of polar radius and angle coords
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho: npt.ArrayLike, phi: npt.ArrayLike) -> (npt.ArrayLike, npt.ArrayLike,):
    """
    polar to cartesian coords

    :param rho: polar radius
    :param phi: polar angle
    :return: tuple of x, y coords
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


class CalcPdf:
    def __init__(self, theta0: float, sd: float, rmax: float, ncel: int = None, halfrng: float = None,
                 mass_balance: bool = False):
        """

        :param theta0: wind direction
        :param sd: std deviation of wind direction
        :param rmax: traveled radius at the end
        :param ncel: number of cell along a dimension
        :param halfrng: half of length of a side
        :param mass_balance: True to conserve mass radially, False to use naive pdf
        """
        if ncel % 2 == 0:
            warnings.warn(f'ncel changed to {ncel + 1}')
            ncel += 1

        # wind info
        self.theta0 = theta0
        self.sd = sd
        self.rmax = rmax

        # grid info
        self.halfrng = halfrng
        self.ncel = ncel
        self.offs = (self.ncel - 1) / 2
        self.res = halfrng * 2 / (ncel - 1)

        # coordinates of dot points (cell centroids)
        self.coords1d = np.linspace(-halfrng, halfrng, self.ncel)
        xx, yy = np.meshgrid(self.coords1d, self.coords1d)
        self.zeros = np.zeros_like(xx)
        self.masked_zeros = np.ma.masked_values(self.zeros, 0)

        # hold on to it
        self.xx = xx
        self.yy = yy

        # coordinates of cross points (cell corners)
        self.coords1dc = np.linspace(-halfrng - .5 * self.res, halfrng + .5 * self.res, self.ncel + 1)
        xxc, yyc = np.meshgrid(self.coords1dc, self.coords1dc)

        # TODO, this is unnecessary, i need only rhoc, and also phic near the origin
        rhoc, phic = cart2pol(xxc, yyc)

        # coordinates in polar
        rho, phi = cart2pol(xx, yy)
        self.rho, self.phi = rho, phi

        # deviation from wind centerline
        theta = phi - self.theta0
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # almost, DONE!  "pseudo" probability at each coords...
        p = norm.pdf(theta, scale=self.sd)
        if mass_balance:
            p /= rho
        no = (self.ncel - 1) // 2
        np.set_printoptions(edgeitems=10)
        np.core.arrayprint._line_width = 180
        # print('no', no)
        # print('shp', p.shape)
        # print('argmax', np.unravel_index(p.argmax(), p.shape))
        # print('phi', phi[(no-2):(no+3), (no-2):(no+3)])
        # print('theta', theta[(no-2):(no+3), (no-2):(no+3)])
        # print('p', p[(no-2):(no+3), (no-2):(no+3)])

        # do better job near the origin
        # at origin density shoud evaluate 1
        no = (self.ncel - 1) // 2
        p[no, no] = 1
        # print('p', p[(no-2):(no+3), (no-2):(no+3)])

        # for 3x3 cells around the origin, integrate using values at cross points
        thetac = phic[(no - 1):(no + 3), (no - 1):(no + 3)] - self.theta0
        thetac = (thetac + np.pi) % (2 * np.pi) - np.pi
        pc = norm.pdf(thetac, scale=self.sd)
        # print('pc', pc)
        if mass_balance:
            pc /= rhoc[(no - 1):(no + 3), (no - 1):(no + 3)]
        p[(no - 1):(no + 2), (no - 1):(no + 2)] = (
                .5 * p[(no - 1):(no + 2), (no - 1):(no + 2)] +
                .125 * pc[:-1, :-1] +
                .125 * pc[:-1, 1:] +
                .125 * pc[1:, :-1] +
                .125 * pc[1:, 1:]
        )
        # print('p', p[(no-2):(no+3), (no-2):(no+3)])
        # raise

        # mask out the protion beyond the rmax
        print('self.rmax', self.rmax)
        self.p = np.ma.masked_where(rho > self.rmax, p)
        # prep for next step:  identify where the front is, and distribution  at front

        # locate whre the front is (circle from the orign with radius rmax)

        # distance to cell corners
        chk = rhoc - rmax

        # for each cell, max and min distance
        chk = np.dstack([chk[:-1, :-1], chk[:-1, 1:], chk[1:, :-1], chk[1:, 1:]])
        chk0 = chk.min(axis=-1)
        chk1 = chk.max(axis=-1)

        # grab cells where sign changes.  the circle passes through the cell
        chkk = chk0 * chk1
        front = chkk <= 0

        # pseudo density along the front (to be used as factor in next step)
        # cut out the pseudo probability at the circle
        # note that i am using unmasked p, just to be sure values i want is not masked (beyond rmax)
        self.dens = np.where(front, p, 0)
        self.dens = np.ma.masked_values(self.dens, 0)

        # calibrate so that sum across the front becomes 2*pi*r
        # self.dens = self.dens * ( 2  * np.pi * rmax / self.dens.sum() )

    @staticmethod
    def clean_array(arr: npt.NDArray, frac_keep: float = .99) -> npt.NDArray:
        """
        Clean Array of PDF to mask out low probability parts

        :param arr: array of pdf
        :param frac_keep: fraction of density to be kept, selecting from high value to low
        :return: cleaned array of pdf
        """
        if isinstance(arr, np.ma.MaskedArray):
            a = arr[~arr.mask]
            if a.size == 0:
                return arr
        else:
            a = arr[...]
        s = np.sort(a.flatten())
        thres = s[(s.cumsum() >= (1 - frac_keep) * (s.sum())).argmax()]

        arr = np.ma.masked_where(arr < thres, arr)

        arr /= frac_keep  # adjust so that total of input dens is unchanged

        return arr

    def slice_array(self, arr: npt.NDArray, x0: float, y0: float) -> npt.NDArray:
        """
        Extract pdf around a given point

        :param arr: array of pdf
        :param x0: x coord of centroid
        :param y0: y coord of centroid
        :return: array of pdf around x0, y0
        """
        # cell offsets (note i am flipping sign of x0, y0)
        i0, j0 = [int(np.round(_ / self.res)) for _ in (-x0, -y0)]
        # print(j0,j0)

        if isinstance(arr, np.ma.MaskedArray):
            # special treatment for masked array
            is_masked = True
            o = self.masked_zeros.copy()
            m = arr.mask
        else:
            is_masked = False
            o = self.zeros.copy()

        if i0 > 0:
            islice_in = slice(i0, None)
            islice_out = slice(None, -i0)
        elif i0 < 0:
            islice_in = slice(None, i0)
            islice_out = slice(-i0, None)
        else:
            islice_in = slice(None, None)
            islice_out = slice(None, None)
        if j0 > 0:
            jslice_in = slice(j0, None)
            jslice_out = slice(None, -j0)
        elif j0 < 0:
            jslice_in = slice(None, j0)
            jslice_out = slice(-j0, None)
        else:
            jslice_in = slice(None, None)
            jslice_out = slice(None, None)

        # slice
        o[jslice_out, islice_out] = arr[jslice_in, islice_in, ]

        if is_masked:

            # copy over the mask from the source
            try:
                o.mask[jslice_out, islice_out] = m[jslice_in, islice_in]
            except IndexError:
                o.mask[jslice_out, islice_out] = m

        return o

    def calc_from_pnt(self, x0: float, y0: float) -> (npt.NDArray, npt.NDArray):
        """
        pdf generated around point (x0, y0)

        returns two arrays of cumulative pdf from t = [0, t], and the pdf at the end (t = t)

        :param x0: x coord of pdf origin
        :param y0: y coord of pdf origin
        :return: typle of translated array of pdf (for t = [0,  t]), and density of front of mass (t = t)
        """
        # translate self.p, self.dense array and return
        p = self.slice_array(self.p, x0, y0)
        dens = self.slice_array(self.dens, x0, y0)
        dens = self.clean_array(dens)

        return np.ma.filled(p, 0), dens

    def calc_from_dens(self, dens: npt.NDArray) -> (npt.NDArray, npt.NDArray):
        """
        pdf generated starting with initial distribution of density

        returns two arrays of cumulative pdf from t = [0, t], and the pdf at the end (t = t)

        :param dens: array of pdf at the begining
        :return: typle of translated array of pdf (for t = [0,  t]), and density of front of mass (t = t)
        """
        # filter density to capture 99% of total picking from top

        p2 = self.zeros.copy()
        dens2 = self.zeros.copy()

        # go over cells where dens is defined
        for f, x0, y0 in zip(dens[~dens.mask], self.xx[~dens.mask], self.yy[~dens.mask]):
            # print(dens.max(), f, x0, y0)

            # grab p and (next) dense starting from a point
            p, d = self.calc_from_pnt(x0, y0)

            # cummulate p and dens
            p2 += (np.ma.filled(p, 0) * f)
            dens2 += (np.ma.filled(d, 0) * f)

        # dens is masked
        dens2 = np.ma.masked_values(dens2, 0)
        # clean further
        dens2 = self.clean_array(dens2)
        return p2, dens2


def backward_trajectory(theta: npt.NDArray, sd: npt.NDArray, rmax: npt.NDArray, ncel: int = None, halfrng: float = None,
                        mass_balance: bool = None, return_list: bool = False,
                        return_obj: bool = False) -> object:
    """
    Given time sereis of wind, pdf is successively applied to estimate back trajectory for the period

    :param theta: series of wind direction
    :param sd: series of stdev of wind direction
    :param rmax: series of distance travled
    :param ncel: number of cells along an 1d axis
    :param halfrng: half of the length of a side
    :param mass_balance: True to conserve mass radially, False to use naive pdf
    :param return_list: True to return list of density array, False to return single array
    :param return_obj: True to return tupe of array and the CalCPDF object
    :return:
    """
    assert len(theta) == len(sd) and len(theta) == len(rmax)

    if mass_balance is None:
        mass_balance = False

    calculator = CalcPdf(theta[0], sd[0], rmax[0], ncel=ncel, halfrng=halfrng, mass_balance=mass_balance)

    if return_list:
        cum_pp = []
    else:
        cum_pp = calculator.zeros.copy()

    dd = None
    for th, s, r in zip(theta, sd, rmax):
        if any((th != calculator.theta0, s != calculator.sd, r != calculator.rmax)):
            calculator = CalcPdf(th, s, r, ncel=ncel, halfrng=halfrng, mass_balance=mass_balance)

        if dd is None:
            pp, dd = calculator.calc_from_pnt(0, 0)
        else:
            # #### NOTE normalizing the initial densitiy here... #####
            # #### not absolutetly sure if i should do this      #####
            print(dd.sum())
            pp, dd = calculator.calc_from_dens(dd / dd.sum())

        if return_list:
            cum_pp.append(pp)
        else:
            cum_pp += pp

    if return_obj:
        return cum_pp, calculator
    else:
        return cum_pp


def superpose_trajectories(theta: npt.NDArray, sd: npt.NDArray, rmax: npt.NDArray, nbackward: int = 1, ncel: int = None,
                           halfrng: float = None, mass_balance: bool = None,
                           return_list: bool = False,
                           return_obj: bool = False) -> object:
    """
    Given time sereis of wind, pdf is generated for each period and superposed

    :param theta: series of wind direction
    :param sd: series of stdev of wind direction
    :param rmax: series of distance travled
    :param nbackward: number of steps to go back trajectory
    :param ncel: number of cells along an 1d axis
    :param halfrng: half of the length of a side
    :param mass_balance: True to conserve mass radially, False to use naive pdf
    :param return_list: True to return list of density array, False to return single array
    :param return_obj: True to return tupe of array and the CalCPDF object
    :return:
    """
    assert len(theta) == len(sd) and len(theta) == len(rmax)

    n = len(theta)
    if return_list:
        cum_pp = []
    else:
        cum_pp = None
    obj0 = None
    for i in range(0, n - nbackward + 1):
        slc = slice(i, i + nbackward )
        pp = backward_trajectory(theta[slc], sd[slc], rmax[slc], ncel, halfrng, mass_balance, return_list=False,
                                 return_obj=return_obj)
        if return_obj:
            pp, obj = pp
            if obj0 is None:
                obj0 = obj

        if return_list:
            cum_pp.append(pp)
        else:
            if cum_pp is None:
                cum_pp = pp
            else:
                cum_pp += pp
    if return_obj:
        return cum_pp, obj0
    else:
        return cum_pp
