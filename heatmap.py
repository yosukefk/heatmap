from numpy import typing as npt

from pdf import CalcPdf


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
        slc = slice(i, i + nbackward)
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