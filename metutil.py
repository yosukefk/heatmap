import numpy as np
import numpy.typing as npt


def arr_uuvv2wdws(uuvv: npt.NDArray) -> npt.NDArray:
    """
    Given 2d array of u/v wind components, returns 2d array of wind direction/speed

    :param uuvv: 2d array of u/v wind components
    :return: 2d array of wind direction/speed
    """
    if len(uuvv.shape) == 1:
        return np.array(uuvv2wdws(uuvv[0], uuvv[1]))
    uu = uuvv[:, 0]
    vv = uuvv[:, 1]
    wd, ws = uuvv2wdws(uu, vv)

    return np.array([wd, ws]).T


def arr_wdws2uuvv(wdws: npt.NDArray) -> npt.NDArray:
    """
    Given 2d array of wind direction/speed, returns 2d array of u/v wind components

    :param wdws: 2d array of wind direction/speed
    :return:  2d array of u/v wind components
    """
    if len(wdws.shape) == 1:
        return np.array(wdws2uuvv(wdws[0], wdws[1]))

    wd = wdws[:, 0]
    ws = wdws[:, 1]
    uu, vv = wdws2uuvv(wd, ws)

    return np.array([uu, vv]).T


def arr_average_wdws(wdws: npt.NDArray) -> npt.NDArray:
    """
    Given 2d vector of wind direction/speed, average them

    :param wdws: 2d array of wind direction/speed
    :return: 1d array of wind direction/speed
    """
    if len(wdws.shape) == 1:
        return np.array(average_wdws(wdws[0], wdws[1]))
    wd = wdws[:, 0]
    ws = wdws[:, 1]
    return np.array(average_wdws(wd, ws))


def uuvv2wdws(uu: npt.ArrayLike, vv: npt.ArrayLike) -> (npt.ArrayLike, npt.ArrayLike):
    """
    Given u/v wind components calculate wind direction/speed

    :param uu: u wind component
    :param vv: v wind component
    :return: tuple of wind direction/speed
    """
    wd = ((np.arctan2(uu, vv) * 180 / np.pi) - 180) % 360
    ws = np.sqrt(uu * uu + vv * vv)
    return wd, ws


def wdws2uuvv(wd: npt.ArrayLike, ws: npt.ArrayLike) -> (npt.ArrayLike, npt.ArrayLike):
    """
    Given wdir/spd, calculate u/v wind components

    :param wd: wind direction
    :param ws: wind speed
    :return: tuple of u and v wind components
    """
    theta = np.pi * (270 - wd) / 180
    uu = ws * np.cos(theta)
    vv = ws * np.sin(theta)
    return uu, vv


def average_wdws(wd: npt.ArrayLike, ws: npt.ArrayLike) -> (npt.ArrayLike, npt.ArrayLike):
    """
    Average Wind speed/direction

    :param wd: wind direction
    :param ws: wind speed
    :return: tuple of average wind direction/speed
    """
    uu, vv = wdws2uuvv(wd, ws)
    uu = uu.mean()
    vv = vv.mean()
    wd, ws = uuvv2wdws(uu, vv)
    return wd, ws


def tester():
    wd = np.array([0, 30, 90, 120, 180, 210, 270, 300])
    ws = 1
    print(wd, ws)
    uu, vv = wdws2uuvv(wd, ws)
    print(uu, vv)
    wd, ws = uuvv2wdws(uu, vv)
    print(wd, ws)

    print(wd[1:3])
    print(ws[1:3])
    awd, aws = average_wdws(wd[1:3], ws[1:3])
    print(awd, aws)

    wdws = np.stack([wd, ws]).T
    print(wdws)
    awdws = arr_average_wdws(wdws)
    print(awdws)


if __name__ == '__main__':
    tester()
