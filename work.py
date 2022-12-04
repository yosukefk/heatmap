import heatmap
import pdf as pdf
import metutils

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap
from matplotlib import ticker

from plotter import plotter_solo as psolo
from plotter.plotter_background import BackgroundManager
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


class MetReader:
    def __init__(self, fname='Data_for_heat_map_toYo1201.csv'):
        self.df = pd.read_csv(fname)
        try:
            self.df.timestamp = pd.to_datetime(self.df.timestamp, utc=True)
        except AttributeError:
            self.df['timestamp'] = pd.to_datetime(self.df.timestamp_min, utc=True)
        self.zeromin = pd.Timedelta(0, 'min')

    def read(self, start_time, total_minutes, subdiv_minutes=1):

        tdiff = self.df.timestamp - start_time

        rec = self.df.loc[(tdiff > pd.Timedelta(-total_minutes, 'min')) & (tdiff <= self.zeromin), :]

        # print(len(rec.index), total_minutes)

        # assert len(rec.index) == total_minutes

        rec = rec.loc[:, ('wspd', 'wdir', 'wd_std', 'timestamp')]

        if subdiv_minutes > 1:
            # repeat each records 3 times, except the start time which is not repeated
            rec = rec.iloc[np.arange(total_minutes).repeat(subdiv_minutes), :][:(1 - subdiv_minutes)]

            # for time stamp, interpolate
            td = np.tile([pd.Timedelta(_, 'sec') for _ in (np.arange(subdiv_minutes) * 60 / subdiv_minutes)],
                         total_minutes)[:(1 - subdiv_minutes)]
            rec['timestamp'] += td

        # reverse the order
        rec = rec[-1::-1]

        return rec


# color scale on log
def mk_color(arr):
    amax = 10 ** np.ceil(np.log10(arr.max()))
    norm = LogNorm(vmax=amax, vmin=amax / 1000, clip=True)

    bdry = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]) / 1000 * amax
    nc = len(bdry)
    viridis = plt.cm.viridis
    vv = [np.round(_ / nc * viridis.N) for _ in range(nc)]
    # vv = [ viridis(np.round(_/nc*viridis.N)) for _ in range(nc)]
    cm = ListedColormap([viridis(int(np.round(_ / nc * viridis.N))) for _ in range(nc)])

    bnorm = BoundaryNorm(bdry, len(bdry), clip=True)
    return norm, bnorm, bdry, cm


# plot with pylab
def mkplt_pylab(arr, fname, halfrng, ttle=None, contour=False, *args, **kwds):
    plt.clf()
    if contour:
        plt.contourf(arr[:, :], extent=[-halfrng, halfrng, -halfrng, halfrng], *args, **kwds)
    else:
        plt.imshow(arr[-1::-1, :], extent=[-halfrng, halfrng, -halfrng, halfrng], *args, **kwds)
    plt.colorbar()
    plt.xlabel('(meters)')
    plt.ylabel('(meters)')
    if not ttle is None:
        plt.title(ttle)

    plt.savefig(fname)


# plot with plotter
def mkplt_plotter(arr, fname, halfrng, lnlt0, start_time, ttle=None, *args, **kwds):
    ncel = arr.shape[-1]
    coords1d = np.linspace(-halfrng, halfrng, ncel)
    ln0, lt0 = lnlt0

    prj = ccrs.LambertConformal(
        central_longitude=ln0, central_latitude=lt0,
        standard_parallels=(lt0, lt0), globe=ccrs.Globe())
    arrx = np.expand_dims(arr, axis=0)
    contour_options = {**{'alpha': .5},
                       **{_: kwds[_] for _ in ('norm', 'levels', 'cmap') if _ in kwds}}
    p = psolo.Plotter(arrx, tstamps=[start_time], x=coords1d, y=coords1d, projection=prj,
                      plotter_options={
                          'contour_options': contour_options,
                          'background_manager': BackgroundManager(
                              add_image_options=[cimgt.GoogleTiles(style='satellite', cache=True)],
                          ),
                      })
    p.savefig(fname)


def tester():
    met_reader = MetReader()
    # sensor location
    ln0, lt0 = -102.072247, 31.893615
    x0, y0 = 0, 0

    # start of simulation
    start_time = pd.to_datetime('2022-11-09 19:10:00+00:00')

    ncel = 201
    halfrng = 500

    total_minutes = 20  # total min going backward
    subdiv_minutes = 1  # subdivision for min:  1= no subdim, 2= 30sec each, 3= 20sec each etc...

    summary_title = 'S2XX-X10NE'

    rec = met_reader.read(start_time, total_minutes, subdiv_minutes)

    print(rec['wdir'])

    wd_mean, ws_mean = metutil.average_wdws(rec['wdir'], rec['wspd'])
    print('mean ws', ws_mean)
    print('mean wd', wd_mean)
    print('mean sd wd', rec['wd_std'].mean())

    # stride
    r = rec['wspd'] * 60. / subdiv_minutes

    # wind vector on cartesian coords, but revised sign
    theta = np.pi / 180. * ((90. - rec['wdir']) % 360.)

    # change unit
    sd = np.pi * rec['wd_std'] / 180.

    print(theta)

    # make them into numpy array...
    r, theta, sd, dte = [_.array for _ in (r, theta, sd, rec['timestamp'])]

    # small demo
    if False:
        calculator = pdf.CalcPdf(theta[0], sd[0], r[0], ncel=ncel, halfrng=halfrng)
        pp1, dd1 = calculator.calc_from_pnt(0, 0)
        # dd1 = calculator.clean_array(dd1)
        pp2, dd2 = calculator.calc_from_dens(dd1 / dd1.sum())
        # dd2 = calculator.clean_array(dd2)
        pp3, dd3 = calculator.calc_from_dens(dd2 / dd2.sum())
        arr = calculator.zeros.copy()
        arr[(ncel - 1) // 2, (ncel - 1) // 2] = 1.0
        mkplt_pylab(arr, 'demo_d0.png', halfrng, 'prep first step')
        mkplt_pylab(pp1, 'demo_p1.png', halfrng, 'end first step')
        mkplt_pylab(dd1, 'demo_d1.png', halfrng, 'prep second step')
        mkplt_pylab(pp2, 'demo_p2.png', halfrng, 'end second step')
        mkplt_pylab(dd2, 'demo_d2.png', halfrng, 'prep third step')
        mkplt_pylab(pp3, 'demo_p3.png', halfrng, 'end third step')

    #  backtrajectory
    if False:
        arrays = heatmap.backward_trajectory(theta, sd, r, ncel=ncel, halfrng=halfrng, return_list=True)

        for i, a in enumerate(arrays):
            fname = f'fig{i:02d}.png'
            ttle = f'{dte[i]}\nws {rec["wspd"].iloc[i]} m/sec, wd {rec["wdir"].iloc[i]} deg, sd_wd {rec["wd_std"].iloc[i]} deg'
            mkplt_pylab(a, fname, halfrng, ttle)

        from functools import reduce
        mkplt_pylab(reduce(np.add, arrays), 'fig.png', halfrng, summary_title)

    # superposition mass balance on
    if True:
        print('here!')
        # FLIPME!!
        mass_balance = True

        arrays, obj = heatmap.superpose_trajectories(theta, sd, r, ncel=ncel, halfrng=halfrng, mass_balance=mass_balance,
                                                     return_list=True, return_obj=True)

        for i, a in enumerate(arrays):
            fname = f'fig{i:02d}.png'
            ttle = f'{dte[i]}\nws {rec["wspd"].iloc[i]} m/sec, wd {rec["wdir"].iloc[i]} deg, sd_wd {rec["wd_std"].iloc[i]} deg'
            if mass_balance:
                mkplt_pylab(a, fname, halfrng, ttle, norm=LogNorm(vmax=1, vmin=0.001, clip=True))
            else:
                mkplt_pylab(a, fname, halfrng, ttle)

        from functools import reduce
        arr = reduce(np.add, arrays) / len(theta)

        if mass_balance:
            amax = 10 ** np.ceil(np.log10(arr.max()))
            norm = LogNorm(vmax=amax, vmin=amax / 1000, clip=True)

            bdry = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]) / 1000 * amax
            nc = len(bdry)
            viridis = plt.cm.viridis
            vv = [np.round(_ / nc * viridis.N) for _ in range(nc)]
            # vv = [ viridis(np.round(_/nc*viridis.N)) for _ in range(nc)]
            cm = ListedColormap([viridis(int(np.round(_ / nc * viridis.N))) for _ in range(nc)])

            bnorm = BoundaryNorm(bdry, len(bdry), clip=True)
            # locator=ticker.LogLocator()
            # locator.nonsingular(vmax=amax, vmin=amax/1000)
            mkplt_pylab(arr, 'tile.png', halfrng, summary_title, contour=False, norm=norm)

            mkplt_pylab(np.maximum(arr, 0.001), 'contour.png', halfrng, summary_title, contour=True, norm=bnorm,
                        levels=bdry, cmap=cm)

            # plotter...
            x = obj.coords1d
            y = obj.coords1d
            prj = ccrs.LambertConformal(
                central_longitude=ln0, central_latitude=lt0,
                standard_parallels=(lt0, lt0), globe=ccrs.Globe())
            arrx = np.expand_dims(arr, axis=0)
            p = psolo.Plotter(arrx, tstamps=[start_time], x=x, y=y, projection=prj,
                              plotter_options={
                                  'contour_options': {
                                      'norm': bnorm,
                                      'levels': bdry,
                                      'cmap': cm,
                                      'alpha': .5,
                                  },
                                  'background_manager': BackgroundManager(
                                      add_image_options=[cimgt.GoogleTiles(style='satellite'), 18],
                                  ),
                              })
            p.savefig('w_bg.png')

        else:
            mkplt_pylab(arr, 'tile.png', halfrng, summary_title, contour=False)

            mkplt_pylab(arr, 'contour.png', halfrng, summary_title, contour=True)


def main(metfname, oroot, start_time, total_minutes, lnlt0=None, halfrng=None, ncel=201,
         subdiv_minutes=1, nbackward=1, summary_title=None, mass_balance=True):
    met_reader = MetReader(metfname)

    # sensor location
    # ln0, lt0 = lnlt0

    x0, y0 = 0, 0

    ## start of simulation
    # start_time = pd.to_datetime('2022-11-09 19:10:00+00:00' )

    # wspd * (nbackward+1) / subdiv_minutes * 60
    # ncel = 201
    # halfrng = 500

    # total_minutes = 20 # total min going backward
    # subdiv_minutes = 1 # subdivision for min:  1= no subdim, 2= 30sec each, 3= 20sec each etc...

    # summary_title = 'S2XX-X10NE'

    rec = met_reader.read(start_time, total_minutes, subdiv_minutes)

    print(rec['wdir'])

    wd_mean, ws_mean = metutil.average_wdws(rec['wdir'], rec['wspd'])
    print('mean ws', ws_mean)
    print('mean wd', wd_mean)
    print('mean sd wd', rec['wd_std'].mean())

    if halfrng is None:
        halfrng = np.around(ws_mean * nbackward / subdiv_minutes * 60, decimals=-2)
        halfrng = max(halfrng, 100)
        print('halfrng', halfrng)

    # stride
    r = rec['wspd'] * 60. / subdiv_minutes

    # wind vector on cartesian coords, but revised sign
    theta = np.pi / 180. * ((90. - rec['wdir']) % 360.)

    # change unit
    sd = np.pi * rec['wd_std'] / 180.

    print(theta)

    # make them into numpy array...
    r, theta, sd, dte = [_.array for _ in (r, theta, sd, rec['timestamp'])]

    arrays, obj = heatmap.superpose_trajectories(theta, sd, r,
                                                 nbackward=nbackward, ncel=ncel, halfrng=halfrng,
                                                 mass_balance=mass_balance, return_list=True, return_obj=True)

    for i, a in enumerate(arrays):
        fname = f'{oroot}_fig{i:02d}.png'
        ttle = f'{dte[i]}\nws {rec["wspd"].iloc[i]:4.1f} m/sec, wd {rec["wdir"].iloc[i]:3.0f} deg, sd_wd {rec["wd_std"].iloc[i]:4.1f} deg'
        if mass_balance:
            mkplt_pylab(a, fname, halfrng, ttle, norm=LogNorm(vmax=1, vmin=0.001, clip=True))
        else:
            mkplt_pylab(a, fname, halfrng, ttle)

    from functools import reduce
    arr = reduce(np.add, arrays) / len(theta)

    if mass_balance:

        norm, bnorm, bdry, cm = mk_color(arr)

        mkplt_pylab(arr, f'{oroot}_tile.png', halfrng, summary_title, contour=False, norm=norm)

        mkplt_pylab(np.maximum(arr, 0.001), f'{oroot}_contour.png', halfrng, summary_title, contour=True, norm=bnorm,
                    levels=bdry, cmap=cm)

        # plotter...
        if not lnlt0 is None:
            mkplt_plotter(arr, f'{oroot}_w_bg.png', halfrng, lnlt0, start_time, norm=bnorm, levels=bdry, cmap=cm)


    else:
        mkplt_pylab(arr, f'{oroot}_tile.png', halfrng, summary_title, contour=False)

        mkplt_pylab(arr, f'{oroot}_contour.png', halfrng, summary_title, contour=True)


if __name__ == '__main__':
    #    tester()
    ##raise

    # site_name = 'S2XX-X10NE'
    # ln0, lt0 = -102.072247, 31.893615
    # main(
    #        'Data_for_heat_map_toYo1201.csv', 
    #        'nov09', 
    #        start_time = pd.to_datetime('2022-11-09 19:10:00+00:00' ), 
    #        total_minutes = 20, # total min going backward
    #        lnlt0 = (ln0, lt0),
    #        summary_title = site_name, 
    # )
    # S2Q1-Y25NE,31.875034,-102.100081,Low (I),S2Q1,Y25NE,Scientific Aviation (I)

    site_name = 'S2Q1-Y25NE'
    ln0, lt0 = -102.100081, 31.875034
    main(
        'Nov12Event_MET.csv',
        'nov09_y25ne',
        start_time=pd.to_datetime('2022-11-12 14:10:00+00:00'),
        total_minutes=25,  # total min going backward
        nbackward=2,  # back track two minutes
        lnlt0=(ln0, lt0),
    )
