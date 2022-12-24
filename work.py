import heatmap as hm
import pdf as pdf
import metutils

from importlib import reload
reload(hm)
reload(pdf)

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap
from matplotlib import ticker
mpl.use('Agg')
# to match with standard mp4
mpl.pylab.rcParams["figure.figsize"] = (4.8, 3.2)
mpl.pylab.rcParams["figure.dpi"] = 300

from plotter import plotter_solo as psolo
from plotter.plotter_background import BackgroundManager
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

import tempfile
import subprocess
import shlex
from pathlib import Path


class MetReader:
    def __init__(self, fname):
        self.df = pd.read_csv(fname)
        try:
            self.df['datetime'] = pd.to_datetime(self.df.timestamp, utc=True)
        except AttributeError:
            self.df['datetime'] = pd.to_datetime(self.df.timestamp_min, utc=True)
        self.zeromin = pd.Timedelta(0, 'min')

    def read(self, start_time, total_minutes, subdiv_minutes=1):

        tdiff = self.df.datetime - start_time

        rec = self.df.loc[(tdiff > pd.Timedelta(-total_minutes, 'min')) & (tdiff <= self.zeromin), :]

        print(rec)
        # print(len(rec.index), total_minutes)

        # assert len(rec.index) == total_minutes

        rec = rec.loc[:, ('wspd', 'wdir', 'wd_std', 'datetime')]

        if subdiv_minutes > 1:
            # repeat each records 3 times, except the start time which is not repeated
            rec = rec.iloc[np.arange(total_minutes).repeat(subdiv_minutes), :][:(1 - subdiv_minutes)]

            # for time stamp, interpolate
            td = np.tile([pd.Timedelta(_, 'sec') for _ in (np.arange(subdiv_minutes) * 60 / subdiv_minutes)],
                         total_minutes)[:(1 - subdiv_minutes)]
            rec['datetime'] += td

        # reverse the order
        rec = rec[-1::-1]

        return rec


# color scale on log
def mk_color(arr):
    amax = 10 ** np.ceil(np.log10(arr.max()))

    fac = (10 * arr.max() / amax)
    if fac < 1.2:
        fac = 1
        amax /= 10
    elif fac <2.4:
        fac=.2
    elif fac < 5.5:
        fac = .5
    else:
        fac = 1


    norm = LogNorm(vmax=(amax * fac), vmin=(amax / 1000 * fac), clip=True)
    if fac == 1:
        bdry = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]) / 1000 * amax
    elif fac == .5:
        bdry = np.array([.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]) / 1000 * amax
    elif fac == .2:
        bdry = np.array([.2, .5, 1, 2, 5, 10, 20, 50, 100, 200 ]) / 1000 * amax

    nc = len(bdry)
    viridis = plt.cm.viridis
    vv = [np.round(_ / nc * viridis.N) for _ in range(nc)]
    # vv = [ viridis(np.round(_/nc*viridis.N)) for _ in range(nc)]
    cm = ListedColormap([viridis(int(np.round(_ / nc * viridis.N))) for _ in range(nc)])

    bnorm = BoundaryNorm(bdry, len(bdry), clip=True)
    return norm, bnorm, bdry, cm


# plot with pylab
def mkplt_pylab_old(arr, fname, halfrng, ttle=None, contour=False, *args, **kwds):
    extent = [-halfrng, halfrng, -halfrng, halfrng]
    mkplt_pylab(arr, fname, extent, ttle, contour, *args, **kwds)
    ### plt.clf()
    ### if contour:
    ###     plt.contourf(arr[:, :], extent=[-halfrng, halfrng, -halfrng, halfrng], *args, **kwds)
    ### else:
    ###     plt.imshow(arr[-1::-1, :], extent=[-halfrng, halfrng, -halfrng, halfrng], *args, **kwds)
    ### plt.colorbar()
    ### plt.xlabel('(meters)')
    ### plt.ylabel('(meters)')
    ### if not ttle is None:
    ###     plt.title(ttle)

    ### plt.savefig(fname)

def mkplt_pylab(arr, fname, extent, ttle=None, contour=False, *args, **kwds):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if contour:
        plt.contourf(arr[:, :], extent=extent, *args, **kwds)
    else:
        plt.imshow(arr[-1::-1, :], extent=extent, *args, **kwds)
    plt.colorbar()
    plt.xlabel('(meters)')
    plt.ylabel('(meters)')
    #ax.set_aspect((extent[3] - extent[2]) / (extent[1] - extent[0]))
    ax.set_aspect(1)
    if not ttle is None:
        plt.title(ttle)

    plt.savefig(fname)


# plot with plotter
def mkplt_plotter(arr, fname, x, y, prj, start_time, ttle=None, *args, **kwds):

    arrx = np.expand_dims(arr, axis=0)
    contour_options = {**{'alpha': .5},
                       **{_: kwds[_] for _ in ('norm', 'levels', 'cmap') if _ in kwds}}
    if cartopy.__version__ >= '0.19':
        imgopts = [cimgt.GoogleTiles(style='satellite', cache=True)]
    else:
        imgopts = [cimgt.GoogleTiles(style='satellite')]
    p = psolo.Plotter(arrx, tstamps=[start_time], x=x, y=y, projection=prj,
                      plotter_options={
                          'contour_options': contour_options,
                          'background_manager': BackgroundManager(
                              add_image_options=imgopts,
                          ),
                      })
    p.savefig(fname)
def mkplt_plotter_old(arr, fname, halfrng, lnlt0, start_time, ttle=None, *args, **kwds):
    ncel = arr.shape[-1]
    coords1d = np.linspace(-halfrng, halfrng, ncel)

    ln0, lt0 = lnlt0

    prj = ccrs.LambertConformal(
        central_longitude=ln0, central_latitude=lt0,
        standard_parallels=(lt0, lt0), globe=ccrs.Globe())

    mkplt_plotter(arr, fname, coords1d, coords1d, prj, start_time, ttle, *args, **kwds)

    ### arrx = np.expand_dims(arr, axis=0)
    ### contour_options = {**{'alpha': .5},
    ###                    **{_: kwds[_] for _ in ('norm', 'levels', 'cmap') if _ in kwds}}
    ### p = psolo.Plotter(arrx, tstamps=[start_time], x=coords1d, y=coords1d, projection=prj,
    ###                   plotter_options={
    ###                       'contour_options': contour_options,
    ###                       'background_manager': BackgroundManager(
    ###                           add_image_options=[cimgt.GoogleTiles(style='satellite', cache=True)],
    ###                       ),
    ###                   })
    ### p.savefig(fname)

def set_xy(df_sites):
    df_use = df_sites.loc[df_sites.use > 0]
    lon = df_use.longitude.mean()
    lat = df_use.latitude.mean()
    print(lon, lat)
    prj = ccrs.LambertConformal(
        central_longitude=lon, central_latitude=lat,
        standard_parallels=(lat, lat), globe=ccrs.Globe())

    xyz =  prj.transform_points(ccrs.PlateCarree(),  df_sites.longitude.values, df_sites.latitude.values)
    df_sites['x'] = xyz[:, 0]
    df_sites['y'] = xyz[:, 1] 
    df_sites['distance'] = (df_sites['x'] ** 2 + df_sites['y'] ** 2).apply(np.sqrt)

    return df_sites, prj


def main(metfname, evtfname, sitefname, oroot, lnlt0=None, halfrng=None, ncel=201,
         subdiv_minutes=1, nbackward=1, summary_title=None, mass_balance=True):


    # event file should be cleand:  have datetime column, and only columns with device names.  entire length must be the time period to be simulated (excl. backward min)
    df_events = pd.read_csv(evtfname)
    df_events = (df_events
            .assign(datetime = pd.to_datetime(df_events.datetime))
            .sort_values('datetime', ascending=False)
            .set_index('datetime')
            )

    # modeling period
    #dtm0 = df_events.index.min() - pd.Timedelta(nbackward / subdiv_minutes , 'min')
    dtm0 = df_events.index.min() - pd.Timedelta(nbackward - 1 , 'min')
    dtm1 = df_events.index.max()
    print(df_events.index.min())
    print(df_events.index.max())
    print(dtm0, dtm1)

    # site information
    df_sites = (
            pd.read_csv(sitefname)
            .set_index('device')
            .assign(use = lambda x: x.index.isin(df_events.columns)) # flag sensort to be modeled
            )
    df_sites, prj = set_xy(df_sites) # pick projection origin, calcuate x,y coords and distance from origin



    # relevant met data are subsetted here
    df_met = pd.read_csv(metfname)
    try:
        df_met['datetime'] = pd.to_datetime(df_met.timestamp, utc=True)
    except AttributeError:
        df_met['datetime'] = pd.to_datetime(df_met.timestamp_min, utc=True)
    df_met = df_met.loc[:, ['datetime', 'device', 'wdir', 'wd_std', 'wspd']]

    # TODO match timestamp if they arent
    df_met['datetime'] = df_met['datetime'].dt.round('min')

    # subset by time and location
    df_met = df_met[(df_met['datetime'] >= dtm0) & (df_met['datetime'] <= dtm1)]
    df_met = df_met.merge(df_sites.loc[:,['x', 'y', 'distance']], on='device', how='left')
    #print(df_met)
    df_met = (df_met
            .loc[df_met.distance == df_met.distance.min(), :] 
            .sort_values('datetime', ascending=False)
            .set_index(['datetime', 'device'])
            )
    #print(df_met)
    df_met = df_met.loc[~np.isnan(df_met.x), :]

    df_sites = (df_sites
            .loc[df_sites.use, :] 
            .drop(columns=['use'])
            )

    heatmap = hm.Heatmap(df_met, df_events, df_sites, nbackward=nbackward, mass_balance=mass_balance)

    arr = heatmap.arr.sum(axis=0)
    norm, bnorm, bdry, cm = mk_color(arr)
    print(bdry[-1], arr.max())

    extent = [heatmap.xcoords[0], heatmap.xcoords[-1], heatmap.ycoords[0], heatmap.ycoords[-1],]
    aspect = ( extent[3]-extent[2]) / ( extent[1] - extent[0])
    print(extent, aspect)

#    tempdir = tempfile.TemporaryDirectory()
#    wdir = Path(tempdir.name)
    wdir = Path('.')
    fpsopt = '-r 2'
    norm_, bnorm_, bdry_, cm_ = mk_color(arr / len(heatmap.df_events.index))
    print(bdry_[-1], (arr/len(heatmap.df_events.index)).max())
    for i, dtm in enumerate(heatmap.df_events.index):
        fname = f'{oroot}_fig{i:02d}.png'
        #fname = f'{i:02d}.png'
        met = heatmap.df_met.loc[dtm]
        ttle = f'{dtm}\nws {met["wspd"].values[0]:4.1f} m/sec, wd {met["wdir"].values[0]:3.0f} deg, sd_wd {met["wd_std"].values[0]:4.1f} deg'
        #a = heatmap.combine_by_time(dtm)
        a = heatmap.arr[i]
        print(a.max(), bdry_[-1])
        #mkplt_pylab(np.maximum(a, 0.001), wdir / fname, extent=extent, ttle=ttle, contour=True, norm=bnorm_, levels=bdry_, cmap=cm_) 
        mkplt_pylab(a, wdir / fname, extent=extent, ttle=ttle, contour=True, norm=bnorm_, levels=bdry_, cmap=cm_) 
    

    fname_sh = f'{oroot}_fig%02d.png'
    #fname_sh = f'%02d.png'
    oname = f'{oroot}_contour.mp4'

    png_w = mpl.pyplot.rcParams['figure.figsize'][0] * mpl.pyplot.rcParams['figure.dpi']
    print(png_w)
    #adjust_width = f'-vf scale={png_w}:-2'
    adjust_width = ''
    #adjust_width = f'-vf scale=640:-2'

    cmd = f'ffmpeg {fpsopt} -i "{Path(wdir) / fname_sh }" {adjust_width} -vframes {len(heatmap.df_events.index)} -crf 3 -vcodec libx264 -pix_fmt yuv420p -f mp4 -y  "{oname}"'
    print(cmd)
    try:
        subprocess.run(shlex.split(cmd), check=True)
    except subprocess.CalledProcessError:
        fname_sh2 = f'{oroot}_fig??.png'
        oname2 = f'{oroot}_contour.gif'
        cmd2 = f'convert -delay 100 "{Path(wdir) / fname_sh2 }" "{oname}"'
        subprocess.run(shlex.split(cmd2), check=False)
    

    #mkplt_pylab(np.maximum(arr, 0.001), f'{oroot}_contour.png', extent=extent, ttle=summary_title, contour=True, norm=bnorm, levels=bdry, cmap=cm)
    mkplt_pylab(arr, f'{oroot}_contour.png', extent=extent, ttle=summary_title, contour=True, norm=bnorm, levels=bdry, cmap=cm)
    mkplt_plotter(arr, f'{oroot}_w_bg.png', heatmap.xcoords, heatmap.ycoords, prj, start_time=heatmap.datetimes[0], norm=bnorm, levels=bdry, cmap=cm)


    return heatmap


def main_old(metfname, oroot, start_time, total_minutes, lnlt0=None, halfrng=None, ncel=201,
         subdiv_minutes=1, nbackward=1, summary_title=None, mass_balance=True):
    met_reader = MetReader(metfname)

    x0, y0 = 0, 0

    rec = met_reader.read(start_time, total_minutes, subdiv_minutes)

    print(rec['wdir'])

    wd_mean, ws_mean = metutils.average_wdws(rec['wdir'], rec['wspd'])
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
    r, theta, sd, dte = [_.array for _ in (r, theta, sd, rec['datetime'])]

    arrays, obj = hm.superpose_trajectories(theta, sd, r,
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
            mkplt_plotter_old(arr, f'{oroot}_w_bg.png', halfrng, lnlt0, start_time, norm=bnorm, levels=bdry, cmap=cm)


    else:
        mkplt_pylab(arr, f'{oroot}_tile.png', halfrng, summary_title, contour=False)

        mkplt_pylab(arr, f'{oroot}_contour.png', halfrng, summary_title, contour=True)


if __name__ == '__main__':
    pass

