
import metutils
#from pdf import CalcPdf
import pdf

#from scipy.interpolate import RBFInterpolator
from invdisttree import Invdisttree
from numpy import typing as npt
import numpy as np

import pandas as pd

from importlib import reload
reload(pdf)


class Heatmap:
    def __init__(self, df_met, df_events, df_sites, nbackward=1, subdiv_minutes=1,
            ncel=None, halfrng=None, mass_balance=True, return_list=True):
#    def superpose_trajectories(theta: npt.NDArray, sd: npt.NDArray, rmax: npt.NDArray, nbackward: int = 1, ncel: int = None,
#                               halfrng: float = None, mass_balance: bool = None,
#                               return_list: bool = False,
#                               return_obj: bool = False) -> object:


        self.df_events = df_events
        self.df_sites = df_sites
        self.df_met = df_met

        self.sites = df_events.columns
        #self.datetimes = df_events.index.values
        self.datetimes = df_met.index.unique(level='datetime')

        self.ncel = ncel
        self.halfrng = halfrng
        self.nbackward = nbackward
        self.subdiv_minutes = subdiv_minutes
        self.mass_balance = mass_balance

        self.prepare()

        self.process_all_tsteps()

    def prepare(self):

        print('nmet:', len(self.df_met.index.unique(level='datetime')))
        print('nevt:', len(self.df_events.index))
        print('nbkw:', self.nbackward)


        assert len(self.df_met.index.unique(level='datetime')) - len(self.df_events.index) + 1 == self.nbackward

        # prepare met (use radian, cartesian coordinates)
        self.df_met = (self.df_met
                .assign(
                    # stride
                    r = lambda x: x['wspd'] * 60. / self.subdiv_minutes, 
                    # wind vector on cartesian coords, but revised sign 
                    theta = lambda x: np.pi / 180. * ((90. - x['wdir']) % 360.),
                    # change unit
                    sd = lambda x: np.pi * x['wd_std'] / 180.,
                    # speed needed for mass balance
                    speed = lambda x: x['wspd'],
                    )
                )

        if self.subdiv_minutes > 1:
            raise NotImplemented('will think about this feature')
            # TODO, need to update this... what shall  i do now, ?  lookup with tstamp, or stick to linear index?
            # repeat each records subdiv times, except the start time which is not repeated
            rec = rec.iloc[np.arange(total_minutes).repeat(subdiv_minutes), :][:(1 - subdiv_minutes)]

            # for time stamp, interpolate
            td = np.tile([pd.Timedelta(_, 'sec') for _ in (np.arange(subdiv_minutes) * 60 / subdiv_minutes)],
                         total_minutes)[:(1 - subdiv_minutes)]
            rec['datetime'] += td




        # layout of calculators
        #  halfrng = max distance traveled in unit time (1 min, typically)
        #  each calculator covers square of 2*halfng+1 cells
        #  but, it calculates density/front for 4*halfrng+1 cells, since they are expected to spill out
        #  when multiple minute trajectories are calculated, those outer cells (2*halfrng+1 witdth, halfrng depth to sides, and halfrngxhalfrng square at corner)
        #  are passed to neiboring calculators for subsequent calculations

        # prepare calculators
        if self.halfrng is None:
            #wd_mean, ws_mean = metutils.average_wdws(self.df_met['wdir'], self.df_met['wspd'])
            ws_max = self.df_met['wspd'].max()
            halfrng = np.around(ws_max / self.subdiv_minutes * 60, decimals=-1) # nearest 10m
#            halfrng = max(halfrng , 100) # at least 100m
#            import pdb; pdb.set_trace()
            #rmax = self.df_met.wspd.max()
            #halfrng = rmax * self.nbackward * 60
            #self.halfrng = np.ceil(halfrng / 100) * 100
            #import pdb; set_trace()
            self.halfrng = halfrng

        # make cell size to be about 1m
        if self.ncel is None:
            if halfrng >= 100:
                self.ncel = 201
            elif halfrng >= 40:
                self.ncel = int(halfrng) * 2 + 1
            else:
                self.ncel = 81

        if self.ncel // 2 == 0:
            self.ncel += 1
            warnings.warn(f'ncel changed to odd number: {self.ncel}')

        halfncel = (self.ncel - 1) // 2
        self.halfncel = halfncel
        self.extended_ncel = self.ncel + 2*halfncel  # size of calculator including buffer (see desc below)

        self.res = (self.halfrng * 2) / (self.ncel - 1)

        # prepare canvas
        print(self.df_sites)
        print(self.df_sites.x.min(), self.df_sites.x.max(), self.df_sites.y.min(), self.df_sites.y.max())
        xmin0 = self.df_sites.x.min() - self.halfrng
        xmax0 = self.df_sites.x.max() + self.halfrng
        ymin0 = self.df_sites.y.min() - self.halfrng
        ymax0 = self.df_sites.y.max() + self.halfrng
        print(xmin0, xmax0, ymin0, ymax0)

        # i have to fill the canvas with tiles of calculators
        # each calculator has 2*hc+1 cells on each side (or, 2*hr+res meters)
        #
        # entire length coverred by m calculators (alined in one direction) is
        # (2*hc+1)*m cells (2*hr+res)*m meters
        #
        # the canvas side has to be hr meters longer on both end (hc cells)
        # since calculator actually calculates region that spilled out from its teriitory
        # (2*hc+1)*m * 2hc cells, or (2*hr+res)*m + 2*hc meters
        #
        # anyways, in order to fit all the sensors within the calculation, they have to fit 
        # inside half of the last calculators in a row (otherwise it plumes immediately spills
        # out of box when wind direction is wrong)
        # so , m (number of calculator in 1d) has to satisfy following condition
        # 
        #  (2*hr+res)*m - 2*hr > xmax-xmin

        # count of calculators on each side
        mx = np.ceil(((xmax0-xmin0) + 2*halfrng)/(2*halfrng+self.res)).astype(int)
        my = np.ceil(((ymax0-ymin0) + 2*halfrng)/(2*halfrng+self.res)).astype(int)

        # count of cells on each side (including spilled area from outermost calculators)
        nx = (2*halfncel+1) * mx + 2*halfncel
        ny = (2*halfncel+1) * my + 2*halfncel

        # x/y range on each side (including spilled area from outermost calculators)
        xrng,yrng = nx*self.res, ny*self.res
        xmin = .5+(xmin0+xmax0) - .5*xrng
        xmax = .5+(xmin0+xmax0) + .5*xrng
        ymin = .5+(ymin0+ymax0) - .5*yrng
        ymax = .5+(ymin0+ymax0) + .5*yrng



        print(xmax0-xmin0, ymax0-ymin0, xmax-xmin, ymax-ymin, self.res, nx, ny, mx, my)


        # canvas
        self.canvas = np.zeros((ny, nx))

        # x/y coord of cell center
        self.xcoords = np.linspace(xmin, xmax, nx)
        self.ycoords = np.linspace(ymin, ymax, ny)
        self.extent = [self.xcoords[0], self.xcoords[-1], self.ycoords[0], self.ycoords[-1]]

        # calculators
        self.calculators = [None] * ( mx * my)

        self.df_calculator = (pd.DataFrame( 
            dict(
            # serial number of calculator
            cdx = np.arange(mx*my), 
            # 2d layout of calculators
            idx = np.tile(range(mx), my),
            jdx = np.repeat(range(my), mx),
            # index of centroid of each calculator
            i0 = np.tile([2*halfncel + (_)*(2*halfncel+1) for _ in range(mx)], my), 
            j0 = np.repeat([2*halfncel + (_)*(2*halfncel+1) for _ in range(my)], mx), 
            # xy coordinae of cell centroid
            x0 = np.tile([self.xcoords[2*halfncel + (_)*(2*halfncel+1)] for _ in range(mx)], my), 
            y0 = np.repeat([self.ycoords[2*halfncel + (_)*(2*halfncel+1)] for _ in range(my)], mx),
            # index of llcorner of each calculator
            illc = np.tile([_*(2*halfncel+1) for _ in range(mx)], my).astype(int),
            jllc = np.repeat([_*(2*halfncel+1) for _ in range(my)], mx).astype(int),
            )
            )
            .set_index('cdx')
            )
        # pointer to neiboring claculators
        self.df_calculator['lo'] = np.where(self.df_calculator.idx == 0, np.nan, (self.df_calculator.index - 1))
        self.df_calculator['ro'] = np.where(self.df_calculator.idx == (mx-1), np.nan, (self.df_calculator.index + 1))
        self.df_calculator['od'] = np.where(self.df_calculator.jdx == 0, np.nan, (self.df_calculator.index - mx))
        self.df_calculator['ou'] = np.where(self.df_calculator.jdx == (my-1), np.nan, (self.df_calculator.index + mx))
        self.df_calculator['ld'] = np.where(self.df_calculator.idx == 0, np.nan, self.df_calculator.od - 1)
        self.df_calculator['rd'] = np.where(self.df_calculator.idx == (mx-1), np.nan, self.df_calculator.od + 1)
        self.df_calculator['lu'] = np.where(self.df_calculator.idx == 0, np.nan, self.df_calculator.ou - 1)
        self.df_calculator['ru'] = np.where(self.df_calculator.idx == (mx-1), np.nan, self.df_calculator.ou + 1)
        self.df_calculator['oo'] = self.df_calculator.index

        
        # for convenience of met field inter polation, get these cords as array
        self.calculator_origin_coords = np.vstack([self.df_calculator.y0, self.df_calculator.x0]).T


        self.df_sites = (self.df_sites
                .assign( 
                    # ijdx of emitter
                    #i = lambda x: np.round(x['x'] / self.res ).astype(int), 
                    #j = lambda x: np.round(x['y'] / self.res ).astype(int),
                    i = lambda p: np.round((p['x'] - self.xcoords[0]) / self.res ).astype(int), 
                    j = lambda p: np.round((p['y'] - self.ycoords[0]) / self.res ).astype(int),

                    # obsolete
                    ioff_obsolete = lambda x: np.round((x['x'] - (self.xcoords[0] + self.halfrng)) / self.res ).astype(int), 
                    joff_obsolete = lambda x: np.round((x['y'] - (self.ycoords[0] + self.halfrng)) / self.res ).astype(int),
                    )
                .assign(
                    # which calculator to start with
                    cdx = lambda p: ((p.j - halfncel) // (2*halfncel+1)) * mx + ((p.i - halfncel) // (2*halfncel+1))
                    )
                )
        # position of device relative to centor of the calculator
        self.df_sites['ioff'] = self.df_sites['i'] - self.df_calculator.loc[self.df_sites.cdx, 'i0'].values
        self.df_sites['joff'] = self.df_sites['j'] - self.df_calculator.loc[self.df_sites.cdx, 'j0'].values
        self.df_sites['xoff'] = self.df_sites['x'] - self.df_calculator.loc[self.df_sites.cdx, 'x0'].values
        self.df_sites['yoff'] = self.df_sites['y'] - self.df_calculator.loc[self.df_sites.cdx, 'y0'].values


        # this dict holds fronteers at each tstep, each age (for nbackward > 1)
        self.fronts = {}

        # 2d output at each tstep
        self.outputs = {}

        # 3d ouptut (t, y, x)
        self.arr = None





        
    def process_all_tsteps(self):
        for i, dtm in enumerate(self.datetimes):
            #dtm_backward = self.datetimes[(i+1):(i+self.nbackward)]
            dtm_forward = self.datetimes[max(i-self.nbackward + 1, 0) : i]#[-1::-1]
            print(i, dtm, dtm_forward)

            ### # purge too far back tranectories
            ### to_purge = [_ for _ in self.fronts.keys() if _ > dtm_forward[-1]]
            ### for dtmf in to_purge:
            ###     del self.fronts[dtmf]

            self.process_one_tstep(dtm, dtm_forward)



        self.arr = np.stack([self.outputs[_] for _ in self.datetimes], axis=0)




    def process_one_tstep(self, dtm, dtm_forward):
        # get list of sites with flag
        # make new pdf with updated wind

        met = self.df_met.loc[dtm]

        # TODO i am getting df of multiple sensors, need to aggregate here...?
        # prepare my * mx calculators, with interpolaated wind
        
        if (len(met.index) < 2):
            # too met station for interpolation
            theta = np.repeat(met.theta.values, len(self.df_calculator.index))
            sd = np.repeat(met.sd.values, len(self.df_calculator.index))
            r = np.repeat(met.r.values, len(self.df_calculator.index))
            speed = np.repeat(met.speed.values, len(self.df_calculator.index))
        else:
            theta = Invdisttree(np.vstack([met.y, met.x]).T, met.theta  )(self.calculator_origin_coords)
            sd =    Invdisttree(np.vstack([met.y, met.x]).T, met.sd     )(self.calculator_origin_coords)
            r =     Invdisttree(np.vstack([met.y, met.x]).T, met.r      )(self.calculator_origin_coords)
            speed = Invdisttree(np.vstack([met.y, met.x]).T, met.speed  )(self.calculator_origin_coords)

        
        self.calculators = [pdf.CalcPdf(_t, _d, _r, _s, ncel=self.extended_ncel, halfrng=2*self.halfrng, mass_balance=self.mass_balance) 
                for (_t, _d, _r, _s) in zip(theta, sd, r, speed)]

        canvas = np.zeros_like(self.canvas, dtype=np.float16)
        for device in self.df_sites.index:
            canvas = self.process_one_tstep_one_device(device, dtm, dtm_forward, canvas)

        self.outputs[dtm] = canvas

    
    def process_one_tstep_one_device(self, device, dtm, dtm_forward, canvas):

        mysite = self.df_sites.loc[device]

        #import pdb; pdb.set_trace()

        # task 1: process front from earlier calculations
        for ib, dtmf in enumerate(dtm_forward):
            try:
                my_fronts = self.fronts[dtmf][device]
            except KeyError:
                continue 

            next_front = {}
            for mycdx, front in my_fronts.items():
                # grab one calculator
                mycalculator = self.calculators[mycdx]
                mycmeta = self.df_calculator.loc[mycdx]
                illc,jllc = [int(mycmeta[_]) for _ in ('illc', 'jllc')]

                dens, front = self.calculators[mycdx].calc_from_dens(front)

                self.process_one_tstep_one_device_worker(mycdx, dens, front, canvas, next_front )

            self.fronts[dtmf][device] = next_front



        # task 2: process new event

        #mycdx = self.df_sites.loc[device, 'cdx']
        mycdx = mysite.loc['cdx'].astype(int)
        mycalculator = self.calculators[mycdx]
        mycmeta = self.df_calculator.loc[mycdx]
        illc,jllc = [int(mycmeta[_]) for _ in ('illc', 'jllc')]
            
        evt = self.df_events[device]
        if np.isnan(evt.get(dtm, np.nan)) or evt[dtm] <= 0:
            return canvas
        xoff, yoff = [self.df_sites.loc[device, _] for _ in ['xoff', 'yoff']]
        dens, front = mycalculator.calc_from_pnt(xoff, yoff)

        assert dtm not in self.fronts or device not in self.fronts[dtm]
        next_front = {}

        self.process_one_tstep_one_device_worker(mycdx, dens, front, canvas, next_front)
        self.fronts.setdefault(dtm, {})[device] = next_front


        ### # add calculated pdf to canvas
        ### canvas[jllc:(jllc+self.extended_ncel), illc:(illc+self.extended_ncel), ] += dens


        ### # partition the frontieer across calculators

        ### # source and target index (on 1d) of the frontier
        ### sbdx = [None,               self.halfncel,          3*self.halfncel + 1,    ]
        ### sedx = [self.halfncel,      3*self.halfncel + 1,    None,                   ]
        ### tbdx = [2*self.halfncel+1,  self.halfncel,          self.halfncel,          ]
        ### tedx = [3*self.halfncel+1,  3*self.halfncel + 1,    2*self.halfncel,        ]

        ### # make those in to slice
        ### sslc = [slice(b,e) for b,e in zip(sbdx, sedx)]
        ### tslc = [slice(b,e) for b,e in zip(tbdx, tedx)]


        ### # create fronteer on each of 9 calculators (8 neigbors and 1 for itself)
        ### for jj in range(3):

        ###     # source/targe indices on y-axis
        ###     sjslc = sslc[jj]
        ###     tjslc = tslc[jj]

        ###     # (part of) name of neigbor
        ###     jneighbor = ['d', 'o', 'u'][jj]

        ###     for ii in range(3):

        ###         # source/targe indices on x-axis
        ###         sislc = sslc[ii]
        ###         tislc = tslc[ii]

        ###         if front[sjslc, sislc].count() == 0:
        ###             continue

        ###         # name of neigbor
        ###         neighbor = ['l', 'o', 'r'][ii] + jneighbor

        ###         # calculator index of the neigbor
        ###         tcdx = mycmeta[neighbor]

        ###         # frontier at the target calculator
        ###         tgt = self.fronts.setdefault(dtm, {}).setdefault(device, {}).setdefault(tcdx, mycalculator.masked_zeros.copy())

        ###         # transfer the frontire to the neigbor
        ###         tgt[tjslc, tislc] = front[sjslc, sislc]

        return canvas



    def process_one_tstep_one_device_worker(self, cdx, dens, front, canvas, next_front ):
        mycalculator = self.calculators[cdx]
        mycmeta = self.df_calculator.loc[cdx]
        illc,jllc = [int(mycmeta[_]) for _ in ('illc', 'jllc')]

        # add calculated pdf to canvas
        canvas[jllc:(jllc+self.extended_ncel), illc:(illc+self.extended_ncel), ] += dens


        # partition the frontieer across calculators

        # source and target index (on 1d) of the frontier
        sbdx = [None,               self.halfncel,          3*self.halfncel + 1,    ]
        sedx = [self.halfncel,      3*self.halfncel + 1,    None,                   ]
        tbdx = [2*self.halfncel+1,  self.halfncel,          self.halfncel,          ]
        tedx = [3*self.halfncel+1,  3*self.halfncel + 1,    2*self.halfncel,        ]

        # make those in to slice
        sslc = [slice(b,e) for b,e in zip(sbdx, sedx)]
        tslc = [slice(b,e) for b,e in zip(tbdx, tedx)]


        # create fronteer on each of 9 calculators (8 neigbors and 1 for itself)
        for jj in range(3):

            # source/targe indices on y-axis
            sjslc = sslc[jj]
            tjslc = tslc[jj]

            # (part of) name of neigbor
            jneighbor = ['d', 'o', 'u'][jj]

            for ii in range(3):

                # source/targe indices on x-axis
                sislc = sslc[ii]
                tislc = tslc[ii]

                if front[sjslc, sislc].count() == 0:
                    continue

                # name of neigbor
                neighbor = ['l', 'o', 'r'][ii] + jneighbor

                # calculator index of the neigbor
                tcdx = mycmeta[neighbor]

                if np.isnan(tcdx): continue

                # frontier at the target calculator
                #tgt = self.fronts.setdefault(dtm, {}).setdefault(device, {}).setdefault(tcdx, mycalculator.masked_zeros.copy())
                tgt = next_front.setdefault(int(tcdx), mycalculator.masked_zeros.copy())


                # transfer the frontire to the neigbor
                tgt[tjslc, tislc] = np.ma.masked_array(
                        (tgt[tjslc, tislc].filled(0) + front[sjslc, sislc].filled(0)), 
                        (tgt[tjslc, tislc].mask & front[sjslc, sislc].mask))



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

    calculator = pdf.CalcPdf(theta[0], sd[0], rmax[0], ncel=ncel, halfrng=halfrng, mass_balance=mass_balance)

    if return_list:
        cum_pp = []
    else:
        cum_pp = calculator.zeros.copy()

    dd = None
    for th, s, r in zip(theta, sd, rmax):
        if any((th != calculator.theta0, s != calculator.sd, r != calculator.rmax)):
            calculator = pdf.CalcPdf(th, s, r, ncel=ncel, halfrng=halfrng, mass_balance=mass_balance)

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
