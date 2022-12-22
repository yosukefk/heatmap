
import metutils
#from pdf import CalcPdf
import pdf

#from scipy.interpolate import RBFInterpolator
from invdisttree import Invdisttree
from numpy import typing as npt
import numpy as np

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




        # prepare calculators
        if self.halfrng is None:
            wd_mean, ws_mean = metutils.average_wdws(self.df_met['wdir'], self.df_met['wspd'])
            halfrng = np.around(ws_mean * self.nbackward / self.subdiv_minutes * 60, decimals=-1) # nearest 10m
            halfrng = max(halfrng , 100) # at least 100m
#            import pdb; pdb.set_trace()
            #rmax = self.df_met.wspd.max()
            #halfrng = rmax * self.nbackward * 60
            #self.halfrng = np.ceil(halfrng / 100) * 100
            #import pdb; set_trace()
            self.halfrng = halfrng

        if self.ncel is None:
            self.ncel = 201

        if self.ncel // 2 == 0:
            self.ncel += 1
            warnings.warn(f'ncel changed to odd number: {self.ncel}')

        halfncel = (self.ncel - 1) // 2

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
        # interval between two calcculators = 101
        # total width becomes 101*m+100  where m is number of calculators
        # 

        mx = np.ceil(((xmax0-xmin0) - halfrng)/(halfrng+1)).astype(int)
        my = np.ceil(((ymax0-ymin0) - halfrng)/(halfrng+1)).astype(int)

        nx = (halfncel+1) * mx + halfncel
        ny = (halfncel+1) * my + halfncel

        xrng,yrng = nx*self.res, ny*self.res
        xmin = .5+(xmin0+xmax0) - .5*xrng
        xmax = .5+(xmin0+xmax0) + .5*xrng
        ymin = .5+(ymin0+ymax0) - .5*yrng
        ymax = .5+(ymin0+ymax0) + .5*yrng



        print(xmax0-xmin0, ymax0-ymin0, xmax-xmin, ymax-ymin, self.res, nx, ny, mx, my)


        self.canvas = np.zeros((ny, nx))
        self.xcoords = np.linspace(xmin, xmax, nx)
        self.ycoords = np.linspace(ymin, ymax, ny)
        self.extent = [self.xcoords[0], self.xcoords[-1], self.ycoords[0], self.ycoords[-1]]

        # calculators
        self.calculators = [[None] * mx] * my
        calculator_i0 = np.tile([halfncel + (_)*(halfncel+1) for _ in range(mx)], my)
        calculator_j0 = np.tile([halfncel + (_)*(halfncel+1) for _ in range(my)], mx)
        calculator_x0 = np.tile([self.xcoords[halfncel + (_)*(halfncel+1)] for _ in range(mx)], my)
        calculator_y0 = np.repeat([self.ycoords[halfncel + (_)*(halfncel+1)] for _ in range(my)], mx)
        self.calculator_origin_ijdx = np.vstack([calculator_j0, calculator_i0]).T
        self.calculator_origin_coords = np.vstack([calculator_y0, calculator_x0]).T

        self.df_sites = (self.df_sites
                .assign( 
                    # ijdx of emitter
                    #i = lambda x: np.round(x['x'] / self.res ).astype(int), 
                    #j = lambda x: np.round(x['y'] / self.res ).astype(int),
                    i = lambda x: np.round((x['x'] - self.xcoords[0]) / self.res ).astype(int), 
                    j = lambda x: np.round((x['y'] - self.ycoords[0]) / self.res ).astype(int),

                    # obsolete
                    ioff_obsolete = lambda x: np.round((x['x'] - (self.xcoords[0] + self.halfrng)) / self.res ).astype(int), 
                    joff_obsolete = lambda x: np.round((x['y'] - (self.ycoords[0] + self.halfrng)) / self.res ).astype(int),
                    )
                .assign(
                    # which calculator to start with
                    cdx = lambda p: ((p.j - halfncel) // (halfncel+1)) * mx + ((p.i - halfncel) // (halfncel+1))
                    )
                .assign(
                    ioff = lambda p: p.i - self.calculator_origin_ijdx[p.cdx, 1],
                    joff = lambda p: p.j - self.calculator_origin_ijdx[p.cdx, 0],
                    )
                )

        # this dict holds pdf at each tstep, each age (for nbackward > 1)
        self.arrays = {}




    def combine_all(self, canvas=None):
        for dtm in self.df_events.index:
            canvas = self.combine_by_time(dtm, canvas)
        return canvas



    def combine_by_time(self, dtm, canvas=None):
        for device in self.df_sites.index:
            canvas = self.combine_by_time_device(device, dtm, canvas)
        return canvas


    def combine_by_time_device(self, device, dtm, canvas=None):

        evt = self.df_events[device]
        if np.isnan(evt[dtm]) or evt[dtm] <= 0:
            return canvas
        mycdx = self.df_sites.loc[device, 'cdx']

        ioff, joff = [self.df_sites.loc[device][_].astype(int) for _ in ('ioff_obsolete', 'joff_obsolete')]

        if canvas is None:
            canvas = np.zeros_like(self.canvas)

        for stuff in self.arrays[dtm]:
            mypdf = stuff['pdf'][mycdx]
            
            canvas[joff:(joff+self.ncel),ioff:(ioff+self.ncel)] += mypdf * evt[dtm]
        return canvas

        
    def process_all_tsteps(self):
        for i, dtm in enumerate(self.datetimes):
            #dtm_backward = self.datetimes[(i+1):(i+self.nbackward)]
            dtm_forward = self.datetimes[max(i-self.nbackward + 1, 0) : i]#[-1::-1]
            print(i, dtm, dtm_forward)

            self.process_one_tstep(dtm, dtm_forward)



    def process_one_tstep(self, dtm, dtm_forward):
        # get list of sites with flag
        # make new pdf with updated wind

        met = self.df_met.loc[dtm]

        # TODO i am getting df of multiple sensors, need to aggregate here...?
        # prepare my * mx calculators, with interpolaated wind
        theta = Invdisttree(np.vstack([met.y, met.x]).T, met.theta)(self.calculator_origin_coords)
        sd = Invdisttree(np.vstack([met.y, met.x]).T, met.theta)(self.calculator_origin_coords)
        r = Invdisttree(np.vstack([met.y, met.x]).T, met.theta)(self.calculator_origin_coords)
        speed = Invdisttree(np.vstack([met.y, met.x]).T, met.theta)(self.calculator_origin_coords)

        
        self.calculators = [pdf.CalcPdf(_t, _d, _r, _s, ncel=self.ncel, halfrng=self.halfrng, mass_balance=self.mass_balance) 
                for (_t, _d, _r, _s) in zip(theta, sd, r, speed)]

        #import pdb; pdb.set_trace()
        # grow old trajectories

        for ib, dtmf in enumerate(dtm_forward):
            lst = self.arrays[dtmf]
            print(dtm, ib, dtmf, len(lst), self.nbackward - ib - 1)
            #assert len(lst) == self.nbackward - ib - 1

            warings.warn('problematic! i cannot do this way, since event may not occur at center of calculator.  so it may be off in different part of arc')
            pp, dd = self.calculator.calc_from_dens(lst[-1]['edge'])
            lst.append({'datetime': dtm, 'pdf': pp, 'edge': dd} )

        # new trajectories
        #pp, dd = self.calculator.calc_from_pnt(0, 0)
        pp , dd = [], []
        for calc in self.calculators:
            ppp,ddd=calc.calc_from_pnt(0,0)
            pp.append(ppp)
            dd.append(ddd)
        self.arrays[dtm] = [{'datetime': dtm, 'pdf':pp, 'edge': dd}]



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
