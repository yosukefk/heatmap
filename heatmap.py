
import metutils
#from pdf import CalcPdf
import pdf

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
        self.datetimes = df_met.index

        self.ncel = ncel
        self.halfrng = halfrng
        self.nbackward = nbackward
        self.subdiv_minutes = subdiv_minutes
        self.mass_balance = mass_balance

        self.prepare()

        self.process_all_tsteps()

    def prepare(self):

        assert len(self.df_met.index) - len(self.df_events.index) + 1 == self.nbackward

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
            halfrng = np.around(ws_mean * self.nbackward / self.subdiv_minutes * 60, decimals=-2)
            halfrng = max(halfrng , 100)
#            import pdb; pdb.set_trace()
            #rmax = self.df_met.wspd.max()
            #halfrng = rmax * self.nbackward * 60
            #self.halfrng = np.ceil(halfrng / 100) * 100
            #import pdb; set_trace()
            self.halfrng = halfrng
        if self.ncel is None:
            self.ncel = 201
        self.res = (self.halfrng * 2) / (self.ncel - 1)

        # prepare canvas
        print(self.df_sites)
        print(self.df_sites.x.min(), self.df_sites.x.max(), self.df_sites.y.min(), self.df_sites.y.max())
        xmin = self.df_sites.x.min() - self.halfrng
        xmax = self.df_sites.x.max() + self.halfrng
        ymin = self.df_sites.y.min() - self.halfrng
        ymax = self.df_sites.y.max() + self.halfrng
        print(xmin, xmax, ymin, ymax)

        nx = np.ceil( (xmax - xmin) / self.res + 1 ).astype(int) 
        ny = np.ceil( (ymax - ymin) / self.res + 1 ).astype(int) 
        print(xmax-xmin, ymax-ymin, self.res, nx, ny)

        self.canvas = np.zeros((ny, nx))
        self.xcoords = np.linspace(xmin, xmax, nx)
        self.ycoords = np.linspace(ymin, ymax, ny)

        self.df_sites = (self.df_sites
                .assign( 
                    i = lambda x: np.round(x['x'] / self.res ).astype(int), 
                    j = lambda x: np.round(x['y'] / self.res ).astype(int),
                    ioff = lambda x: np.round((x['x'] - (self.xcoords[0] + self.halfrng)) / self.res ).astype(int), 
                    joff = lambda x: np.round((x['y'] - (self.ycoords[0] + self.halfrng)) / self.res ).astype(int),
                    ))

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
        if evt[dtm] <= 0:
            return canvas

        ioff, joff = [self.df_sites.loc[device][_].astype(int) for _ in ('ioff', 'joff')]

        if canvas is None:
            canvas = np.zeros_like(self.canvas)

        for stuff in self.arrays[dtm]:
            canvas[joff:(joff+self.ncel),ioff:(ioff+self.ncel)] += stuff['pdf']
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

        self.calculator = pdf.CalcPdf(met.theta, met.sd, met.r, met.speed, ncel=self.ncel, halfrng=self.halfrng, mass_balance=self.mass_balance)

        # grow old trajectories

        for ib, dtmf in enumerate(dtm_forward):
            lst = self.arrays[dtmf]
            print(dtm, ib, dtmf, len(lst), self.nbackward - ib - 1)
            #assert len(lst) == self.nbackward - ib - 1

            pp, dd = self.calculator.calc_from_dens(lst[-1]['edge'])
            lst.append({'datetime': dtm, 'pdf': pp, 'edge': dd} )

        # new trajectories
        pp, dd = self.calculator.calc_from_pnt(0, 0)
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
