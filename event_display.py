# Coding: utf-8

_all_ = [ 'event_display' ]

import os; from os import path as op
from tqdm import tqdm
import argparse
import awkward as ak
import glob
import numpy as np
import uproot as up
import pandas as pd

from bokeh import models as bm
from bokeh.layouts import layout
from bokeh.palettes import Colorblind as palette
palette = palette[8]
from bokeh import plotting as bp

class DatumManip():
    def __init__(self, datum, tracksters=False):
        self.datum = datum
        avars = ('x', 'y', 'z', 'R', 'L', 'e', 'c')
        self.ntracksters = len(self.datum['e'])

        for avar in avars:
            if avar not in self.datum:
                continue
            
            # self.x, self.y, ...
            setattr(self, avar, self.flatten(avar))
            # self.x_un, self.y_un, ...
            setattr(self, avar + '_un', np.unique(getattr(self, avar)))
            # self.x_trk, self.y_trk, ...
            if tracksters:
                st = self.sum_tracksters(avar)
                # self.x_trk, self.y_trk, ...
                setattr(self, avar + '_trk', st[0])
                # self.x_range, self.y_range, ...
                setattr(self, avar + '_range', st[1])

    def __getitem__(self, key):
        return getattr(self, key)

    def flatten(self, v):
        """Flattens the array into one dimension."""
        return ak.ravel(self.datum[v]).to_numpy()

    def sum_tracksters(self, v):
        """
        Sums the energy of each trackster binned in `v`. Helpful to, for instance,
        show how much energy each trackster deposited in each layer.
        All unique values of `v` are also returned for convenience.
        """
        un = np.unique(self.flatten(v))
        arr = []
        for n in range(self.ntracksters):
            arr.append([])
            for bin in un:
                arr[n].append(sum(self.datum['e'][n][self.datum[v][n]==bin]))

        # if computational efficiency is needed, try something like:
        # flat_int = lambda v : ak.flatten(datum[v]).to_numpy().astype(int)
        # wght_sum = lambda v : np.bincount(flat_int(v), weights=ak.flatten(datum['e']).to_numpy(), minlength=51)[1:]
        # the first entry corresponds to "layer 0", which does not exist, hence [1:]
        return arr, un

class Plot():
    def __init__(self, outpath, lc, sh, evid, tag='default'):
        self.libraries = ('mpl', 'bokeh', 'dash')
        self.colors = palette * 100
        self.lc = lc
        self.sh = sh
        self.evid = evid
        self.tag = tag
        self._check_dims()

        self.vpairs = {('x', 'y'): ('x [cm]', 'y [cm]'),
                       ('z', 'R'): ('z [cm]', 'R [cm]'),
                       ('z', 'x'): ('z [cm]', 'x [cm]'),
                       ('z', 'y'): ('z [cm]', 'y [cm]'),
                       # ('L', 'R'): ('Layer', 'R [cm]'),
                       # ('L', 'x'): ('Layer', 'x [cm]'),
                       # ('L', 'y'): ('Layer', 'y [cm]'),
                       }

        self.opt_line_z = dict(x=[364.5, 364.5], color='gray', line_dash='dashed')
        self.opt_line_L = dict(x=[26.5, 26.5],   color='gray', line_dash='dashed')

        self.savefig = lambda x : op.join(outpath, x)
        
        self.width = {'bokeh': 800, 'dash': 800, 'mpl': 800}

    def _add_vertical_line(self, fig, data, avars, lib):
        assert lib in self.libraries
        
        if avars[0] in ('z', 'L'):
            xmin, xmax = ak.min(data['x']), ak.max(data['x'])
            ymin, ymax = ak.min(data['y']), ak.max(data['y'])
            rmin, rmax = ak.min(data['R']), ak.max(data['R'])
            if lib == 'bokeh':
                if avars[0] == 'z':
                    if avars[1] == 'x':
                        fig.line(y=[xmin, xmax], **self.opt_line_z)
                    if avars[1] == 'y':
                        fig.line(y=[ymin, ymax], **self.opt_line_z)
                    elif avars[1] == 'R':
                        fig.line(y=[rmin, rmax], **self.opt_line_z)
                elif avars[0] == 'L':
                    if avars[1] == 'x':
                        fig.line(y=[xmin, xmax], **self.opt_line_L)
                    elif avars[1] == 'y':
                        fig.line(y=[ymin, ymax], **self.opt_line_L)
                    elif avars[1] == 'R':
                        fig.line(y=[rmin, rmax], **self.opt_line_L)
            elif lib == 'dash':
                raise NotImplementedError()
            elif lib == 'mpl':
                raise NotImplementedError()

    def bokeh(self):
        """Produces a bokeh interactive plot layout."""
        fname = 'events_display_' + self.tag
        ext = '.html'
        bp.output_file(self.savefig(fname) + ext)
        lay = layout(self._create_layout('bokeh'))
        bp.save(lay)

    def _check_dims(self):
        assert len(self.lc) == len(self.evid)
        assert len(self.sh) == len(self.evid)

    def dash(self):
        """Produces a dash interactive plot layout."""
        self._create_layout('dash')

    def _data_single_event(self, evid):
        """Access single event data"""
        return self.lc[evid], self.sh[evid]

    def _data_manip_single_event(self, evid):
        """Access single event data with helper class"""
        lc_datum = self.lc[evid]
        sh_datum = self.sh[evid]
        return DatumManip(lc_datum, tracksters=True), DatumManip(sh_datum)     

    def mpl(self):
        NotImplementedError()

    def _common_bokeh_attributes(self, fig):
        fig.output_backend = 'svg'
        fig.toolbar.logo = None
        fig.title.align = "left"
        fig.title.text_font_size = "15px"
        fig.ygrid.grid_line_alpha = 0.5
        fig.ygrid.grid_line_dash = [6, 4]
        fig.xgrid.grid_line_color = None
        fig.outline_line_color = None

    def _create_layout(self, lib):
        lay = []

        for idat in self.evid: # loop per event
            row_hi, row_lo = ([] for _ in range(2))

            lc_manip, sh_manip = self._data_manip_single_event(idat)
            lc_source, sh_source = self._preprocess(idat, lib)
                                                   
            for vk,vv in self.vpairs.items():
                if lib == 'bokeh':
                    popt = dict(ev=self.evid[idat], avars=vk, labels=vv)

                    p = self._single_bokeh_plot(lc_source, sh_source, **popt)
                    self._add_vertical_line(fig=p, data=sh_manip, avars=vk, lib=lib)

                    zoomx, zoomy = self._zoom_ranges(sh_manip, avars=vk, gapx=0.03, gapy=0.1)
                    p.x_range = bm.Range1d(zoomx[0], zoomx[1])
                    p.y_range = bm.Range1d(zoomy[0], zoomy[1])
                    
                    p_lo = self._single_bokeh_subplot(p, lc_manip, sh_manip, **popt)
                    slider = self._single_bokeh_slider(lc_source, sh_source)
                    
                elif lib == 'dash':
                    p = self._single_dash_plot()
                elif lib == 'mpl':
                    p = self._single_mpl_plot()

                row_hi.append(p)
                if lib == 'bokeh':
                    row_lo.append(p_lo)

            if lib == 'bokeh':
                lay.append(slider)
            lay.append(row_hi)
            if lib == 'bokeh':
                lay.append(row_lo)

        return lay

    def _preprocess(self, evid, lib):
        assert lib in self.libraries

        lc_datum, sh_datum = self._data_single_event(evid)
        lc_manip, sh_manip = self._data_manip_single_event(evid)
        
        # handle colors of layer clusters
        lc_counts = [ak.count(x) for x in lc_datum['x']]
        lc_colors = []
        for ilcc,lcc in enumerate(lc_counts):
            nc = self.colors[ilcc]
            lc_colors.extend([nc for _ in range(lcc)])
        lc_colors = np.array(lc_colors)
            
        sf = 3
        
        lc_source = dict(x=lc_manip.x, y=lc_manip.y, z=lc_manip.z, R=lc_manip.R, L=lc_manip.L,
                         size=sf*lc_manip.e, size_init=np.copy(sf*lc_manip.e), c=lc_colors)
        sh_source = dict(x=sh_manip.x, y=sh_manip.y, z=sh_manip.z, R=sh_manip.R, L=sh_manip.L,
                         size=2*sf*sh_manip.e, size_init=np.copy(2*sf*sh_manip.e),
                         c=[self.colors[k] for k in sh_manip.c])

        if lib == 'bokeh':
            lc_source = bm.ColumnDataSource(data=lc_source)
            sh_source = bm.ColumnDataSource(data=sh_source)
        elif lib == 'dash':
            lc_source = pd.DataFrame(lc_source)
            sh_source = pd.DataFrame(sh_source)
        elif lib == 'mpl':
            raise NotImplementedError()

        return lc_source, sh_source

    def _single_bokeh_plot(self, lc_source, sh_source, ev, avars, labels):
        p = bp.figure(height=500, width=self.width['bokeh'], background_fill_color="white",
                      title="Event {} | {} vs. {}".format(ev, avars[0], avars[1]),
                      tools="pan,save,box_select,box_zoom,wheel_zoom,reset,undo,redo")
        p.circle(x=avars[0], y=avars[1], color='c', size='size', source=lc_source, legend_label="LCs")
        p.circle(x=avars[0], y=avars[1], color='c', size='size', alpha=0.3, source=sh_source, legend_label="SimHits")

        self._common_bokeh_attributes(p)
        
        p.ygrid.grid_line_color = None
        p.yaxis.axis_label = labels[1]
        p.min_border_bottom = 0
        p.legend.click_policy='hide'
        p.legend.location = "top_right"
        p.legend.label_text_font_size = '12pt'
        #p.xaxis.major_label_text_font_size = '0pt'
        return p

    def _single_bokeh_slider(self, lc_source, sh_source):
        """Creates slider to control widget size."""
        slider = bm.Slider(title='Layer Cluster size', value=1., start=0.1, end=10., step=0.1, width=self.width['bokeh'])
        callback = bm.CustomJS(args=dict(s1=lc_source, s2=sh_source), code="""
        var val = cb_obj.value;
        var data1 = s1.data;
        var data2 = s2.data;
        for (var i=0; i<data1.size_init.length; i++) {
        data1.size[i] = val * data1.size_init[i];
        }
        for (var i=0; i<data2.size_init.length; i++) {
        data2.size[i] = val * data2.size_init[i];
        }
        s1.change.emit();
        s2.change.emit();
        """)
        slider.js_on_change('value', callback)
        return slider

    def _single_bokeh_subplot(self, main_plot, lc_data, sh_data, ev, avars, labels):
        """
        Subplot to be placed under the main plot.
        Shows a 1D distribution of summer layer cluster energy.
        """
        p = bp.figure(height=150, width=self.width['bokeh'], background_fill_color="white",
                      title="", x_range=main_plot.x_range, tools="pan")

        self._common_bokeh_attributes(p)
        
        thisxrange = lc_data[avars[0]+'_range']
        for itrk in range(lc_data.ntracksters):
            thisc = self.colors[itrk]
            p.circle(x=thisxrange, y=lc_data[avars[0]+'_trk'][itrk], color=thisc, size=6)
            p.line(x=thisxrange, y=lc_data[avars[0]+'_trk'][itrk], line_color=thisc, line_width=2)

        p.yaxis.axis_label = "Energy [GeV]"
        p.min_border_top = 0
        p.ygrid.grid_line_color = "gray"
        p.xaxis.axis_label = labels[0]

        return p


    def _single_dash_plot(self):
        raise NotImplementedError()

    def _single_mpl_plot(self):
        raise NotImplementedError()
    
    def _zoom_ranges(self, data, avars, gapx, gapy):
        """
        Provide the ranges of a zoom on the data, with respect to its minima and maxima.
        Cuts `gapx/y` percent of the ranges on both sides.
        """
        max_x, min_x = np.max(data[avars[0]]), np.min(data[avars[0]])
        diff_x = max_x - min_x
        max_y, min_y = np.max(data[avars[1]]), np.min(data[avars[1]])
        diff_y = max_y - min_y
        x_range = min_x + gapx*diff_x, max_x - gapx*diff_x
        y_range = min_y + gapy*diff_y, max_y - gapy*diff_y
        return x_range, y_range

class DisplayEvent():
    def __init__(self, tree, infiles, outpath, tag):
        self.tag = tag

        allvars = self._select_vars()
        lc_data, sh_data = ([] for _ in range(2))
        nplots = 4
        for batch in tqdm(up.iterate(infiles + ":" + tree, 
                                     step_size=nplots, library='ak', # load only the events to plot for efficiency
                                     filter_name="/" + "|".join(allvars) + "/"), total=len(glob.glob(infiles))):
            for iev in range(nplots):
                lcs = self._process_layer_cluster(batch, iev)
                simhits = self._process_sim_hits(batch, iev)
                lc_data.append(lcs)
                sh_data.append(simhits)
            break

        p = Plot(outpath, lc_data, sh_data, [x for x in range(nplots)], tag)
        p.bokeh()
        # p.dash()
        
    def _calculate_radius(self, x, y):
        return [np.sqrt(a**2 + b**2) for a,b in zip(x,y)]

    def _process_layer_cluster(self, data, evid):
        ntracksters = ak.count(data.multiclus_eta, axis=1)[evid]
        get_info = lambda v : [getattr(data, 'cluster2d_' + v)[evid, data.multiclus_cluster2d[evid][k]]
                               for k in range(ntracksters)]

        lc_x = get_info('x')
        lc_y = get_info('y')
        lc_r = self._calculate_radius(lc_x, lc_y)
        lc_z = get_info('z')
        lc_l = get_info('layer')
        lc_e = get_info('energy')
        return {'x': lc_x, 'y': lc_y, 'z': lc_z, 'R': lc_r, 'L': lc_l, 'e': lc_e}

    def _process_sim_hits(self, data, evid):
        # when there is pile-up, gun is the first calo particle (second endcap here)
        idx = data['simcluster_hits_indices'][evid][0]
        idx = idx[idx > -1]
        frac = data['simcluster_fractions'][evid][0]
        frac = frac[idx > -1]

        sh_x = data.rechit_x[evid][idx]
        sh_y = data.rechit_y[evid][idx]
        sh_r = self._calculate_radius(sh_x, sh_y)
        sh_z = data.rechit_z[evid][idx]
        sh_l = data.rechit_layer[evid][idx]
        sh_e = frac * data.rechit_energy[evid][idx]
        sh_c = data.rechit_cluster2d[evid][idx]

        return {'x': sh_x, 'y': sh_y, 'z': sh_z, 'R': sh_r, 'L': sh_l, 'e': sh_e, 'c': sh_c}

    def _select_vars(self):
        """Selects the columns to load. Supports regular expressions."""
        v = ["gunparticle_.*", "multiclus_.*", "cluster2d_.*",]
        v += ["rechit_.*", "simcluster_hits_indices", "simcluster_fractions"]
        #v += ["simcluster_.*", "gen_.*"]
        v += ["event",]
        return set(v)
        
    
def event_display(args):
    """Data exploration."""
    outpath = "/eos/user/b/bfontana/www/HadronReco"

    base = "/data_CMS/cms/alves"
    tree = "ana/hgc"
    infiles = (op.join(base, "SinglePion_0PU_10En200_11Jul/step3/step3_1.root"),
               #op.join(base, "SinglePion_0PU_10En200_30Jun/step3_linking/step3_*.root")
               )
    tags = ('clue3d',) #('clue3d', 'linking')
    for inf,tag in zip(infiles,tags):
        de = DisplayEvent(tree, inf, outpath=outpath, tag="single_" + tag)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data exploration of single pion gun.')
    parser.add_argument('--tag', default='default',
                        help='Tag to store and load the histograms. Skips histogram production. Useful when only plot tweaks are necessary.')
    FLAGS = parser.parse_args()

    event_display(FLAGS)
