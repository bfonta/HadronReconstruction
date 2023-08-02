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

# bokeh
from bokeh import models as bm
from bokeh.layouts import layout
from bokeh.palettes import Colorblind as palette
palette = palette[8]
from bokeh import plotting as bp

# dash
from dash import Dash, dcc, html, Input, Output
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly import subplots as sp
import dash_bootstrap_components as dbc

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
        self._check_dims()

        self.outpath = outpath
        self.fname = 'events_display_' + tag

        self.vlabels = {'x': 'x [cm]', 'y': 'y [cm]', 'z': 'z [cm]',
                        'R': 'R [cm]', 'L': 'Layer'}
        self.vpairs = {('x', 'y'), ('z', 'R'), ('z', 'x'), ('z', 'y'),
                       # ('L', 'R'), ('L', 'x'), ('L', 'y'),
                       }

        self.opt_line_z = dict(x=[364.5, 364.5], color='gray', line_dash='dashed')
        self.opt_line_L = dict(x=[26.5, 26.5],   color='gray', line_dash='dashed')

        if not os.path.exists(op.join(outpath, 'events_dash')):
            os.makedirs(op.join(outpath, 'events_dash'))
        
        self.width = {'bokeh': 800, 'dash': 2000, 'mpl': 800}

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
        fn = self.fname + '_bokeh'
        ext = '.html'
        out = op.join(self.outpath, fn)
        bp.output_file(out + ext)
        lay = layout(self._create_layout('bokeh'))
        bp.save(lay)

    def _build_labels_pair(self, var_pair):
        """Transform a variable pair into their labels."""
        return (self.vlabels[var_pair[0]], self.vlabels[var_pair[1]])
         
    def _check_dims(self):
        assert len(self.lc) == len(self.evid)
        assert len(self.sh) == len(self.evid)

    def dash(self):
        """Produces a dash interactive plot layout."""
        app = Dash("my_dash")
        fn = self.fname + '_dash'
        ext = '.html'

        #out = op.join(self.outpath, 'events_dash', fn)
        out = op.join(self.outpath, fn)

        lay = self._create_layout('dash')
        nrows = len(lay)
        sp_titles = ['Event #{}'.format(ev) for ev in self.evid]
        fig = sp.make_subplots(rows=nrows, cols=1,
                               specs=[[{'type': 'scene'}]]*nrows,
                               subplot_titles=sp_titles,
                               horizontal_spacing = 0.05, vertical_spacing = 0.01,
                               )

        for il,l in enumerate(lay):
            trace_opt = dict(row=il+1, col=1)
            fig.add_trace(l[0], **trace_opt)
            fig.add_trace(l[1], **trace_opt)

        scene_def = dict(xaxis_title=self.vlabels['z'],
                         yaxis_title=self.vlabels['y'],
                         zaxis_title=self.vlabels['x'])
        scenes = {'scene' + str(n+1): scene_def for n in range(nrows)}
        scenes.update({'scene' + str(n+1) + '_aspectmode' : 'data' for n in range(nrows)})

        # camera initial position
        camera = dict(up=dict(x=0, y=0, z=1.), #camera tilt
                      center=dict(x=0, y=0, z=0.), #focal point
                      eye=dict(x=-2., y=-1., z=0.1))
        scenes.update({'scene' + str(n+1) + '_camera' : camera for n in range(nrows)})

        fig.update_layout(
            showlegend=True,
            template="plotly_white",
            autosize=False,
            width=self.width['dash'],
            height=3000,
            margin=dict(l=0,r=0,b=20,t=40,pad=0),
            paper_bgcolor="white",
            font_family="Courier New",
            font_color="black",
            title_font_family="Times New Roman",
            title_font_color="blue",
            legend_title_font_color="black",
            **scenes
        )

        plotly.offline.plot(fig, filename=out+ext, auto_open=False)

    def _data_single_event(self, evid):
        """Access single event data"""
        return self.lc[evid], self.sh[evid]

    def _data_manip_single_event(self, evid):
        """Access single event data with helper class"""
        lc_datum = self.lc[evid]
        sh_datum = self.sh[evid]
        return DatumManip(lc_datum, tracksters=True), DatumManip(sh_datum)     

    def mpl(self):
        # fn = self.fname + '_mpl'
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
        fig.toolbar.active_scroll = fig.select_one(bm.WheelZoomTool)

    def _create_layout(self, lib):
        """
        Creates list with single plots representing the layout on the browser.
        """
        lay = []

        for idat in self.evid: # loop per event
            row_hi, row_lo = ([] for _ in range(2))

            lc_manip, sh_manip = self._data_manip_single_event(idat)
            lc_source, sh_source = self._preprocess(idat, lib)

            # one plot per variable pair
            for vp in self.vpairs:
                popt = dict(ev=self.evid[idat], avars=vp, labels=self._build_labels_pair(vp))
                
                if lib == 'bokeh':
                    p = self._single_bokeh_plot(lc_source, sh_source, **popt)
                    self._add_vertical_line(fig=p, data=sh_manip, avars=vp, lib=lib)

                    zoomx, zoomy = self._zoom_ranges(sh_manip, avars=vp, gapx=0.03, gapy=0.1)
                    p.x_range = bm.Range1d(zoomx[0], zoomx[1])
                    p.y_range = bm.Range1d(zoomy[0], zoomy[1])
                    
                    p_lo = self._single_bokeh_subplot(p, lc_manip, sh_manip, **popt)
                    slider_lc = self._single_bokeh_slider(lc_source, title='Layer Clusters size')
                    slider_sh = self._single_bokeh_slider(sh_source, title='SimHits size')

                elif lib == 'dash':
                    continue
                    
                elif lib == 'mpl':
                    p = self._single_mpl_plot()

                row_hi.append(p)
                if lib == 'bokeh':
                    row_lo.append(p_lo)

            # general plots
            if lib == 'dash':
                row_hi = self._single_dash_plot(lc_source, sh_source, **popt)

            # finalize plot layout
            if lib == 'bokeh':
                lay.append([slider_lc, slider_sh])
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
                         e=lc_manip.e, size=sf*lc_manip.e, size_init=np.copy(sf*lc_manip.e), c=lc_colors)
        sh_source = dict(x=sh_manip.x, y=sh_manip.y, z=sh_manip.z, R=sh_manip.R, L=sh_manip.L,
                         e=sh_manip.e, size=2*sf*sh_manip.e, size_init=np.copy(2*sf*sh_manip.e),
                         c=[self.colors[k] for k in sh_manip.c])

        if lib == 'bokeh':
            lc_source = bm.ColumnDataSource(data=lc_source)
            sh_source = bm.ColumnDataSource(data=sh_source)
        elif lib == 'dash':
            lc_source = lc_source
            sh_source = sh_source
        elif lib == 'mpl':
            raise NotImplementedError()

        return lc_source, sh_source

    def _single_bokeh_plot(self, lc_source, sh_source, ev, avars, labels):
        tooltips = [("Energy [GeV]", "@e"),
                    ("(x, y, z) [cm]", "(@x, @y, @z)"),
                    ("R [cm], Layer", "@R, @L")]
        p = bp.figure(height=500, width=self.width['bokeh'], background_fill_color="white",
                      title="Event {} | {} vs. {}".format(ev, avars[0], avars[1]),
                      tools="pan,save,box_select,box_zoom,wheel_zoom,reset,undo,redo,hover",
                      tooltips=tooltips)
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

    def _single_bokeh_slider(self, source, title='', value=1., start=0.1, end=10., step=0.1):
        """Creates slider to control widget size."""
        slider = bm.Slider(title=title, value=value, start=start, end=end, step=step,
                           width=int(self.width['bokeh']/2))
        callback = bm.CustomJS(args=dict(s=source), code="""
        var val = cb_obj.value;
        var data = s.data;
        for (var i=0; i<data.size_init.length; i++) {
        data.size[i] = val * data.size_init[i];
        }
        s.change.emit();
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

    def _single_dash_plot(self, lc_source, sh_source, ev, avars, labels):
        p_lc = go.Scatter3d(x=lc_source['z'], y=lc_source['y'], z=lc_source['x'],
                            mode='markers', name='LCs (event #{})'.format(ev),
                            marker=dict(size=lc_source['size'],
                                        color=lc_source['c'],
                                        opacity=1.))
        p_sh = go.Scatter3d(x=sh_source['z'], y=sh_source['y'], z=sh_source['x'],
                            mode='markers', name='SimHits (event #{})'.format(ev),
                            marker=dict(size=sh_source['size'],
                                        color=sh_source['c'],
                                        opacity=1.))
        return p_lc, p_sh


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
        p.dash()
        
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
