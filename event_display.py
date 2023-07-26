# Coding: utf-8

_all_ = [ 'event_display' ]

import os; from os import path as op
from tqdm import tqdm
import argparse
import awkward as ak
import glob
import numpy as np
import uproot as up

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
        
class DisplayEvent():
    def __init__(self, tree, infiles, outpath, tag):
        self.tag = tag
        self.savef = lambda x : op.join(outpath, x)

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

        self._plot_bokeh(lc_data, sh_data, [x for x in range(nplots)])

    def _calculate_radius(self, x, y):
        return [np.sqrt(a**2 + b**2) for a,b in zip(x,y)]
    
    def _plot_bokeh(self, lc_data, sh_data, evid):
        assert len(lc_data) == len(evid)
        assert len(sh_data) == len(evid)

        colors = palette * 100
        
        vpairs = {('x', 'y'): ('x [cm]', 'y [cm]'),
                  ('z', 'R'): ('z [cm]', 'R [cm]'),
                  ('z', 'x'): ('z [cm]', 'x [cm]'),
                  ('z', 'y'): ('z [cm]', 'y [cm]'),
                  # ('L', 'R'): ('Layer', 'R [cm]'),
                  # ('L', 'x'): ('Layer', 'x [cm]'),
                  # ('L', 'y'): ('Layer', 'y [cm]'),
                  }

        line_opt_z = dict(x=[364.5, 364.5], color='gray', line_dash='dashed')
        line_opt_L = dict(x=[26.5, 26.5],   color='gray', line_dash='dashed')
        lay = []

        for idat, (lc_datum, sh_datum) in enumerate(zip(lc_data,sh_data)): # loop per event
            row_hi, row_lo = ([] for _ in range(2))

            # handle colors of layer clusters
            lc_counts = [ak.count(x) for x in lc_datum['x']]
            lc_colors, id_tracksters = ([] for _ in range(2))
            for ilcc,lcc in enumerate(lc_counts):
                id_tracksters.extend([ilcc for _ in range(lcc)])
                nc = colors[ilcc]
                lc_colors.extend([nc for _ in range(lcc)])
            lc_colors = np.array(lc_colors)
            id_tracksters = np.array(id_tracksters)

            sf = 3
            lc_manip = DatumManip(lc_datum, tracksters=True)
            lc_source = bm.ColumnDataSource(data=dict(x=lc_manip.x, y=lc_manip.y, z=lc_manip.z, R=lc_manip.R, L=lc_manip.L,
                                                      size=sf*lc_manip.e, size_init=np.copy(sf*lc_manip.e), c=lc_colors))
            sh_manip = DatumManip(sh_datum)
            sh_source = bm.ColumnDataSource(data=dict(x=sh_manip.x, y=sh_manip.y, z=sh_manip.z, R=sh_manip.R, L=sh_manip.L,
                                                      size=2*sf*sh_manip.e, size_init=np.copy(2*sf*sh_manip.e),
                                                      c=[colors[k] for k in sh_manip.c]))
                                                   
            for vk,vv in vpairs.items():
                width = 800
                p = bp.figure(height=500, width=width, background_fill_color="white",
                              title="Event {} | {} vs. {}".format(evid[idat], vk[0], vk[1]),
                              tools="pan,save,box_select,box_zoom,wheel_zoom,reset,undo,redo")
                p.circle(x=vk[0], y=vk[1], color='c', size='size', source=lc_source, legend_label="LCs")
                p.circle(x=vk[0], y=vk[1], color='c', size='size', alpha=0.3, source=sh_source, legend_label="SimHits")

                gap_x, gap_y = 0.03, 0.1
                _max_x, _min_x = np.max(sh_manip[vk[0]]), np.min(sh_manip[vk[0]])
                _diff_x = _max_x - _min_x
                _max_y, _min_y = np.max(sh_manip[vk[1]]), np.min(sh_manip[vk[1]])
                _diff_y = _max_y - _min_y
                p.x_range = bm.Range1d(_min_x + gap_x*_diff_x, _max_x - gap_x*_diff_x)
                p.y_range = bm.Range1d(_min_y + gap_y*_diff_y, _max_y - gap_y*_diff_y)
                
                p_lo = bp.figure(height=150, width=width, background_fill_color="white", title="",
                                 x_range=p.x_range, tools="pan")

                thisxrange = lc_manip[vk[0]+'_range']
                for itrk in range(lc_manip.ntracksters):
                    thisc = colors[itrk]
                    p_lo.circle(x=thisxrange, y=lc_manip[vk[0]+'_trk'][itrk], color=thisc, size=6)
                    p_lo.line(x=thisxrange, y=lc_manip[vk[0]+'_trk'][itrk], line_color=thisc, line_width=2)
                
                if vk[0] in ('z', 'L'):
                    xmin, xmax = ak.min(sh_datum['x']), ak.max(sh_datum['x'])
                    ymin, ymax = ak.min(sh_datum['y']), ak.max(sh_datum['y'])
                    rmin, rmax = ak.min(sh_datum['R']), ak.max(sh_datum['R'])
                    if vk[0] == 'z':
                        if vk[1] == 'x':
                            p.line(y=[xmin, xmax], **line_opt_z)
                        if vk[1] == 'y':
                            p.line(y=[ymin, ymax], **line_opt_z)
                        elif vk[1] == 'R':
                            p.line(y=[rmin, rmax], **line_opt_z)
                    elif vk[0] == 'L':
                        if vk[1] == 'x':
                            p.line(y=[xmin, xmax], **line_opt_L)
                        elif vk[1] == 'y':
                            p.line(y=[ymin, ymax], **line_opt_L)
                        elif vk[1] == 'R':
                            p.line(y=[rmin, rmax], **line_opt_L)

                for thisp in (p, p_lo):
                    thisp.output_backend = 'svg'
                    thisp.toolbar.logo = None
                    thisp.title.align = "left"
                    thisp.title.text_font_size = "15px"
                    thisp.ygrid.grid_line_alpha = 0.5
                    thisp.ygrid.grid_line_dash = [6, 4]
                    thisp.xgrid.grid_line_color = None
                    thisp.outline_line_color = None

                p.ygrid.grid_line_color = None
                p.yaxis.axis_label = vv[1]
                p.min_border_bottom = 0
                p.legend.click_policy='hide'
                p.legend.location = "top_right"
                p.legend.label_text_font_size = '12pt'

                #p.xaxis.major_label_text_font_size = '0pt'
                p_lo.yaxis.axis_label = "Energy [GeV]"
                p_lo.min_border_top = 0
                p_lo.ygrid.grid_line_color = "gray"
                p_lo.xaxis.axis_label = vv[0]
                
                row_hi.append(p)
                row_lo.append(p_lo)

            slider = bm.Slider(title='Layer Cluster size', value=1., start=0.1, end=10., step=0.1, width=width)
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
            lay.append(slider)
            lay.append(row_hi)
            lay.append(row_lo)
        
        bp.output_file(self.savef('events_display_' + self.tag)+'.html')
        lay = layout(lay)
        bp.save(lay)

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
    savef = lambda x : op.join(outpath, x)

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
