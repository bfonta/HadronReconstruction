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
    def __init__(self, datum):
        self.datum = datum
        avars = ('x', 'y', 'z', 'R', 'L', 'e')
        self.ntracksters = len(self.datum['e'])
        for avar in avars:
            # self.x, self.y, ...
            setattr(self, avar, self.flatten(avar))
            # self.x_un, self.y_un, ...
            setattr(self, avar + '_un', np.unique(getattr(self, avar)))
            # self.x_trk, self.y_trk, ...
            st = self.sum_tracksters(avar)
            # self.x_trk, self.y_trk, ...
            setattr(self, avar + '_trk', st[0])
            # self.x_range, self.y_range, ...
            setattr(self, avar + '_range', st[1])

    def __getitem__(self, key):
        return getattr(self, key)

    def flatten(self, v):
        """Flattens the array into one dimension."""
        return ak.flatten(self.datum[v]).to_numpy()

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
        event_data = []
        nplots = 20
        for batch in tqdm(up.iterate(infiles + ":" + tree, 
                                     step_size=10000, library='ak',
                                     filter_name="/" + "|".join(allvars) + "/"), total=len(glob.glob(infiles))):
            for iev in range(nplots):
                event_data.append(self._process_event(batch, iev))
            break

        self._plot_bokeh(event_data, [x for x in range(nplots)])

    def _plot_bokeh(self, data, evid):
        assert len(data) == len(evid)
        colors = palette * 10

        vpairs = (('x', 'y'),
                  ('z', 'R'), ('z', 'x'), ('z', 'y'),
                  ('L', 'R'), ('L', 'x'), ('L', 'y'))
        vlabels = (('x [cm]', 'y [cm]'),
                   ('z [cm]', 'R [cm]'), ('z [cm]', 'x [cm]'), ('z [cm]', 'y [cm]'),
                   ('Layer', 'R [cm]'), ('Layer', 'x [cm]'),  ('Layer', 'y [cm]'))
        line_opt_z = dict(x=[364.5, 364.5], color='gray', line_dash='dashed')
        line_opt_L = dict(x=[26.5, 26.5],   color='gray', line_dash='dashed')
        lay = []
        for idat, datum in enumerate(data):
            row_hi, row_lo = ([] for _ in range(2))
            lc_counts = [ak.count(x) for x in datum['x']]
            lc_colors, id_tracksters = ([] for _ in range(2))
            for ilcc,lcc in enumerate(lc_counts):
                id_tracksters.extend([ilcc for _ in range(lcc)])
                nc = colors[ilcc]
                lc_colors.extend([nc for _ in range(lcc)])

            lc_colors = np.array(lc_colors)
            id_tracksters = np.array(id_tracksters)
            
            manip = DatumManip(datum)
            source = bm.ColumnDataSource(data=dict(x=manip.x, y=manip.y, z=manip.z, R=manip.R, L=manip.L,
                                                   size=4*manip.e, size_init=np.copy(4*manip.e), c=lc_colors))

            en_x_bins = [sum(manip.e[manip.x == xx]) for xx in manip.x_un]
            en_L_bins = [sum(manip.e[manip.L == xx]) for xx in manip.L_un]
            en_z_bins = [sum(manip.e[manip.z == xx]) for xx in manip.z_un]
            source_long = bm.ColumnDataSource(data=dict(
                z=manip.z_un, L=manip.L_un, en_z=en_z_bins, en_L=en_L_bins))
            source_tran = bm.ColumnDataSource(data=dict(x=manip.x_un, en_x=en_x_bins))
                                                   
            for iv,vp in enumerate(vpairs):
                p = bp.figure(height=350, width=700, background_fill_color="white",
                              title="Event {} | {} vs. {}".format(evid[idat], vp[0], vp[1]))                    
                p.circle(x=vp[0], y=vp[1], color='c', size='size', source=source)

                p_lo = bp.figure(height=150, width=700, background_fill_color="white", title="", x_range=p.x_range)
                # stmp = source_tran if vp[0] == 'x' else source_long
                # p_lo.circle(x=vp[0], y='en_'+vp[0], color='black', size=5, source=stmp)
                # p_lo.line(x=vp[0], y='en_'+vp[0], line_color='black', line_width=2, source=stmp)

                thisxrange = manip[vp[0]+'_range']
                for itrk in range(manip.ntracksters):
                    thisc = colors[itrk]
                    p_lo.circle(x=thisxrange, y=manip[vp[0]+'_trk'][itrk], color=thisc, size=6)
                    p_lo.line(x=thisxrange, y=manip[vp[0]+'_trk'][itrk], line_color=thisc, line_width=2)

                for itrk in np.unique(id_tracksters):
                    ax = ak.flatten(datum['x'])[id_tracksters == itrk]
                    ay = ak.flatten(datum['e'])[id_tracksters == itrk]
                    ac = lc_colors[id_tracksters == itrk]
                    #p_lo.line(x=ax.to_numpy(), y=ay.to_numpy(), line_color=ac[0], line_width=2)
                    break
                
                if vp[0] in ('z', 'L'):
                    xmin, xmax = ak.min(datum['x']), ak.max(datum['x'])
                    ymin, ymax = ak.min(datum['y']), ak.max(datum['y'])
                    rmin, rmax = ak.min(datum['R']), ak.max(datum['R'])
                    if vp[0] == 'z':
                        if vp[1] == 'x':
                            p.line(y=[xmin, xmax], **line_opt_z)
                        if vp[1] == 'y':
                            p.line(y=[ymin, ymax], **line_opt_z)
                        elif vp[1] == 'R':
                            p.line(y=[rmin, rmax], **line_opt_z)
                    elif vp[0] == 'L':
                        if vp[1] == 'x':
                            p.line(y=[xmin, xmax], **line_opt_L)
                        elif vp[1] == 'y':
                            p.line(y=[ymin, ymax], **line_opt_L)
                        elif vp[1] == 'R':
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
                p.yaxis.axis_label = vlabels[iv][1]
                p.min_border_bottom = 0
                #p.xaxis.major_label_text_font_size = '0pt'
                p_lo.yaxis.axis_label = "Energy [GeV]"
                p_lo.min_border_top = 0
                p_lo.ygrid.grid_line_color = "gray"
                p_lo.xaxis.axis_label = vlabels[iv][0]
                
                row_hi.append(p)
                row_lo.append(p_lo)

            slider = bm.Slider(title='Layer Cluster size', value=1., start=0.1, end=4., step=0.1, width=700)
            callback = bm.CustomJS(args=dict(source=source), code="""
            var val = cb_obj.value;
            var data = source.data;
            for (var i=0; i<data.size_init.length; i++) {
            data.size[i] = val * data.size_init[i];
            }
            source.change.emit();
            """)
            slider.js_on_change('value', callback)
            lay.append(slider)
            lay.append(row_hi)
            lay.append(row_lo)
        
        bp.output_file(self.savef('events_display_' + self.tag)+'.html')
        lay = layout(lay)
        bp.save(lay)

    def _process_event(self, data, evid):
        ntracksters = ak.count(data.multiclus_eta, axis=1)[evid]
        get_info = lambda v : [getattr(data, 'cluster2d_' + v)[evid, data.multiclus_cluster2d[evid][k]]
                               for k in range(ntracksters)]

        lc_x = get_info('x')
        lc_y = get_info('y')
        lc_z = get_info('z')
        lc_l = get_info('layer')
        lc_e = get_info('energy')
        lc_r = [np.sqrt(x**2 + y**2) for x,y in zip(lc_x,lc_y)]
        return {'x': lc_x, 'y': lc_y, 'z': lc_z, 'R': lc_r, 'L': lc_l, 'e': lc_e}
                    
    def _select_vars(self):
        v = ["gunparticle_.*", "multiclus_.*", "cluster2d_.*",]
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
