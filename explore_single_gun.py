# coding: utf-8

_all_ = [ 'explore_single_gun' ]

from tqdm import tqdm
import argparse
import awkward as ak
import glob
import hist
import itertools as it
import multiprocessing
import numpy as np
import os; from os import path as op
import pickle
import uproot as up
import warnings

from bokeh import models as bm
from bokeh.layouts import layout
from bokeh.palettes import Dark2_5 as palette
from bokeh.plotting import figure, save, output_file

import matplotlib; import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

class DisplayEvent():
    def __init__(self, tree, infiles, outpath, tag):
        self.savef = lambda x : op.join(outpath, x)

        allvars = self._select_vars()
        event_data = []
        nplots = 20
        for batch in tqdm(up.iterate(infiles + ":" + tree, 
                                     step_size=1000, library='ak', 
                                     filter_name="/" + "|".join(allvars) + "/"), total=len(glob.glob(infiles))):
            for iev in range(nplots):
                event_data.append(self._process_event(batch, iev))
            break

        self._plot_bokeh(event_data, [x for x in range(nplots)])

    def _plot_bokeh(self, data, evid):
        assert len(data) == len(evid)
        colors = it.cycle(palette)
        vpairs = (('x', 'y'),
                  ('z', 'R'), ('z', 'x'), ('z', 'y'),
                  ('L', 'R'), ('L', 'x'), ('L', 'y'))
        vlabels = (('x [cm]', 'y [cm]'),
                   ('z [cm]', 'R [cm]'), ('z [cm]', 'x [cm]'), ('z [cm]', 'y [cm]'),
                   ('Layer', 'R [cm]'), ('Layer', 'x [cm]'),  ('Layer', 'y [cm]'))
        line_opt_z = dict(x=[364.5, 364.5], color='gray', line_dash="dashed")
        line_opt_L = dict(x=[26.5, 26.5],   color='gray', line_dash="dashed")
        lay = []
        for idat, datum in enumerate(data):
            row = []
            lc_counts = [ak.count(x) for x in datum['x']]
            lc_colors = []
            for lcc in lc_counts:
                nc = next(colors)
                lc_colors.extend([nc for _ in range(lcc)])
     
            source = bm.ColumnDataSource(data=dict(x=ak.flatten(datum['x']).to_numpy(),
                                                   y=ak.flatten(datum['y']).to_numpy(),
                                                   z=ak.flatten(datum['z']).to_numpy(),
                                                   L=ak.flatten(datum['L']).to_numpy(),
                                                   R=ak.flatten(datum['R']).to_numpy(),
                                                   size=ak.flatten(datum['e']).to_numpy(),
                                                   size_init=ak.flatten(datum['e']).to_numpy(),
                                                   c=np.array(lc_colors)))

            for iv,vp in enumerate(vpairs):
                p = figure(height=400, width=700, background_fill_color="white",
                           title="Event {} | {} vs. {}".format(evid[idat], vp[0], vp[1]))
                p.circle(x=vp[0], y=vp[1], color='c', size='size', source=source)
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
                    
                p.output_backend = 'svg'
                p.toolbar.logo = None
                p.min_border_bottom = 5
                p.title.align = "left"
                p.title.text_font_size = "15px"
            
                p.xgrid.grid_line_color = None
                p.ygrid.grid_line_color = None

                p.xaxis.axis_label = vlabels[iv][0]
                p.yaxis.axis_label = vlabels[iv][1]
                
                row.append(p)

            slider = bm.Slider(title='Layer Cluster size (multiple of energy in GeV)', value=1., start=0.1, end=4., step=0.1, width=700)
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
            lay.append(row)
        
        output_file(self.savef('events_display')+'.html')
        lay = layout(lay)
        save(lay)

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
        v += ["event",]
        return set(v)
        
        
class AccumulateHistos():
    def __init__(self, tree, infiles, tag):
        self.nevents = 0
        self.nbins = 50
        self.types = ('hgen', 'htrackster', 'hntrackster',
                      'hfrac1', 'hfrac2', 'hfrac5', 'hfrac10')
        self.pickle_ext = ".pkl"
        self.adir = "histos_" + tag

        if op.isdir(self.adir) and len(os.listdir(self.adir)) == len(self.types):
            print('Loading histograms with tag {}...'.format(tag))
            self._load()
        else:
            if not op.isdir(self.adir):
                os.makedirs(self.adir)
            self._save(tree, infiles)

    def _accumulate(self, data):
        self.hgen.fill(en  = ak.flatten(data.gunparticle_energy[:, :, 0]),
                       eta = ak.flatten(data.gunparticle_eta[:, :, 0]),
                       phi = ak.flatten(data.gunparticle_phi[:, :, 0]),
                       pt  = ak.flatten(data.gunparticle_pt[:, :, 0]),
                       )

        self.htrackster.fill(en  = ak.flatten(data.multiclus_energy),
                             eta = ak.flatten(data.multiclus_eta),
                             phi = ak.flatten(data.multiclus_phi),
                             pt  = ak.flatten(data.multiclus_energy),
                             )

        self.hntrackster.fill(n = ak.count(data.multiclus_energy, axis=1),)

        frac = lambda n : ( ak.sum(ak.sort(data.multiclus_energy)[:,-n:], axis=1) /
                            ak.sum(data.multiclus_energy, axis=1) )
        self.hfrac1.fill(frac = frac(1))
        self.hfrac2.fill(frac = frac(2))
        self.hfrac5.fill(frac = frac(5))
        self.hfrac10.fill(frac = frac(10))

    def _load(self):
        for t in self.types:
            with open(op.join(self.adir, t + self.pickle_ext), 'rb') as f:
                setattr(self, t, pickle.load(f))

        self.nevents = int(sum(self.hgen.project('en').counts()))

    def _save(self, tree, infiles):
        # gen variables
        setattr(self, 'hgen', hist.Hist(
            hist.axis.Regular(self.nbins, 0,    1050, name="en"),
            hist.axis.Regular(self.nbins, 1.5,  2.9, name="eta"),
            hist.axis.Regular(self.nbins, -3.2, 3.2, name="phi"),
            hist.axis.Regular(self.nbins, 0,    600, name="pt"),
        ))

        setattr(self, 'htrackster', hist.Hist(
            hist.axis.Regular(self.nbins, 0,    20, name="en"),
            hist.axis.Regular(self.nbins, 1.5,  3.1,  name="eta"),
            hist.axis.Regular(self.nbins, -3.2, 3.2,  name="phi"),
            hist.axis.Regular(self.nbins, 0,    18,  name="pt"),
        ))

        nmax = 61
        nn = nmax+1
        setattr(self, 'hntrackster', hist.Hist(
            hist.axis.Regular(nn, 0, nmax,  name="n"),
        ))

        nb = 80
        setattr(self, 'hfrac1', hist.Hist(hist.axis.Regular(nb, 0.2, 1.02,  name="frac")))
        setattr(self, 'hfrac2', hist.Hist(hist.axis.Regular(nb, 0.2, 1.02,  name="frac")))
        setattr(self, 'hfrac5', hist.Hist(hist.axis.Regular(nb, 0.2, 1.02,  name="frac")))
        setattr(self, 'hfrac10', hist.Hist(hist.axis.Regular(nb, 0.2, 1.02,  name="frac")))

        allvars = self._select_vars()
        for batch in tqdm(up.iterate(infiles + ":" + tree, 
                                     step_size=1000, library='ak', 
                                     filter_name="/" + "|".join(allvars) + "/"), total=len(glob.glob(infiles))):
            self.nevents += int(ak.count(batch.event))
            self._accumulate(batch)
            
        # store
        for t in self.types:
            with open(op.join(self.adir, t + self.pickle_ext), 'wb') as f:
                pickle.dump(getattr(self, t), f)

    def _select_vars(self):
        v = ["gunparticle_.*", "multiclus_.*", "cluster2d_.*",]
        v += ["event", "track_energy"]
        return set(v)

    def _theta(self, eta):
        return 2 * np.arctan(np.exp(-eta))
        
def plot_bokeh(hists, title, legs, xlabel, legloc="top_right",
               ylabel=None, ylog=False, density=False, frac=False):
    """Plot using the Bokeh package"""
    if not isinstance(hists, (tuple,list)):
        hists = [hists]
    if not isinstance(legs, (tuple,list)):
        legs = [legs]
    assert len(hists) == len(legs)
    assert not (density and frac)
    colors = it.cycle(palette)
    
    p = figure(height=400, width=600, background_fill_color="white", title=title,
               y_axis_type='log' if ylog else 'linear')
    for h,l in zip(hists, legs):
        # p.quad(top='top', bottom='bottom', left='left', right='right', source=source,
        #        legend_label=l, fill_color=next(colors), line_color="white", alpha=0.5)
        if density:
            source = bm.ColumnDataSource(data=dict(y=h.density(), x=h.axes[0].centers))
        elif frac:
            source = bm.ColumnDataSource(data=dict(y=h.values()/h.sum(), x=h.axes[0].centers))
        else:
            source = bm.ColumnDataSource(data=dict(y=h.values(), x=h.axes[0].centers))
        step_opt = dict(y='y', x='x', source=source, mode='before', line_color=next(colors), line_width=3)
        if len(legs)>1:
            step_opt.update({'legend_label': l})
        p.step(**step_opt)
        
    p.output_backend = 'svg'
    p.toolbar.logo = None
    if len(legs)>1:
        p.legend.click_policy='hide'
        p.legend.location = legloc
        p.legend.label_text_font_size = '12pt'
    p.min_border_bottom = 5
    p.xaxis.visible = True
    p.title.align = "left"
    p.title.text_font_size = "15px"
    
    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    p.xaxis.axis_label = xlabel
    p.xaxis.axis_label_text_font_size = "10pt"
    p.yaxis.axis_label_text_font_size = "10pt"
    if ylabel is None:
        p.yaxis.axis_label = 'a.u.' if density else 'Counts'
    else:
        p.yaxis.axis_label = ylabel

    # whisk = bm.Whisker(base="xscan", upper="upper", lower="lower", source=source,
    #                 level="annotation", line_width=8, line_color=c)
    # whisk.upper_head.size=8
    # whisk.lower_head.size=8
    # p.add_layout(whisk)
    return p
    
def plot_hist_mpl(hists, out, title, xlabel, legs, ylabel=None, ylog=False, density=False):
    """Matplotlib histograms."""
    if not isinstance(hist, (tuple,list)):
        hists = [hists]

    colors = itertools.cycle(palette)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel("a.u." if density else "Counts")
    else:
        plt.ylabel(ylabel)
    if ylog:
        ax.set_yscale('log')
    
    for i, (h,leg) in enumerate(zip(hists,legs)):
        hep.histplot(h, ax=ax, color=next(colors), label=leg, density=density)

    if len(legs) > 1:
        plt.legend(loc="upper right")
    #plt.tight_layout()

    hep.cms.text('Simulation')
    hep.cms.lumitext(title)
    for ext in ('.png', '.pdf'):
        plt.savefig(out + ext)

    plt.close()
    
def explore_single_gun(args):
    """Data exploration."""
    outpath = "/eos/user/b/bfontana/www/HadronReco"
    savef = lambda x : op.join(outpath, x)

    base = "/data_CMS/cms/alves"
    tree = "ana/hgc"
    infiles = op.join(base, "SinglePion_0PU_10En200_30Jun/step3/step3_1.root")

    if args.display:
        de = DisplayEvent(tree, infiles, outpath=outpath, tag="single_" + args.tag)

    else:
        hacc = AccumulateHistos(tree, infiles, tag="single_" + args.tag)    
        title = "Single π, {} events".format(hacc.nevents)
        opt = dict(title=title)

        avars = ('en', 'eta', 'phi', 'pt')
        xlabels      = {'en': "Energy [GeV]", 'eta': "η", 'phi': "ϕ", 'pt': 'pT [GeV]'}
        xlabels_diff = {'en': "ΔE [GeV]", 'eta': "Δη", 'phi': "Δϕ", 'pt': 'ΔpT [GeV]'}

        # matplotlib
        # for avar in avars:
        #     opt = dict(legs=[''] if nd==1 else distances, title=title)
        #     plot_hist_mpl(hacc.hgen.project(avar),
        #                   xlabel=xlabels[avar], out=savef("single_"+avar+"_both"), **opt)
        
        # for avar in avars:
        #     opt = dict(legs=[''] if nd==1 else distances, title=title)
        #     plot_hist_mpl([hacc.hgendiffs.project(avar),
        #                   xlabel='Arc-dist [cm]' if avar=='phi' else xlabels[avar],
        #                   out=savef("single_"+avar+"_diff"), **opt)

        # bokeh
        output_file(savef('single_bokeh')+'.html')
        gen_row = []
        for avar in avars:
            opt = dict(legs=[''])
            p = plot_bokeh(hacc.hgen.project(avar),
                           title=title, xlabel=xlabels[avar], **opt)
            gen_row.append(p)

        trackster_row = []
        for avar in avars:
            opt = dict(legs=[''], ylog=True if avar=="pt" or avar=="en" else False)
            p = plot_bokeh(hacc.htrackster.project(avar),
                           title=title, xlabel=xlabels[avar], **opt)
            trackster_row.append(p)

        ntrackster_row = []
        opt = dict(legs=['highest-energy trackster', '2 highest-energy tracksters',
                         '3 highest-energy tracksters', '10 highest-energy tracksters'],
                   legloc="top_left")
        p = plot_bokeh([hacc.hfrac1.project("frac"), hacc.hfrac2.project("frac"),
                        hacc.hfrac5.project("frac"), hacc.hfrac10.project("frac")], title=title,
                       xlabel="Fraction of the total trackster energy in an event", **opt)
        ntrackster_row.append(p)
        opt = dict(legs=[''])
        # p = plot_bokeh(hacc.hntrackster.project("n"),
        #                title=title, xlabel="# Tracksters", **opt)
        # ntrackster_row.append(p)
        p = plot_bokeh(hacc.hntrackster.project("n"),
                       title=title, xlabel="# Tracksters", ylabel="Fraction of events",
                       frac=True, **opt)
        ntrackster_row.append(p)

        lay = layout([gen_row, ntrackster_row, trackster_row])
        save(lay)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data exploration of single pion gun.')
    parser.add_argument('--tag', default='default',
                        help='Tag to store and load the histograms. Skips histogram production. Useful when only plot tweaks are necessary.')
    parser.add_argument('--display', action="store_true",
                        help='Whether to display an event or to plot histograms.')
    FLAGS = parser.parse_args()
    if FLAGS.display and FLAGS.tag != parser.get_default('tag'):
        warnings.warn('Specifying a tag has no effect when using `--display`.')
    
    explore_single_gun(FLAGS)
