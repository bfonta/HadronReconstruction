# coding: utf-8

_all_ = [ 'explore' ]

import os
from os import path as op
import argparse
import numpy as np
import uproot as up
import awkward as ak
import hist
import itertools
from tqdm import tqdm
import pickle
import glob

from bokeh.palettes import Dark2_5 as palette
from bokeh.models import ColumnDataSource, Whisker
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import layout

import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

class AccumulateHistos():
    def __init__(self, tree, infiles, tag):
        self.nevents = 0
        self.nbins = 50
        #self.types = ('hgens', 'hgendiffs', 'hrh')
        self.types = ('hgens', 'hgendiffs')
        self.vgens = ('both', 'first', 'second')
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
        self.hgens['both'].fill(en  = ak.flatten(data.gen_energy),
                                eta = ak.flatten(data.gen_eta),
                                phi = ak.flatten(data.gen_phi),
                                pt  = ak.flatten(data.gen_pt),
                                )
        self.hgens['first'].fill(en  = data.gen_energy[:,0],
                                 eta = data.gen_eta[:,0],
                                 phi = data.gen_phi[:,0],
                                 pt  = data.gen_pt[:,0],
                                 )
        self.hgens['second'].fill(en  = data.gen_energy[:,1],
                                  eta = data.gen_eta[:,1],
                                  phi = data.gen_phi[:,1],
                                  pt  = data.gen_pt[:,1],
                                  )
        self.hgendiffs.fill(en  = data.gen_energy[:,1] - data.gen_energy[:,0],
                            eta = data.gen_eta[:,1]    - data.gen_eta[:,0],
                            phi = (data.gen_phi[:,1]    - data.gen_phi[:,0])*320.5,
                            pt  = data.gen_pt[:,1]     - data.gen_pt[:,0],
                            )
        # self.hrh.fill(en  = ak.flatten(data.rechit_energy),
        #               eta = ak.flatten(data.rechit_eta),
        #               phi = ak.flatten(data.rechit_phi),
        #               pt  = ak.flatten(data.rechit_pt),
        #               )

    def _load(self):
        for t in self.types:
            with open(op.join(self.adir, t + self.pickle_ext), 'rb') as f:
                setattr(self, t, pickle.load(f))
        self.nevents = len(self.hgens['first'].view())
        
    def _save(self, tree, infiles):
        # gen variables
        setattr(self, 'hgens', {k:
                                hist.Hist(
                                    hist.axis.Regular(self.nbins, 0,    210, name="en"),
                                    hist.axis.Regular(self.nbins, 1.5,  2.9, name="eta"),
                                    hist.axis.Regular(self.nbins, -3.2, 3.2, name="phi"),
                                    hist.axis.Regular(self.nbins, 0,    85, name="pt"),
                                )
                                for k in self.vgens})

        # gen diffs
        setattr(self, 'hgendiffs', hist.Hist(
            hist.axis.Regular(self.nbins, -15, 15, name="en"),
            hist.axis.Regular(self.nbins, -.3, .3, name="eta"),
            hist.axis.Regular(self.nbins, -5, 5, name="phi"),
            hist.axis.Regular(self.nbins, -15, 15, name="pt"),
        ))

        # rechits
        # setattr(self, 'hrh', hist.Hist(
        #     hist.axis.Regular(self.nbins, 0,    2.5, name="en"),
        #     hist.axis.Regular(self.nbins, -3.2, 3.2, name="eta"),
        #     hist.axis.Regular(self.nbins, -3.2, 3.2, name="phi"),
        #     hist.axis.Regular(self.nbins, 0,    2.5, name="pt"),
        # ))

        allvars = self._select_vars()
        for batch in tqdm(up.iterate(infiles + ":" + tree, 
                                     step_size=1000, library='ak', 
                                     filter_name="/" + "|".join(allvars) + "/"), total=len(glob.glob(infiles))):
            self.nevents += ak.count(batch.event)
            self._accumulate(batch)

        # store
        for t in self.types:
            with open(op.join(self.adir, t + self.pickle_ext), 'wb') as f:
                pickle.dump(getattr(self, t), f)

    def gen_histos(self, mode):
        """Selection of histogram with generated variables."""
        assert mode in self.vgens
        return self.hgens[mode]
        
    def _select_vars(self):
        v = ["gen_.*"]
        v += ["rechit_energy", "rechit_eta", "rechit_phi", "rechit_pt"]
        v += ["event", "track_energy"]
        return set(v)

def plot_bokeh(hists, title, xlabel, legs, ylog=False):
    """Plot using the Bokeh package"""
    if not isinstance(hists, (tuple,list)):
        hists = [hists]
    if not isinstance(legs, (tuple,list)):
        legs = [legs]
    assert len(hists) == len(legs)

    colors = itertools.cycle(palette)
    
    p = figure(height=500, width=500, background_fill_color="#efefef", title=title,
               y_axis_type='log' if ylog else 'linear')
    for h,l in zip(hists, legs):
        # source = ColumnDataSource(data=dict(top=h.values(), bottom=np.zeros_like(h.axes[0].centers)+1e-2,
        #                                     left=h.axes[0].edges[:-1], right=h.axes[0].edges[1:]))
        # p.quad(top='top', bottom='bottom', left='left', right='right', source=source,
        #        legend_label=l, fill_color=next(colors), line_color="white", alpha=0.5)
        source = ColumnDataSource(data=dict(y=h.values(), x=h.axes[0].centers))
        step_opt = dict(y='y', x='x', source=source, mode='center', line_color=next(colors))
        if len(legs)>1:
            step_opt.update({'legend_label': l})
        p.step(**step_opt)
        
    p.output_backend = 'svg'
    p.toolbar.logo = None
    if len(legs)>1:
        p.legend.click_policy='hide'
        p.legend.location = 'bottom_right'
        p.legend.label_text_font_size = '8pt'
    p.min_border_bottom = 5
    p.xaxis.visible = True
    p.title.align = "left"
    p.title.text_font_size = "15px"
    
    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    #p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = 'Counts'
        
    # whisk = Whisker(base="xscan", upper="upper", lower="lower", source=source,
    #                 level="annotation", line_width=8, line_color=c)
    # whisk.upper_head.size=8
    # whisk.lower_head.size=8
    # p.add_layout(whisk)
    return p
    
def plot_hist_mpl(hists, out, title, xlabel, legs, ylog=False):
    """Matplotlib histograms."""
    if not isinstance(hist, (tuple,list)):
        hists = [hists]

    colors = itertools.cycle(palette)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    if ylog:
        ax.set_yscale('log')
    
    for i, (h,leg) in enumerate(zip(hists,legs)):
        hep.histplot(h, ax=ax, color=next(colors), label=leg)
        #plt.plot(h.axes[0].centers, h.view(), "-o", color=next(colors), label=leg)

    if len(legs) > 1:
        plt.legend(loc="upper right")    
    #plt.tight_layout()

    hep.cms.text('Simulation')
    hep.cms.lumitext(title)
    for ext in ('.png', '.pdf'):
        plt.savefig(out + ext)

def explore(args):
    """Data exploration."""
    base = "/data_CMS/cms/alves"
    dist = "10cm"
    tree = "ana/hgc"
    outpath = "/eos/user/b/bfontana/www/HadronReco"

    
    infiles = op.join(base,
                      "DoublePion_0PU_10En200_3Jul_{}/step3/step3_1[0-5][0-9].root".format(dist)
                      )
    hacc = AccumulateHistos(tree, infiles, tag=args.tag)

    savef = lambda x : op.join(outpath, dist + x)
    title = "Double π ({}), {} events".format(dist, hacc.nevents)
    opt = dict(title=title)

    avars = ('en', 'eta', 'phi', 'pt')
    xlabels = {'en': "Energy [GeV]", 'eta': "η", 'phi': "ϕ", 'pt': 'pT [GeV]'}
    
    for avar in avars:
        opt = dict(legs=[''], title=title)
        plot_hist_mpl(hacc.gen_histos('both').project(avar),
                      xlabel=xlabels[avar], out=savef("_"+avar+"_both"), **opt)

    for avar in avars:
        opt = dict(legs=['τ #1', 'τ #2'], title=title)
        plot_hist_mpl([hacc.gen_histos('first').project(avar), hacc.gen_histos('second').project(avar)],
                  xlabel=xlabels[avar], out=savef("_"+avar+"_split"), **opt)

    for avar in avars:
        opt = dict(legs=[''], title=title)
        plot_hist_mpl(hacc.hgendiffs.project(avar),
                      xlabel='Arc-dist [cm]' if avar=='phi' else xlabels[avar],
                      out=savef("_"+avar+"_diff"), **opt)

    # for avar in avars:
    #     opt = dict(legs=[''], title=title)
    #     plot_hist_mpl(hacc.hrh.project(avar),
    #                   xlabel=xlabels[avar], out=savef("_"+avar+"_rh"), **opt)

    output_file(savef('_bokeh')+'.html')
    genboth_row = []
    for avar in avars:
        opt = dict(legs=[''])
        p = plot_bokeh(hacc.gen_histos('both').project(avar), title=title,
                       xlabel=xlabels[avar], **opt)
        genboth_row.append(p)

    gensplit_row = []
    for avar in avars:
        opt = dict(legs=['τ #1', 'τ #2'])
        p = plot_bokeh([hacc.gen_histos('first').project(avar), hacc.gen_histos('second').project(avar)],
                        title=title, xlabel=xlabels[avar], **opt)
        gensplit_row.append(p)

    gendiff_row = []
    for avar in avars:
        opt = dict(legs=[''])
        p = plot_bokeh(hacc.hgendiffs.project(avar), title=title,
                       xlabel=xlabels[avar], **opt)
        gendiff_row.append(p)

    # rh_row = []
    # for avar in avars:
    #     opt = dict(legs=[''], ylog=True if avar=='pt' else False)
    #     p = plot_bokeh(hacc.hrh.project(avar), xlabel=xlabels[avar], **opt)
    #     rh_row.append(p)

    #lay = layout([genboth_row, gensplit_row, gendiff_row, rh_row])
    lay = layout([genboth_row, gensplit_row, gendiff_row])
    save(lay)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data exploration.')
    parser.add_argument('--tag', default='v1',
                        help='Tag to store and load the histograms. Skips histogram production. Useful when only plot tweaks are necessary.')
    FLAGS = parser.parse_args()

    explore(FLAGS)
