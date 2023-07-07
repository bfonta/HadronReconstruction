# coding: utf-8

_all_ = [ 'explore' ]

import os
from os import path as op
import argparse
import numpy as np
import uproot as up
import awkward as ak
import hist
import itertools as it
from tqdm import tqdm
import pickle
import glob
import multiprocessing

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
        
        self.hgens['both'].fill(en  = ak.flatten(data.gunparticle_energy[:, :, 0]),
                                eta = ak.flatten(data.gunparticle_eta[:, :, 0]),
                                phi = ak.flatten(data.gunparticle_phi[:, :, 0]),
                                pt  = ak.flatten(data.gunparticle_pt[:, :, 0]),
                                )
        self.hgens['first'].fill(en  = data.gunparticle_energy[:, 0, 0],
                                 eta = data.gunparticle_eta[:, 0, 0],
                                 phi = data.gunparticle_phi[:, 0, 0],
                                 pt  = data.gunparticle_pt[:, 0, 0],
                                 )
        self.hgens['second'].fill(en  = data.gunparticle_energy[:, 1, 0],
                                  eta = data.gunparticle_eta[:, 1, 0],
                                  phi = data.gunparticle_phi[:, 1, 0],
                                  pt  = data.gunparticle_pt[:, 1, 0],
                                  )
        #R = 320.5 / np.cos(self._theta(data.gunparticle_eta[:,1]))
        self.hgendiffs.fill(en  = data.gunparticle_energy[:,1,0] - data.gunparticle_energy[:,0,0],
                            eta = data.gunparticle_eta[:,1,0]    - data.gunparticle_eta[:,0,0],
                            phi = (data.gunparticle_phi[:,1,0]   - data.gunparticle_phi[:,0,0]),
                            pt  = data.gunparticle_pt[:,1,0]     - data.gunparticle_pt[:,0,0],
                            )

    def _load(self):
        for t in self.types:
            with open(op.join(self.adir, t + self.pickle_ext), 'rb') as f:
                setattr(self, t, pickle.load(f))

        # meta data
        self.nevents = int(sum(self.gen_histos('first').project('en').counts()))
        # with open(op.join(self.adir, 'meta.txt'), 'r') as f:
        #     lines = f.readlines()
        #     line_num = -1
        #     for k,line in enumerate(lines):
        #         if line.find("nevents") != -1:
        #             breakpoint()
        #             self.nevents = line

    def gen_histos(self, mode):
        """Selection of histogram with generated variables."""
        assert mode in self.vgens
        return self.hgens[mode]

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
            hist.axis.Regular(self.nbins, -50, 50, name="en"),
            hist.axis.Regular(self.nbins, -0.02, 0.02, name="eta"),
            hist.axis.Regular(self.nbins, 0.01, 0.3, name="phi"),
            hist.axis.Regular(self.nbins, -35, 35, name="pt"),
        ))

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

        # meta data
        # with open(op.join(self.adir, 'meta.txt'), 'w') as f:
        #     f.write('nevents = {}'.format(self.nevents))

    def _select_vars(self):
        v = ["gunparticle_.*"]
        # v += ["rechit_energy", "rechit_eta", "rechit_phi", "rechit_pt"]
        v += ["event", "track_energy"]
        return set(v)

    def _theta(self, eta):
        return 2 * np.arctan(np.exp(-eta))
        
def plot_bokeh(hists, title, xlabel, legs, ylog=False, density=False):
    """Plot using the Bokeh package"""
    if not isinstance(hists, (tuple,list)):
        hists = [hists]
    if not isinstance(legs, (tuple,list)):
        legs = [legs]
    assert len(hists) == len(legs)

    colors = it.cycle(palette)
    
    p = figure(height=500, width=500, background_fill_color="white", title=title,
               y_axis_type='log' if ylog else 'linear')
    for h,l in zip(hists, legs):
        # p.quad(top='top', bottom='bottom', left='left', right='right', source=source,
        #        legend_label=l, fill_color=next(colors), line_color="white", alpha=0.5)
        if density:
            source = ColumnDataSource(data=dict(y=h.density(), x=h.axes[0].centers))
        else:
            source = ColumnDataSource(data=dict(y=h.values(), x=h.axes[0].centers))
        step_opt = dict(y='y', x='x', source=source, mode='center', line_color=next(colors), line_width=3)
        if len(legs)>1:
            step_opt.update({'legend_label': l})
        p.step(**step_opt)
        
    p.output_backend = 'svg'
    p.toolbar.logo = None
    if len(legs)>1:
        p.legend.click_policy='hide'
        p.legend.location = 'top_right'
        p.legend.label_text_font_size = '8pt'
    p.min_border_bottom = 5
    p.xaxis.visible = True
    p.title.align = "left"
    p.title.text_font_size = "15px"
    
    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = 'a.u.' if density else 'Counts'
        
    # whisk = Whisker(base="xscan", upper="upper", lower="lower", source=source,
    #                 level="annotation", line_width=8, line_color=c)
    # whisk.upper_head.size=8
    # whisk.lower_head.size=8
    # p.add_layout(whisk)
    return p
    
def plot_hist_mpl(hists, out, title, xlabel, legs, ylog=False, density=False):
    """Matplotlib histograms."""
    if not isinstance(hist, (tuple,list)):
        hists = [hists]

    colors = itertools.cycle(palette)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel(xlabel)
    plt.ylabel("a.u." if density else "Counts")
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

def accumulate_all(dist, tag):
    base = "/data_CMS/cms/alves"
    tree = "ana/hgc"
    infiles = op.join(base, "DoublePion_0PU_10En200_3Jul_{}/step3/step3_1[0-9].root".format(dist))
    #hacc.update({dist: AccumulateHistos(tree, infiles, tag=dist + '_' + tag)})
    return AccumulateHistos(tree, infiles, tag=dist + '_' + tag)
    
def explore(args):
    """Data exploration."""
    distances = tuple(x + "cm" for x in args.dist)
    nd = len(distances)

    outpath = "/eos/user/b/bfontana/www/HadronReco"
    savef = lambda x : op.join(outpath, dist + x)

    hacc = {}
    pool = multiprocessing.Pool(nd)
    # parallellize production
    ret = pool.starmap(accumulate_all, zip(distances, it.repeat(args.tag)))
    for i, dist in enumerate(distances): #store result in a dict
        hacc[dist] = ret[i]
    
    nevents = sum([hacc[k].nevents for k in hacc.keys()])
    if nd == 1:
        title = "Double π ({}), {} events".format(distances[0], nevents)
    else:
        title = "Double π, {} events".format(nevents)
    opt = dict(title=title)

    avars = ('en', 'eta', 'phi', 'pt')
    xlabels      = {'en': "Energy [GeV]", 'eta': "η", 'phi': "ϕ", 'pt': 'pT [GeV]'}
    xlabels_diff = {'en': "ΔE [GeV]", 'eta': "Δη", 'phi': "Δϕ", 'pt': 'ΔpT [GeV]'}

    # matplotlib
    # for avar in avars:
    #     opt = dict(legs=[''] if nd==1 else distances, title=title)
    #     plot_hist_mpl([hacc[k].gen_histos('both').project(avar) for k in hacc.keys()],
    #                   xlabel=xlabels[avar], out=savef("_"+avar+"_both"), **opt)

    # for dist in distances:
    #     for avar in avars:
    #         opt = dict(legs=['τ #1', 'τ #2'], title=title)
    #         plot_hist_mpl([hacc[dist].gen_histos('first').project(avar), hacc[dist].gen_histos('second').project(avar)],
    #                       xlabel=xlabels[avar], out=savef("_"+avar+"_split"), **opt)

    # for avar in avars:
    #     opt = dict(legs=[''] if nd==1 else distances, title=title)
    #     plot_hist_mpl([hacc[k].hgendiffs.project(avar) for k in hacc.keys()],
    #                   xlabel='Arc-dist [cm]' if avar=='phi' else xlabels[avar],
    #                   out=savef("_"+avar+"_diff"), **opt)

    # bokeh
    output_file(savef('_bokeh')+'.html')
    genboth_row = []
    for avar in avars:
        opt = dict(legs=[''] if nd==1 else distances)
        p = plot_bokeh([hacc[k].gen_histos('both').project(avar) for k in hacc.keys()],
                       title=title, xlabel=xlabels[avar], **opt)
        genboth_row.append(p)

    # genfirst_row = []
    # for avar in avars:
    #     opt = dict(legs=[''] if nd==1 else distances)
    #     p = plot_bokeh([hacc[k].gen_histos('first').project(avar) for k in hacc.keys()],
    #                    title=title, xlabel=xlabels[avar], **opt)
    #     genfirst_row.append(p)

    # gensecond_row = []
    # for avar in avars:
    #     opt = dict(legs=[''] if nd==1 else distances)
    #     p = plot_bokeh([hacc[k].gen_histos('second').project(avar) for k in hacc.keys()],
    #                    title=title, xlabel=xlabels[avar], **opt)
    #     gensecond_row.append(p)

    if nd == 1:
        gensplit_row = []
        for avar in avars:
            opt = dict(legs=['τ #1', 'τ #2'])
            p = plot_bokeh([hacc[distances[0]].gen_histos('first').project(avar), hacc[distances[0]].gen_histos('second').project(avar)],
                           title=title, xlabel=xlabels[avar], **opt)
            gensplit_row.append(p)

    gendiff_row = []
    for avar in avars:
        opt = dict(legs=[''] if nd==1 else distances, density=True)
        p = plot_bokeh([hacc[k].hgendiffs.project(avar) for k in hacc.keys()],
                       title=title, xlabel=xlabels_diff[avar], **opt)
        gendiff_row.append(p)

    if nd > 1:
        lay = layout([genboth_row, gendiff_row])
    else:
        lay = layout([genboth_row, gensplit_row, gendiff_row])
    save(lay)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data exploration.')
    parser.add_argument('--dist', default=['10'], nargs='+',
                        help='Arc-distances in centimeters along phi between the two guns.')
    parser.add_argument('--tag', default='default',
                        help='Tag to store and load the histograms. Skips histogram production. Useful when only plot tweaks are necessary.')
    FLAGS = parser.parse_args()

    explore(FLAGS)
