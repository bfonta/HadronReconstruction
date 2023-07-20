# Coding: utf-8

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

from bokeh import models as bm
from bokeh.layouts import layout
from bokeh.palettes import Colorblind as palette, OrRd9 as palette_2d
palette = palette[8]
from bokeh.plotting import figure, save, output_file
from bokeh.transform import linear_cmap

import matplotlib; import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

def histedges_equalN(x, nbins):
    """
    Define bin boundaries with the same numbers of events.
    `x` represents the array to be binned.
    """
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbins+1),
                     np.arange(npt),
                     np.sort(x))        
        
class AccumulateHistos():
    def __init__(self, tree, infiles, tag):
        self.nevents = 0
        self.nbins = 50
        self.types = ('hgen', 'htrackster', 'hntrackster', 'hntrackster_2d',
                      'hfrac_trks', 'hfrac_trks_sel',
                      'hfrac_em_had', 'hresp')
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
        self.hgen.fill(en  = ak.ravel(data.gunparticle_energy),
                       eta = ak.ravel(data.gunparticle_eta),
                       phi = ak.ravel(data.gunparticle_phi),
                       pt  = ak.ravel(data.gunparticle_pt))

        self.htrackster.fill(en  = ak.ravel(data.multiclus_energy),
                             eta = ak.ravel(data.multiclus_eta),
                             phi = ak.ravel(data.multiclus_phi),
                             pt  = ak.ravel(data.multiclus_energy))

        self.hntrackster.fill(n = ak.count(data.multiclus_energy, axis=1),)

        self.hntrackster_2d.fill(n   = ak.count(data.multiclus_energy, axis=1),
                                 en  = ak.ravel(data.gunparticle_energy),
                                 eta = ak.ravel(data.gunparticle_eta),
                                 phi = ak.ravel(data.gunparticle_phi),
                                 pt  = ak.ravel(data.gunparticle_pt))

        def frac_trks(n, sel):
            """
            Selects n highest-energy tracksters for each event.
            If `sel==True`, only considers events with >= n tracksters.
            The latter avoids a peak at fraction=1 for events with # tracksters < n.
            """
            en_top_n = ak.sort(data.multiclus_energy)[:,-n:]
            if sel: # performs selection
                s = ak.count(data.multiclus_energy, axis=1) > n
            else: # does not mask
                s = np.full_like(en_top_n, True, dtype=bool)
            return ak.sum(en_top_n[s], axis=1) / ak.sum(data.multiclus_energy[s], axis=1)
                           
        for k in self.hfrac_trks.keys():
            self.hfrac_trks[k].fill(frac = frac_trks(k, sel=False))
        for k in self.hfrac_trks_sel.keys():
            self.hfrac_trks_sel[k].fill(frac = frac_trks(k, sel=True))

        def frac_ceh():
            """Computes several ratios related to energy deposited in CEE and CEH."""
            num_ceh = ak.sum(data.cluster2d_energy[data.cluster2d_layer > 26], axis=1)
            num_cee = ak.sum(data.cluster2d_energy[data.cluster2d_layer <= 26], axis=1)
            den = ak.sum(data.cluster2d_energy, axis=1)
            den_gen = ak.ravel(data.gunparticle_energy)
            frac_ceh = num_ceh/den
            frac_cee = num_cee/den
            thresh = 0.91

            num_resp_ceh = ak.sum(data.cluster2d_energy[frac_ceh>thresh], axis=1)
            den_resp_ceh = ak.ravel(data.gunparticle_energy[frac_ceh>thresh])
            
            num_resp_cee = ak.sum(data.cluster2d_energy[frac_cee<thresh], axis=1)
            den_resp_cee = ak.ravel(data.gunparticle_energy[frac_cee<thresh])
            
            return (frac_ceh, frac_cee, num_ceh/den_gen, num_cee/den_gen,
                    (den/den_gen)-1, (num_resp_ceh/den_resp_ceh)-1, (num_resp_cee/den_resp_cee)-1)

        fceh = frac_ceh()
        self.hfrac_em_had.fill(frac_ceh=fceh[0], frac_cee=fceh[1], frac_ceh_gen=fceh[2], frac_cee_gen=fceh[3])
        for ik,k in enumerate(self.hresp.keys()):
            self.hresp[k].fill(resp=fceh[ik+4])
            
    def _load(self):
        for t in self.types:
            with open(op.join(self.adir, t + self.pickle_ext), 'rb') as f:
                setattr(self, t, pickle.load(f))

        self.nevents = int(sum(self.hgen.project('en').counts()))

    def _save(self, tree, infiles):
        ranges = {'en': (5., 205.), 'eta': (1.55, 2.85), 'phi': (-3.25, 3.25), 'pt': (0, 75)}
        
        # gen variables
        self.hgen = hist.Hist(
            hist.axis.Regular(self.nbins, ranges['en'][0],  ranges['en'][1],  name="en"),
            hist.axis.Regular(self.nbins, ranges['eta'][0], ranges['eta'][1], name="eta"),
            hist.axis.Regular(self.nbins, ranges['phi'][0], ranges['phi'][1], name="phi"),
            hist.axis.Regular(self.nbins, ranges['pt'][0],  ranges['pt'][1],  name="pt"),
        )

        self.htrackster = hist.Hist(
            hist.axis.Regular(self.nbins, 0,    20, name="en"),
            hist.axis.Regular(self.nbins, 1.5,  3.1,  name="eta"),
            hist.axis.Regular(self.nbins, -3.2, 3.2,  name="phi"),
            hist.axis.Regular(self.nbins, 0,    18,  name="pt"),
        )

        nmax = 25
        nn = nmax+1
        self.hntrackster = hist.Hist(hist.axis.Regular(nn, 0, nmax,  name="n"))

        nn2 = 10
        self.hntrackster_2d = hist.Hist(
            hist.axis.Regular(nn,  0, nmax,  name="n"),
            hist.axis.Regular(nn2, ranges['en'][0],  ranges['en'][1],  name="en"),
            hist.axis.Regular(nn2, ranges['eta'][0], ranges['eta'][1], name="eta"),
            hist.axis.Regular(nn2, ranges['phi'][0], ranges['phi'][1], name="phi"),
            hist.axis.Regular(nn2, ranges['pt'][0],  ranges['pt'][1],  name="pt"),
        )

        nb = 80
        frac_trks_keys = [1, 2, 5, 10]
        self.hfrac_trks = {k:hist.Hist(hist.axis.Regular(nb, 0.2, 1.02,  name="frac")) for k in frac_trks_keys}
        self.hfrac_trks_sel = {k:hist.Hist(hist.axis.Regular(nb, 0.2, 1.02,  name="frac")) for k in frac_trks_keys}
        
        self.hfrac_em_had = hist.Hist(
            hist.axis.Regular(self.nbins, 0.,  1., name="frac_ceh"),
            hist.axis.Regular(self.nbins, 0.,  1., name="frac_cee"),
            hist.axis.Regular(self.nbins, 0.,  1., name="frac_ceh_gen"),
            hist.axis.Regular(self.nbins, 0.,  1., name="frac_cee_gen"),
        )
        self.hresp = {k:hist.Hist(hist.axis.Regular(self.nbins, -1, 0.5, name="resp")) for k in ('full', 'cee', 'ceh')}
        
        allvars = self._select_vars()
        for batch in tqdm(up.iterate(infiles + ":" + tree, 
                                     step_size=10000, library='ak',
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
               ylabel=None, ylog=False, density=False, frac=False,
               mode='step', xerr=False, yerr=False):
    """Plot using the Bokeh package"""
    if ylog is True and density is True:
        raise RuntimeError('Currently ylog and density cannot be used simultaneously.')
    
    if not isinstance(hists, (tuple,list)):
        hists = [hists]
    if not isinstance(legs, (tuple,list)):
        legs = [legs]
    assert len(hists) == len(legs)
    assert not (density and frac)
    colors = it.cycle(palette)

    p = figure(height=400, width=600, background_fill_color="white", title=title,
               y_axis_type='log' if ylog else 'linear',
               tools="save,box_select,box_zoom,wheel_zoom,reset,undo,redo")
    for h,l in zip(hists, legs):
        if mode == '2d':
            xbins  = np.array([x for x in h.axes[0].centers for _ in range(len(h.axes[1].centers))])
            wbins  = np.array([x for x in (h.axes[0].edges[1:] - h.axes[0].edges[:-1]) for _ in range(len(h.axes[1].centers))])
            xtextbins  = np.array([x-w/3 for x,w in zip(h.axes[0].centers,wbins) for _ in range(len(h.axes[1].centers))])
            ybins  = ak.concatenate([h.axes[1].centers for _ in range(len(h.axes[0].centers))]).to_numpy()
            hbins  = ak.concatenate([(h.axes[1].edges[1:] - h.axes[1].edges[:-1]) for _ in range(len(h.axes[1].centers))]).to_numpy()
            vals   = np.nan_to_num(np.round(ak.ravel(h.values()).to_numpy(),2), nan=0.)
            cmap   = linear_cmap("vals", palette=palette_2d[::-1], low=min(vals), high=max(vals))
            source = bm.ColumnDataSource(data=dict(xbins=xbins, ybins=ybins, height=hbins, width=wbins, vals=vals, xtext=xtextbins))

        else:
            hcenters = h.axes[0].centers
            if density:
                hvals = h.density()
                herrup, herrdown = np.zeros_like(hcenters), np.zeros_like(hcenters)
            elif frac:
                hvals = h.values()/h.sum()
                herrup, herrdown = np.zeros_like(hcenters), np.zeros_like(hcenters)
            else:
                hvals = h.values()
                herrup, herrdown = h.variances()/2, h.variances()/2
            if ylog:
                hvals[hvals==0.] = 0.5 # avoid infinities
            source = bm.ColumnDataSource(data=dict(y=hvals, x=hcenters, xup=herrup, xdown=herrdown))
            
        glyph_opt = dict(y='y', x='x', source=source)
        if len(legs)>1:
            glyph_opt.update({'legend_label': l})

        if mode == 'step':
            p.step(**glyph_opt, mode='before', line_color=next(colors), line_width=3)

        elif mode == 'point':
            thisc = next(colors)
            p.circle(**glyph_opt, color=thisc, size=6)

            if xerr:
                x_err_x = []
                x_err_y = []
                for px, py, err in zip(h.axes[0].centers, h.values(), (h.axes[0].edges[1:] - h.axes[0].edges[:-1])/2):
                    x_err_x.append((px - err, px + err))
                    x_err_y.append((py, py))
                p.multi_line(x_err_x, x_err_y, color=thisc)

            if yerr:
                y_err_x = []
                y_err_y = []
                for px, py, err in zip(h.axes[0].centers, h.values(), np.sqrt(h.variances())/2):
                    y_err_x.append((px, px))
                    y_err_y.append((py - err, py + err))
                p.multi_line(y_err_x, y_err_y, color=thisc)

        elif mode == '2d':
            r = p.rect(x='xbins', y='ybins', width='width', height='height', source=source,
                       color=cmap, line_width=2, line_color='black')
            color_bar = r.construct_color_bar(padding=0, ticker=p.yaxis.ticker, formatter=p.yaxis.formatter)
            color_bar.title="Average number of tracksters"
            color_bar.title_text_align="center"
            color_bar.title_text_font_size = '10pt'
            p.add_layout(color_bar, 'right')

            text = bm.Text(x="xtext", y="ybins", text="vals", angle=0.0, text_color="black", text_font_size="9pt")
            p.add_glyph(source, text)

    p.output_backend = 'svg'
    p.toolbar.logo = None
    if len(legs)>1:
        p.legend.click_policy='hide'
        p.legend.location = legloc
        p.legend.label_text_font_size = '12pt'
    p.min_border_bottom = 10
    p.xaxis.visible = True
    p.title.align = "left"
    p.title.text_font_size = "15px"
    
    p.xgrid.grid_line_color = None
    #p.y_range.start = 0

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
        hists = [[h if h>0. else 0.1] for hist in hists for h in hist]
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
    infiles = (op.join(base, "SinglePion_0PU_10En200_11Jul/step3/step3_1.root"),
               #op.join(base, "SinglePion_0PU_10En200_30Jun/step3_linking/step3_*.root")
               )
    tags = ('clue3d',) #('clue3d', 'linking')
    for inf,tag in zip(infiles,tags):
        hacc = AccumulateHistos(tree, inf, tag="single_"+tag)    
        title = "Single π, {} events".format(hacc.nevents)
        opt = dict(title=title)

        avars = ('en', 'eta', 'phi', 'pt')
        xlabels      = {'en': "Energy [GeV]", 'eta': "|η|", 'phi': "ϕ", 'pt': 'pT [GeV]'}
        xlabels_diff = {'en': "ΔE [GeV]", 'eta': "Δη", 'phi': "Δϕ", 'pt': 'ΔpT [GeV]'}

        # matplotlib
        # for avar in avars:
        #     opt = dict(legs=[''] if nd==1 else distances, title=title)
        #     plot_hist_mpl(hacc.hgen.project(avar),
        #                   xlabel=xlabels[avar], out=savef("single_"+avar+"_both"), **opt)
    
        # bokeh
        output_file(savef('single_bokeh_'+tag)+'.html')
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

        fracs_row = []
        opt = dict(legs=['CEH En. Fraction', 'CEH En. Fraction with respect to Gen En.'],
                   legloc="top_left")
        p = plot_bokeh([hacc.hfrac_em_had.project("frac_ceh"), hacc.hfrac_em_had.project("frac_ceh_gen")],
                       title=title, density=True, xlabel="Fraction", **opt)
        fracs_row.append(p)
        opt = dict(legs=['CEE En. Fraction', 'CEE En. Fraction with respect to Gen En.'],
                   legloc="top_right", title=title)
        p = plot_bokeh([hacc.hfrac_em_had.project("frac_cee"), hacc.hfrac_em_had.project("frac_cee_gen")],
                       density=True, xlabel="Fraction", **opt)
        fracs_row.append(p)
        opt = dict(legs=['Full', 'CEH', 'CEE'], legloc="top_right", title=title)
        p = plot_bokeh([x.project("resp") for x in hacc.hresp.values()],
                       density=True, xlabel="Layer Clusters' Response", **opt)
        respmax = max(hacc.hresp['full'].project("resp").density())
        respmin = min(hacc.hresp['full'].project("resp").density())
        p.line(x=[0., 0.], y=[respmin,respmax], line_color='black', line_width=2,
               color='gray', line_dash='dashed')
        fracs_row.append(p)
            
        ntrackster_row = []
        opt = dict(legs=['highest-energy trackster', '2 highest-energy tracksters',
                         '5 highest-energy tracksters', '10 highest-energy tracksters'],
                   legloc="top_left",  title=title)
        p = plot_bokeh([x.project("frac") for x in hacc.hfrac_trks.values()], ylog=True,
                       xlabel="Fraction of the total trackster energy in an event", **opt)
        ntrackster_row.append(p)
        p = plot_bokeh([x.project("frac") for x in hacc.hfrac_trks_sel.values()], density=True,
                       xlabel="Fraction of the total trackster energy in an event after saturation removal", **opt)
        ntrackster_row.append(p)
        opt = dict(legs=[''])
        # p = plot_bokeh(hacc.hntrackster.project("n"),
        #                title=title, xlabel="# Tracksters", **opt)
        # ntrackster_row.append(p)
        p = plot_bokeh(hacc.hntrackster.project("n"),
                       title=title, xlabel="# Tracksters", ylabel="Fraction of events",
                       frac=True, **opt)
        ntrackster_row.append(p)

        ntrackster_2d_split_row = []
        opt = dict(legs=[''])
        for avar in avars:
            p = plot_bokeh(hacc.hntrackster_2d.project(avar, "n").profile("n"),
                           title=title, ylabel="# Tracksters", xlabel=xlabels[avar],
                           mode='point', xerr=True, yerr=True, **opt)

            ntrackster_2d_split_row.append(p)

        ntrackster_2d_row = []
        opt = dict(legs=[''])
        for ivar in ("en", "pt"):
            for jvar in ("eta", "phi"):
                p = plot_bokeh(hacc.hntrackster_2d.project(jvar, ivar, "n").profile("n"),
                               title=title, xlabel=xlabels[jvar], ylabel=xlabels[ivar],
                               mode='2d', **opt)
                ntrackster_2d_row.append(p)
        p = plot_bokeh(hacc.hntrackster_2d.project("eta", "phi", "n").profile("n"),
                       title=title, xlabel=xlabels["eta"], ylabel=xlabels["phi"],
                       mode='2d', **opt)
        ntrackster_2d_row.append(p)

        lay = layout([gen_row, trackster_row, fracs_row,
                      ntrackster_row, ntrackster_2d_split_row, ntrackster_2d_row])
        save(lay)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data exploration of single pion gun.')
    parser.add_argument('--tag', default='default',
                        help='Tag to store and load the histograms. Skips histogram production. Useful when only plot tweaks are necessary.')
    FLAGS = parser.parse_args()
    
    explore_single_gun(FLAGS)
