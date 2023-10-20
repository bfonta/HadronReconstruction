# Coding: utf-8

_all_ = [ 'analyse_single_gun' ]

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

class ScanParameters:
    def __setitem__(self, key, value):
        """Support item assignment"""
        setattr(self, key, value)

    def __getitem__(self, key):
        """Support item assignment"""
        return getattr(self, key)

    def __init__(self, text_format=True, **kwargs):
        """Initialize parameters to be scanned."""
        
        defaults = {
            'critical_density'         : ('cdens', [0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.7, 2.0, 2.5]),
            'critical_etaphi_distance' : ('cdist', [0.01, 0.02, 0.025, 0.03, 0.05]),
            'kernel_density_factor'    : ('kdens', [0.2])
        }

        # sets anything provided by the user
        for key, val in kwargs.items():
            self[key] = val

        # defaults are used if not provided by the user
        for par in defaults.keys():
            if par not in kwargs.keys():
                self[par] = defaults[par][1]

        if text_format:
            self._set_string_format(defaults)

        # define aliases for user convenience
        for key, val in defaults.items():
            self[val[0]] = self[key]
        
    def _set_string_format(self, defaults):
        for key in defaults:
            self[key] = [str(x).replace('.', 'p').replace('-', 'm') for x in self[key]]
            
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
    def __init__(self, tree, infiles, tag, mode='light'):
        assert mode in ('light', 'standard', 'intensive')
        self.nevents = 0
        self.nbins = 60
        self.mode = mode

        _types = ['hgen', 'htrackster', 'hntrackster', 'hntrackster_2d']
        if self.mode != 'light':
            _types += ['hfrac_trks_mult', 'hfrac_trks_sel', 'hfrac_em_had', 'hresp',
                       'bad_lcs', 'good_lcs']
        _extratypes = ['hfrac_trks_ceh']
        self.types = _types + _extratypes if self.mode == "intensive" else _types
        
        self.pickle_ext = ".pkl"
        self.adir = "histos_" + tag

        self.last_cee_layer = 26
        self.ceh_thresh = 0.90

        did_intensive_run = op.isdir(self.adir) and ( (len(os.listdir(self.adir)) == len(self.types) and self.mode == "intensive") or
                                                      (len(os.listdir(self.adir)) == len(_types) and not self.mode == "intensive") )
        if did_intensive_run:
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

        def frac_trks_multiplicity(n, sel):
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

        if self.mode == "intensive":
            self.fill_frac_trks_ceh_energy(data)
            self.fill_lone_layer_clusters(data)

        if self.mode != "light":
            for k in self.hfrac_trks_mult.keys():
                self.hfrac_trks_mult[k].fill(frac = frac_trks_multiplicity(k, sel=False))
            for k in self.hfrac_trks_sel.keys():
                self.hfrac_trks_sel[k].fill(frac = frac_trks_multiplicity(k, sel=True))

        def frac_ceh(thresh):
            """Computes several ratios related to energy deposited in CEE and CEH."""
            num_ceh = ak.sum(data.cluster2d_energy[data.cluster2d_layer > self.last_cee_layer], axis=1)
            num_cee = ak.sum(data.cluster2d_energy[data.cluster2d_layer <= self.last_cee_layer], axis=1)
            den = ak.sum(data.cluster2d_energy, axis=1)
            den_gen = ak.ravel(data.gunparticle_energy)
            frac_ceh = num_ceh/den
            frac_cee = num_cee/den
            missing = (den_gen - num_cee - num_ceh) / den_gen
            
            num_resp_ceh = ak.sum(data.cluster2d_energy[frac_ceh>thresh], axis=1)
            den_resp_ceh = ak.ravel(data.gunparticle_energy[frac_ceh>thresh])
            
            num_resp_cee = ak.sum(data.cluster2d_energy[frac_cee<=thresh], axis=1)
            den_resp_cee = ak.ravel(data.gunparticle_energy[frac_cee<=thresh])

            return (frac_ceh, frac_cee, num_ceh/den_gen, num_cee/den_gen, missing,
                    (den/den_gen)-1, (num_resp_ceh/den_resp_ceh)-1, (num_resp_cee/den_resp_cee)-1)

        if self.mode != "light":
            fceh = frac_ceh(thresh=self.ceh_thresh)
            self.hfrac_em_had.fill(frac_ceh=fceh[0], frac_cee=fceh[1],
                                   frac_ceh_gen=fceh[2], frac_cee_gen=fceh[3], frac_miss_gen=fceh[4])
            for ik,k in enumerate(self.hresp.keys()):
                self.hresp[k].fill(resp=fceh[ik+5])
            
    def _load(self):
        for t in self.types:
            with open(op.join(self.adir, t + self.pickle_ext), 'rb') as f:
                setattr(self, t, pickle.load(f))

        self.nevents = int(sum(self.hgen.project('en').counts()))

    def fill_frac_trks_ceh_energy(self, data):
        info = lambda evid, trkid, avar : getattr(data, 'cluster2d_' + avar)[evid, data.multiclus_cluster2d[evid][trkid]]

        nevents = len(ak.count(data.multiclus_eta, axis=1))
        for evid in range(nevents):
            ntracksters = ak.count(data.multiclus_eta, axis=1)[evid]

            for trkid in range(ntracksters):
                lc_l = info(evid, trkid, 'layer')
                lc_e = info(evid, trkid, 'energy')

                if self.mode == "intensive":
                    en_ceh = ak.sum(lc_e[lc_l > self.last_cee_layer])
                    en_cee = ak.sum(lc_e[lc_l <= self.last_cee_layer])
                    f = en_ceh / (en_ceh + en_cee)
                    f_gen = en_ceh / ak.flatten(data.gunparticle_energy)[evid][0]

                    self.hfrac_trks_ceh['full_en'].fill(frac=f, frac_gen=f_gen)
                    if en_ceh > 0 and en_cee > 0:
                        self.hfrac_trks_ceh['split_en'].fill(frac=f, frac_gen=f_gen)

                        l_ceh = len(np.unique(lc_l[lc_l > self.last_cee_layer]))
                        l_cee = len(np.unique(ak.sum(lc_l[lc_l <= self.last_cee_layer])))
                        f = l_ceh / (l_ceh + l_cee)
                
                        self.hfrac_trks_ceh['full_layer'].fill(frac=f, frac_gen=f)
                        if l_ceh > 0 and l_cee > 0:
                            self.hfrac_trks_ceh['split_layer'].fill(frac=f, frac_gen=f)

    def fill_lone_layer_clusters(self, data):
        """Study LCs whcih were not associated to a trackster."""
        info = lambda evid, mask, avar : data['cluster2d_' + avar][evid, mask]

        nevents = len(ak.count(data.multiclus_eta, axis=1))
        for evid in range(nevents):
            good_lc_idx = ak.ravel(data.multiclus_cluster2d[evid])
            mask = np.ones(len(data.cluster2d_energy[evid]), bool)
            mask[good_lc_idx] = 0
            mask_inv = np.zeros(len(data.cluster2d_energy[evid]), bool)
            mask_inv[good_lc_idx] = 1
            
            bad_lc_en = info(evid, mask, 'energy')
            bad_lc_eta = info(evid, mask, 'eta')
            bad_lc_phi = info(evid, mask, 'phi')
            bad_lc_pt = info(evid, mask, 'pt')
            for lcid in range(len(bad_lc_en)):
                self.bad_lcs.fill(en=bad_lc_en[lcid],
                                  eta=bad_lc_eta[lcid],
                                  phi=bad_lc_phi[lcid],
                                  pt=bad_lc_pt[lcid])
            
            good_lc_en = info(evid, mask_inv, 'energy')
            good_lc_eta = info(evid, mask_inv, 'eta')
            good_lc_phi = info(evid, mask_inv, 'phi')
            good_lc_pt = info(evid, mask_inv, 'pt')
            for lcid in range(len(good_lc_en)):
                self.good_lcs.fill(en=good_lc_en[lcid],
                                   eta=good_lc_eta[lcid],
                                   phi=good_lc_phi[lcid],
                                   pt=good_lc_pt[lcid])


    def _save(self, tree, infiles):
        ranges = {'en': (5., 205.),    'eta': (1.55, 2.85),    'phi': (-3.25, 3.25),    'pt': (0, 75),
                  'lc_en': (0., 20.), 'lc_eta': (1.55, 2.85), 'lc_phi': (-3.25, 3.25), 'lc_pt': (0, 15)}
        
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

        if self.mode != 'light':

            nb = 80
            if self.mode == "intensive":
                self.hfrac_trks_ceh = {k:hist.Hist(
                    hist.axis.Regular(nb, -0.01, 1.01, name="frac"),
                    hist.axis.Regular(nb, -0.01, 1.01, name="frac_gen"),
                ) for k in ('full_en', 'split_en', 'full_layer', 'split_layer')}

            frac_trks_keys = [1, 2, 5, 10]
            self.hfrac_trks_mult = {k:hist.Hist(hist.axis.Regular(nb, 0.2, 1.02, name="frac")) for k in frac_trks_keys}
            self.hfrac_trks_sel = {k:hist.Hist(hist.axis.Regular(nb, 0.2, 1.02, name="frac")) for k in frac_trks_keys}
            
            self.hfrac_em_had = hist.Hist(
                hist.axis.Regular(self.nbins, -0.01,  1.01, name="frac_ceh"),
                hist.axis.Regular(self.nbins, -0.01,  1.01, name="frac_cee"),
                hist.axis.Regular(self.nbins, -0.01,  1.01, name="frac_ceh_gen"),
                hist.axis.Regular(self.nbins, -0.01,  1.01, name="frac_cee_gen"),
                hist.axis.Regular(self.nbins, -0.01,  1.01, name="frac_miss_gen"),
            )
            self.hresp = {k:hist.Hist(hist.axis.Regular(self.nbins, -1, 0.5, name="resp")) for k in ('full', 'cee', 'ceh')}
     
            self.bad_lcs = hist.Hist(
                hist.axis.Regular(self.nbins, ranges['lc_en'][0],  ranges['lc_en'][1],  name="en"),
                hist.axis.Regular(self.nbins, ranges['lc_eta'][0], ranges['lc_eta'][1], name="eta"),
                hist.axis.Regular(self.nbins, ranges['lc_phi'][0], ranges['lc_phi'][1], name="phi"),
                hist.axis.Regular(self.nbins, ranges['lc_pt'][0],  ranges['lc_pt'][1],  name="pt"),
            )
            self.good_lcs = hist.Hist(
                hist.axis.Regular(self.nbins, ranges['lc_en'][0],  ranges['lc_en'][1],  name="en"),
                hist.axis.Regular(self.nbins, ranges['lc_eta'][0], ranges['lc_eta'][1], name="eta"),
                hist.axis.Regular(self.nbins, ranges['lc_phi'][0], ranges['lc_phi'][1], name="phi"),
                hist.axis.Regular(self.nbins, ranges['lc_pt'][0],  ranges['lc_pt'][1],  name="pt"),
            )
     
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
               mode='step', xerr=False, yerr=False, text=False, zlog=''):
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
            xtextbins = np.array([x-w/3 for x,w in zip(h.axes[0].centers,wbins) for _ in range(len(h.axes[1].centers))])
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
        if len(legs)>0 and legs[0]!='':
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
            color_bar.title=zlog
            color_bar.title_text_align="center"
            color_bar.title_text_font_size = '10pt'
            p.add_layout(color_bar, 'right')

            if text:
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

def plot_mpl(hists, title, xlabel, ylabel, legs, mode='point', xerr=True, yerr=True):
    """
    Matplotlib plots. If mode=='2d', `x` and `y` are bin edges.
    """
    if not isinstance(hists, (tuple,list)):
        hists = [hists]
    if not isinstance(legs, (tuple,list)):
        legs = [legs]
    assert len(hists) == len(legs)

    colors = it.cycle(palette)

    wsize, hsize = 16, 16

    fig = plt.figure(figsize=(wsize, hsize),)
    ax = plt.subplot(111)
    ax.title.set_size(100)

    #ax.axhline(y=0., color='gray', linestyle='dashed')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for h,l in zip(hists, legs):
        #ax.plot(h.axes[0].centers, h.values(), "-o", color=next(colors), label=l)

        x_lo, x_hi = [], []
        y_lo, y_hi = [], []
        if xerr:
            for px, py, err in zip(h.axes[0].centers, h.values(), (h.axes[0].edges[1:] - h.axes[0].edges[:-1])/2):
                x_lo.append(err)
                x_hi.append(err)
        if yerr:
            for px, py, err in zip(h.axes[0].centers, h.values(), np.sqrt(h.variances())/2):
                y_lo.append(err)
                y_hi.append(err)

        ax.errorbar(h.axes[0].centers, h.values(), xerr=[x_lo, x_hi], yerr=[y_lo, y_hi],
                     fmt="-o", color=next(colors), label=l)

                
    hep.cms.text('Preliminary', fontsize=wsize*2.5)
    hep.cms.lumitext(title, fontsize=wsize*2.5) # r"138 $fb^{-1}$ (13 TeV)"

    for ext in ('.png', '.pdf'):
        name = "test" + '_' + mode + ext
        print('Stored in {}'.format(name))
        plt.savefig(name, dpi=600)
    plt.close()

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

def build_dashboard(infiles, labels, tree, args):
    avars = list(labels.keys())
    xlabels = [k[0] for k,v in labels.items()]
    xlabels_diff = [k[1] for k,v in labels.items()]
    
    for inf,_label in zip(infiles,('clue3d',)):
        label = "stats_single_gun_" + _label + '_' + args.tag
        hacc = AccumulateHistos(tree, inf, label, args.mode)
        title = "Single π, {} events".format(hacc.nevents)
        opt = dict(title=title)

        # bokeh
        output_file(op.join(args.outpath, label)+'.html')
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
        opt = dict(legs=['Missed energy fraction with respect to Gen. Energy',],
                   legloc="top_right", title=title)
        p = plot_bokeh([hacc.hfrac_em_had.project("frac_miss_gen"),],
                       density=True, xlabel="Fraction", **opt)
        fracs_row.append(p)
        opt = dict(legs=[''], ylabel="Missed energy fraction", xlabel="Hadronic energy fraction",
                   zlog='Number of events')
        p = plot_bokeh(hacc.hfrac_em_had.project("frac_ceh", "frac_miss_gen")[::3j,::3j],
                       title=title, mode='2d', **opt)
        fracs_row.append(p)
        opt = dict(legs=['Full', 'CEH  > {}'.format(hacc.ceh_thresh), 'CEH < {}'.format(hacc.ceh_thresh)],
                   legloc="top_right", title=title)
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
        p = plot_bokeh([x.project("frac") for x in hacc.hfrac_trks_mult.values()], ylog=True,
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

        if args.mode == "intensive":
            ntrackster_frac_row = []
            opt = dict(legs=['Trackster energy fraction', 'Trackster energy fraction w/ respect to Gen'],
                       legloc="bottom_left", title=title, ylog=True,)
            p = plot_bokeh([hacc.hfrac_trks_ceh['full_en'].project(x) for x in ("frac", "frac_gen")], 
                           xlabel="Fraction of the energy left by tracksters in the CEH", **opt)
            ntrackster_frac_row.append(p)
            opt = dict(legs=['Trackster energy fraction', 'Trackster energy fraction w/ respect to Gen'], legloc="top_right", title=title)
            p = plot_bokeh([hacc.hfrac_trks_ceh['split_en'].project(x) for x in ("frac", "frac_gen")],
                           xlabel="Fraction of the energy left by split tracksters in the CEH", **opt)
            ntrackster_frac_row.append(p)
            opt = dict(legs=['Trackster layer fraction'], legloc="top_left", title=title)
            p = plot_bokeh([hacc.hfrac_trks_ceh['full_layer'].project("frac")], ylog=True,
                           xlabel="Fraction of the layers by tracksters in the CEH", **opt)
            ntrackster_frac_row.append(p)
            p = plot_bokeh([hacc.hfrac_trks_ceh['split_layer'].project("frac")], ylog=True,
                           xlabel="Fraction of the layers by split tracksters in the CEH", **opt)
            ntrackster_frac_row.append(p)
        
            lcs_row = []
            opt = dict(legs=['Bad LCs', 'Good LCs'], legloc="bottom_left", title=title, ylog=False,)
            for avar in avars:
                p = plot_bokeh([hacc.bad_lcs.project(avar), hacc.good_lcs.project(avar)], xlabel=xlabels[avar], **opt)
                lcs_row.append(p)

        ntrackster_2d_split_row = []
        opt = dict(legs=[''])
        for avar in avars:
            p = plot_bokeh(hacc.hntrackster_2d.project(avar, "n").profile("n"),
                           title=title, ylabel="# Tracksters", xlabel=xlabels[avar],
                           mode='point', xerr=True, yerr=True, **opt)

            ntrackster_2d_split_row.append(p)

        ntrackster_2d_row = []
        opt = dict(legs=[''], text=True, zlog="Average number of tracksters")
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

        lay = [gen_row, trackster_row, fracs_row,
               ntrackster_2d_split_row, ntrackster_2d_row, ntrackster_row]
        if args.mode == "intensive":
            lay.extend([ntrackster_frac_row, lcs_row])
        lay = layout(lay)
        save(lay)

def run_scan(infiles, tags, labels, tree, args):
    avars = list(labels.keys())

    # hacc = {tag: AccumulateHistos(tree, inf, tag, args.intensive)
    #         for inf,tag in zip(infiles,tags)}

    hists = []
    for avar in avars:
        for inf, tag in zip(infiles, tags):
            hacc = AccumulateHistos(tree, inf, tag, args.mode)
            hists.append(hacc.hntrackster_2d.project(avar, "n").profile("n"))
            break
    
        plot_mpl(hists, title="Grid Scan", ylabel="# Tracksters", xlabel=labels[avar][0],
                 mode='point', legs=[''])
        break

def analyse_single_gun(args):
    """Data analysis."""
    base = op.join("/data_CMS/cms/alves", args.dataset, "step3")
    tree = "ana/hgc"

    labels = {'en': ("Energy [GeV]", "ΔE [GeV]"),
              'eta': ("|η|", "Δη"),
              'phi': ("ϕ", "Δϕ"),
              'pt': ('pT [GeV]', 'ΔpT [GeV]')}

    if args.dashboard:
        infiles = (op.join(base, "step3_1.root"),)
        build_dashboard(infiles, labels, tree, args)
    else:
        infiles, tags, pars = [], [], ScanParameters()
        for i in it.product(pars.cdens, pars.cdist, pars.kdens):
            tag = "22_CDENS{}_CDIST{}_KDENS{}_V1".format(*i)
            tags.append(tag)
            infiles.append(op.join(base, "step3_" + tag + ".root"))
        run_scan(infiles, tags, labels, tree, args)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data analysis of a scan of single pion gun samples.')
    parser.add_argument('--tag', default='_default',
                        help='Tag to store and load the histograms. Skips histogram production. Useful when only plot tweaks are necessary.')
    parser.add_argument('--dataset', default='SinglePion_0PU_10En200_11Jul', help='Dataset to use.')
    parser.add_argument('--mode', default="light", choices=("light", "standard", "intensive"), help='Run more time consuming trackster-related calculations.')
    parser.add_argument('--dashboard', action="store_true", help='Run the full dashboard over one default sample.')
    parser.add_argument('--outpath', default="/eos/home-b/bfontana/www/HadronReco", help='Output directory.')
    FLAGS = parser.parse_args()
    
    analyse_single_gun(FLAGS)
