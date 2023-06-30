# coding: utf-8

_all_ = [ 'explore' ]

import os
from os import path as op
import argparse
import uproot as up
import awkward as ak
import boost_histogram as bh
import itertools

from bokeh.palettes import Dark2_5 as palette

import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

def select_vars(tree):
    v = [x for x in tree.keys() if 'gen_' in x]
    v += ["event", "rechit_energy", "track_energy"]
    return set(v)

def plot_hist(arr, out, title, xlabel, legend, nbins=30, hmax=0., hmin=220., ylog=False):
    """Matplotlib histograms."""
    assert len(arr) == len(legend)
    colors = itertools.cycle(palette)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    if ylog:
        ax.set_yscale('log')
    
    for i, (ar,leg) in enumerate(zip(arr,legend)):
        hist = bh.Histogram(bh.axis.Regular(nbins, hmax, hmin))
        hist.fill(ar)
        plt.plot(hist.axes[0].centers, hist.view(),
                 "-o", color=next(colors), label=leg)

    if len(legend) > 1:
        plt.legend(loc="upper right")    
    #plt.tight_layout()

    hep.cms.text('Simulation')
    hep.cms.lumitext(title)
    for ext in ('.png', '.pdf'):
        plt.savefig(out + ext)

def explore():
    """Data exploration."""
    base = "/data_CMS/cms/alves"
    dist = "10cm"
    infile = op.join(base,
                     "DoublePion_0PU_10En200_30Jun_{}/step3/step3_80.root".format(dist))
    tree = "ana/hgc"
    outpath = "/eos/user/b/bfontana/www/HadronReco"
    
    with up.open(infile, array_cache="550 MB", num_workers=1) as f:
        tree = f[tree]
        allvars = select_vars(tree)
        data = tree.arrays(filter_name="/" + "|".join(allvars) + "/", library="ak")

    nevents = ak.count(data.event)
    arrays = {'en': [ak.flatten(data.gen_energy),
                     data.gen_energy[:,0],data.gen_energy[:,1]],
              'eta': [ak.flatten(data.gen_eta),
                     data.gen_eta[:,0],data.gen_eta[:,1]],
              'phi': [ak.flatten(data.gen_phi),
                     data.gen_phi[:,0],data.gen_phi[:,1]],
              'pt': [ak.flatten(data.gen_pt),
                     data.gen_pt[:,0],data.gen_pt[:,1]],
              'rh_en': [ak.flatten(data.rechit_energy),
                     data.rechit_energy[:,0],data.rechit_energy[:,1]],
              # 'tk_en': [ak.flatten(data.track_energy),
              #        data.track_energy[:,0],data.track_energy[:,1]],
              }
    title = "Double π ({}), {} events".format(dist, nevents)

    # opt = dict(legend=[''], title=title)
    # plot_hist([arrays["en"][0]], xlabel="Energy [GeV]",
    #           out=op.join(outpath, dist + "_en_both"),
    #           nbins=30, hmin=0, hmax=221., **opt)
    # plot_hist([arrays["eta"][0]], xlabel="η",
    #           out=op.join(outpath, dist + "_eta_both"),
    #           nbins=30, hmin=0, hmax=3.1, **opt)
    # plot_hist([arrays["phi"][0]], xlabel="ϕ",
    #           out=op.join(outpath, dist + "_phi_both"),
    #           nbins=30, hmin=-3.2, hmax=3.2, **opt)
    
    opt = dict(legend=['τ #1', 'τ #2'], title=title)
    plot_hist(arrays["en"][1:], xlabel="Energy [GeV]",
              out=op.join(outpath, dist + "_en_join"),
              nbins=30, hmin=0, hmax=221., **opt)
    plot_hist(arrays["eta"][1:], xlabel="η",
              out=op.join(outpath, dist + "_eta_join"),
              nbins=30, hmin=0, hmax=3.1, **opt)
    plot_hist(arrays["phi"][1:], xlabel="ϕ",
              out=op.join(outpath, dist + "_phi_join"),
              nbins=30, hmin=-3.2, hmax=3.2, **opt)
    plot_hist(arrays["pt"][1:], xlabel=r"$p_{T}$" + " [GeV]",
              out=op.join(outpath, dist + "_pt_join"),
              nbins=30, hmin=0, hmax=150, **opt)
    plot_hist(arrays["rh_en"][1:], xlabel=r"RecHit Energy [GeV]",
              out=op.join(outpath, dist + "_rh_en_join"),
              nbins=30, hmin=0, hmax=1.5, ylog=True, **opt)
    # plot_hist(arrays["tk_en"][1:], xlabel=r"Trackster Energy [GeV]",
    #           out=op.join(outpath, dist + "_tk_en_join"),
    #           nbins=30, hmin=0, hmax=5., ylog=True, **opt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data exploration.')
    FLAGS = parser.parse_args()

    explore()
