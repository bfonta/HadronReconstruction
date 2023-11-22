
def histedges_equalN(x, nbins):
    """
    Define bin boundaries with the same numbers of events.
    `x` represents the array to be binned.
    """
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbins+1),
                     np.arange(npt),
                     np.sort(x))        

def nformat(s, inv=False):
    """Easy format."""
    if inv:
        return str(s).replace('p', '.').replace('m', '-')
    else:
        return str(s).replace('.', 'p').replace('-', 'm')
