from matplotlib.pylab import *

def plot(mfcc):

    fig, (ax1) = subplots(1)
    fig.subplots_adjust(hspace=.5)

    ax1.imshow(mfcc.T, interpolation='nearest', origin='lower', aspect='auto')
    show()
