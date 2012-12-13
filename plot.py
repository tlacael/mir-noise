from matplotlib.pylab import *

def plot(mfcc, centroids):

    fig, (ax1, ax2) = subplots(2)
    fig.subplots_adjust(hspace=.5)

    ax1.imshow(mfcc.T, interpolation='nearest', origin='lower', aspect='auto')
    
    ax2.imshow(centroids.T, interpolation='nearest', origin='lower', aspect='auto')
    show()
