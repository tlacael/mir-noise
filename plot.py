from matplotlib.pylab import *

def plot(mfcc, centroids, classes):

    fig, (ax1, ax2, ax3, ax4) = subplots(4)
    fig.subplots_adjust(hspace=.5)

    print "plot MFCC"
    ax1.imshow(mfcc.T, interpolation='nearest', origin='lower', aspect='auto')

    print "Plot classes"
    ax2.plot(classes, 'x')
    ax2.set_xlim(0, len(classes))
    
    print "plot centroids"
    ax3.imshow(centroids.T, interpolation='nearest', origin='lower', aspect='auto')

    print "plot class histrogram"
    ax4.hist(classes, classes.max())
    
    ax4.set_xlim(0, classes.max())

    show()
