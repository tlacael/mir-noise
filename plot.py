from matplotlib.pylab import *

def plot(mfcc, eventIndecies, centroids, classes):

    fig, (ax1, ax2, ax3, ax4) = subplots(4)
    fig.subplots_adjust(hspace=.5)

    ax1.imshow(mfcc.T, interpolation='nearest', origin='lower', aspect='auto')

    eventIndYs = ones(len(eventIndecies)) * mfcc.shape[1]
    ax1.stem(eventIndecies, eventIndYs, markerfmt='k.')
    ax1.set_xlim(0, mfcc.shape[0])
    ax1.set_ylim(0, mfcc.shape[1])

    ax2.plot(classes, 'x')
    eventIndYs = ones(len(eventIndecies)) * classes.max() + 1
    ax2.stem(eventIndecies, eventIndYs, markerfmt='k.')
    ax2.set_xlim(0, len(classes))
    
    ax3.imshow(centroids.T, interpolation='nearest', origin='lower', aspect='auto')

    class_id = arange(len(centroids))
    class_count = []
    for i in class_id:
        class_count.append( len((classes==i).nonzero()[0]) )
    ax4.bar(class_id - .45, class_count, color='r', width=.9)
    ax4.set_xlim(-.5, len(centroids) - .5)

    show()
