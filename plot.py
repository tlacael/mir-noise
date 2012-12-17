from matplotlib.pylab import *
from mirlib import mir_utils



def plotTimeline(mfcc, eventIndecies, centroids, classes):

    fig, (ax1, ax2, ax3, ax4) = subplots(4)
    fig.subplots_adjust(hspace=.5)

    print "plot MFCC"
    ax1.imshow(mfcc.T, interpolation='nearest', origin='lower', aspect='auto')


    print "Plot classes"

    eventIndYs = ones(len(eventIndecies)) * mfcc.shape[1]
    ax1.stem(eventIndecies, eventIndYs, markerfmt='k.')
    ax1.set_xlim(0, mfcc.shape[0])
    ax1.set_ylim(0, mfcc.shape[1])


    ax2.plot(classes, 'x')
    eventIndYs = ones(len(eventIndecies)) * classes.max() + 1
    ax2.stem(eventIndecies, eventIndYs, markerfmt='k.')
    ax2.set_xlim(0, len(classes))
    
    print "plot centroids"
    ax3.imshow(centroids.T, interpolation='nearest', origin='lower', aspect='auto')


    print "plot class histrogram"

    
    ax4.set_xlim(0, classes.max())

    class_id = arange(len(centroids))
    class_count = []
    for i in class_id:
        class_count.append( len((classes==i).nonzero()[0]) )
    ax4.bar(class_id - .45, class_count, color='r', width=.9)
    ax4.set_xlim(-.5, len(centroids) - .5)


    show()
    
def plot(mfcc, sones, eventIndecies, centroids, classes):

    fig, (ax1, ax2, ax3, ax4) = subplots(4)
    fig.subplots_adjust(hspace=.5)

    ax1.imshow(mfcc.T, interpolation='nearest', origin='lower', aspect='auto')
    ax1.set_title("Event Segment MFCCs")
    ax1.title.set_size("small")

    eventIndYs = ones(len(eventIndecies)) * mfcc.shape[1]
    ax1.stem(eventIndecies, eventIndYs, markerfmt='k.')
    ax1.set_xlim(0, mfcc.shape[0])
    ax1.set_ylim(0, mfcc.shape[1])


    ax2.plot(classes, 'x')
    eventIndYs = ones(len(eventIndecies)) * classes.max() + 1
    ax2.stem(eventIndecies, eventIndYs, markerfmt='k.')
    ax2.set_xlim(0, len(classes))
    ax2.set_title("Segment MFCC Class Labels")
    ax2.title.set_size("small")
    
    ax3.imshow(centroids.T, interpolation='nearest', origin='lower', aspect='auto')
    ax3.set_title("Centroids")
    ax3.title.set_size("small")


    class_id = arange(len(centroids))
    class_count = []
    for i in class_id:
        class_count.append( len((classes==i).nonzero()[0]) )
    ax4.bar(class_id - .45, class_count, color='r', width=.9)
    ax4.set_xlim(-.5, len(centroids) - .5)
    ax4.set_title("Centroid Histogram")
    ax4.title.set_size("small")

    fig, (ax1) = subplots(1)
    fig.subplots_adjust(hspace=.5)

    inter_class_dist_matrix = mir_utils.GetSquareDistanceMatrix(centroids)
    ax1.imshow(inter_class_dist_matrix, interpolation='nearest', origin='lower', aspect='auto')
    ax1.set_title("Centroid Distance Matrix")
    ax1.set_xlabel("Seconds")

    fig, (ax1) = subplots(1)
    ax1.plot(sones)
    ax1.set_title("Segment Sones")

    show()
