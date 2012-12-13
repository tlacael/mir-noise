import time
import sys
import argparse
import unittest

import numpy as np
from numpy import vstack, array
from numpy.random import rand
from mirlib.feature_analysis import kmeans


class Test_k_means(unittest.TestCase):

    def setUp(self):
        pass

    def test_euclid_dist(self):
        # Test ndim == 1
        vec1 = array(0)
        vec2 = array(1)
        dist = kmeans.euclid_dist(vec1, vec2)
        self.assertEquals(dist, 1)

        # Test ndim == 2; a simple 3-4-5 triangle
        vec1 = array([[0, 0]])
        vec2 = array([[3, 4]])
        dist = kmeans.euclid_dist(vec1, vec2)
        self.assertEquals(dist, 5)

    def test_getnearestcentroids(self):
        # Four points, equally spaced in each quadrant
        data = array([ [2,2],
                       [-2,2],
                       [-2,-2],
                       [2,-2]])
        # Two centroids, one in I, one in III
        centroids = array([ [3, 3],
                          [-4, -4]])

        nearest = kmeans.GetNearestCentroids(data, centroids)
        desired_result = array( [0, 0, 1, 0] )

        self.assertTrue( (nearest == desired_result).all() )

def kmr_test_plot(data, k, end_thresh):
    from matplotlib.pylab import ion, figure, draw, ioff, show, plot, cla
    ion()
    fig = figure()
    ax = fig.add_subplot(111)
    ax.grid(True)

    # get k centroids
    kmr = kmeans.kmeans_runner(k, end_thresh)
    kmr.init_data(data)
    print kmr.centroids

    plot(data[:,0], data[:,1], 'o')

    i = 0
    while kmr.stop_flag is False:
        kmr.iterate()
        #print kmr.centroids, kmr.itr_count
        plot(kmr.centroids[:, 0], kmr.centroids[:, 1], 'sr')
        time.sleep(.2)
        draw()
        i += 1

    print "N Iterations: %d" % (i)
    plot(kmr.centroids[:, 0], kmr.centroids[:, 1], 'g^', linewidth=3)

    ioff()
    show()
    print kmr.itr_count, kmr.centroids

def test(data, k, end_thresh):
    centroids, nItr = kmeans.kmeans(data, k, end_thresh)
    print "N Iterations: %d" % (nItr)
    print "Means:", centroids

def onlineplot(data, k, end_thresh, on_denom):
    from matplotlib.pylab import ion, figure, draw, ioff, show, plot
    print "onlineplot", k, end_thresh, on_denom

    ion()
    fig = figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_ylim((-1.5, 1.5))
    ax.set_xlim((-1.5, 1.5))

    # get k centroids
    kmr = kmeans.kmeans_runner(k, end_thresh)

    cutoffIdx = np.ceil(len(data) / np.float(on_denom))
    np.random.shuffle(data)
    initialData = data[:cutoffIdx]
    remainingData = data[cutoffIdx:]
                        
    kmr.init_data(initialData)

    print "Initial:", kmr.centroids

    plot(initialData[:,0], initialData[:,1], 'o')

    for point in remainingData:
        kmr.iterate_online(point)
        
        plot(kmr.centroids[:, 0], kmr.centroids[:, 1], 'sr')
        plot(point[0], point[1], 'ob')
        #time.sleep(.)
        draw()

    plot(kmr.centroids[:, 0], kmr.centroids[:, 1], 'g^', linewidth=3)

    ioff()
    show()
    print "Final Counts:", kmr.k_counts
    print kmr.centroids

def onlinetest(data, k, end_thresh, on_denom):
    print "onlinetest", k, end_thresh, on_denom
    kmr = kmeans.kmeans_runner(k, end_thresh)

    cutoffIdx = np.ceil(len(data) / np.float(on_denom))
    np.random.shuffle(data)
    kmr.init_data(data[:(cutoffIdx)])

    print "Initial:", kmr.centroids

    for point in data[cutoffIdx:]:
        kmr.iterate_online(point)

    print "Final Counts:", kmr.k_counts
    print kmr.centroids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--k_classes", help="Number of test classes",
                        default=4, type=int)
    parser.add_argument("-t", "--end_thresh", help="Distance threshold to end",
                        default=0.01, type=float)

    # Define program modes
    subparsers = parser.add_subparsers(help="Program Mode Help", dest="command")
    parser_unittest = subparsers.add_parser("unittest", help="unit testing")

    parser_test = subparsers.add_parser("test", help="Run kmeans test")
    parser_test.set_defaults(func=test)

    parser_onlinetest = subparsers.add_parser("onlinetest", help="Run online kmeans test")
    parser_onlinetest.set_defaults(func=onlinetest)
    parser_onlinetest.add_argument("--dim_denom", help="Denominator of the fraction of the dataset that is to be used with each increment", default=6, type=int)

    parser_plot = subparsers.add_parser("plot", help="Run interactive plot")
    parser_plot.set_defaults(func=kmr_test_plot)

    parser_onlineplot = subparsers.add_parser("onlineplot", help="Run online kmeans test")
    parser_onlineplot.set_defaults(func=onlineplot)
    parser_onlineplot.add_argument("--dim_denom", help="Denominator of the fraction of the dataset that is to be used with each increment", default=6, type=int)
    
    args = parser.parse_args()
    
    print "mirlib k-means test"

    print "1. Create a test set"
    # data generation
    data = vstack((rand(150,2) + array([.5,.5]), rand(150,2), (rand(150,2) - array([.75, 1.2])) ))
        
    print "2. Run k-means"
    if args.func is not onlineplot and args.func is not onlinetest:
        args.func(data, args.k_classes, args.end_thresh)
    else:
        args.func(data, args.k_classes, args.end_thresh, args.dim_denom)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        unittest.main()
