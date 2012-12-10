import numpy as np
from numpy import vstack, array
from numpy.random import rand
from mirlib.feature_analysis import kmeans
from pylab import *
import time

import argparse

def kmr_test_plot(data, k, learn_fact, end_thresh):
    ion()
    fig = figure()
    ax = fig.add_subplot(111)
    ax.grid(True)

    # get k centroids
    kmr = kmeans.kmeans_runner(k, learn_fact, end_thresh)
    kmr.init_data(data)
    print kmr.centroids

    plot(data[:,0], data[:,1], 'o')

    while kmr.stop_flag is False:
        kmr.iterate()
        #print kmr.centroids, kmr.itr_count
        plot(kmr.centroids[:, 0], kmr.centroids[:, 1], 'sr')
        time.sleep(.2)
        draw()

    ioff()
    show()
    print kmr.itr_count, kmr.centroids

def test(data, k, learn_fact, end_thresh):
    centroids, nItr = kmeans.kmeans(data, k, learn_fact, end_thresh)
    print "N Iterations: %d" % (nItr)
    print "Means:", centroids

def onlineplot(data, k, learn_fact, end_thresh, on_denom):
    print "onlineplot", k, learn_fact, end_thresh, on_denom

    ion()
    fig = figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_ylim((-1.5, 1.5))
    ax.set_xlim((-1.5, 1.5))

    # get k centroids
    kmr = kmeans.kmeans_runner(k, learn_fact, end_thresh)

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

    ioff()
    show()
    print "Final Counts:", kmr.k_counts
    print kmr.centroids

def onlinetest(data, k, learn_fact, end_thresh, on_denom):
    print "onlinetest", k, learn_fact, end_thresh, on_denom
    kmr = kmeans.kmeans_runner(k, learn_fact, end_thresh)

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
    parser.add_argument("-u", "--update_mult", help="Learning Multiplier for each step",
                        default=0.01, type=float)
    parser.add_argument("-t", "--end_thresh", help="Distance threshold to end",
                        default=0.01, type=float)

    # Define program modes
    subparsers = parser.add_subparsers(help="Program Mode Help")
    parser_plot = subparsers.add_parser("plot", help="Run interactive plot")
    parser_plot.set_defaults(func=kmr_test_plot)
    parser_test = subparsers.add_parser("test", help="Run kmeans test")
    parser_test.set_defaults(func=test)
    parser_onlineplot = subparsers.add_parser("onlineplot", help="Run online kmeans test")
    parser_onlineplot.set_defaults(func=onlineplot)
    parser_onlineplot.add_argument("--dim_denom", help="Denominator of the fraction of the dataset that is to be used with each increment", default=6, type=int)
    parser_onlinetest = subparsers.add_parser("onlinetest", help="Run online kmeans test")
    parser_onlinetest.set_defaults(func=onlinetest)
    parser_onlinetest.add_argument("--dim_denom", help="Denominator of the fraction of the dataset that is to be used with each increment", default=6, type=int)
    args = parser.parse_args()
    
    print "mirlib k-means test"

    print "1. Create a test set"
    # data generation
    data = vstack((rand(150,2) + array([.5,.5]), rand(150,2), (rand(150,2) - array([.75, 1.2])) ))

    print "2. Run k-means"
    if args.func is not onlineplot and args.func is not onlinetest:
        args.func(data, args.k_classes, args.update_mult, args.end_thresh)
    else:
        args.func(data, args.k_classes, args.update_mult, args.end_thresh, args.dim_denom)

if __name__ == "__main__":
    main()
