''' @author Christopher Jacoby

algorithms for implementing k-means.

In the end... so far we just use the scipy versions, because these 
are returning weird errors in low-level numpy for large k that I
can't seem to find the cause of.

Tests located in the mirlib_test folder.
'''

import numpy as np
from .. import mir_utils
import scipy.cluster.vq

def GetNearestCentroids(data, centroids):
    # Use Euclidean Distance to get the nearest centroids for each point
    k = len(centroids)
    if len(data.shape) == 2:
        nDim = len(data)
    else:
        nDim = 1
    dist = np.zeros([k, nDim]) # the dist matrix for each centroid
        
    for i in range(k):
        dist[i] = mir_utils.euclid_dist(data, centroids[i])

    return dist.argmin(axis=0)

    #def GetVectorQuantizationStatistics(data, centroids):
    

class kmeans_runner:
    def __init__(self, k, end_thresh=0.01):
        ''' k is the number of centroids to use. '''
        self.k_classes = k
        self.k_counts = np.zeros(self.k_classes)
        self.itr_count = 0
        self.stop_flag = False
        self.end_thresh = end_thresh

    def init_data(self, data):
        ''' Given an array of data, compute initial centroids, choosing random data
        to start with.
        returns: the centroids.'''
        self.data = data
        self.data_dim = data.shape[0]
        self.feature_dim = data.shape[1]

        # Initialize the centroids to a random data point
        rand_index = np.random.randint(0, self.data_dim, self.k_classes)
        
        self.centroids = data[rand_index]

    def iterate(self):

        self.UpdateCentroids_Offline()
        self.itr_count += 1

        if (self.update_changes < self.end_thresh).all():
            self.stop_flag = True

    def UpdateCentroids_Offline(self):
        nearest_centroids = GetNearestCentroids(self.data, self.centroids)
        centroid_update = np.zeros(self.centroids.shape)

        for i in range(self.k_classes):
            centroid_update[i] = np.mean( self.data[(nearest_centroids == i).nonzero()], axis=0 )

        self.update_changes = centroid_update - self.centroids
        self.centroids = self.centroids + self.update_changes

    def UpdateCentroids_Online(self, data):
        # because we're sending only one point in here, should return the index
        # of the nearest centroid
        nearest_centroid = GetNearestCentroids(data, self.centroids)
        self.k_counts[nearest_centroid] += 1

        # m_i = m_i + (1/n_i) * (x - m_i)
        self.centroids[nearest_centroid] += (1 / np.float(self.k_counts[nearest_centroid])) * (data - self.centroids[nearest_centroid])

    def run_offline(self, data):
        self.init_data(data)
        while self.stop_flag is False:
            self.iterate()

        return self.centroids, self.itr_count

    def iterate_online(self, data):
        self.UpdateCentroids_Online(data)        
        return self.centroids, self.itr_count

def kmeans(data, k, thresh):
    kmr = kmeans_runner(k, thresh)
    return kmr.run_offline(data)

def vq(data, centroids):
    return GetNearestCentroids(data, centroids)
    
def scipy_kmeans(data, k):
    return scipy.cluster.vq.kmeans(data, k)

def scipy_vq(data, centroids):
    return scipy.cluster.vq.vq(data, centroids)

def print_vq_stats(data, centroids):
    idx, dist = scipy_vq(data, centroids)

    print "Mean & Std Dev Dist to Centroid for each class:"
    for i in range(len(centroids)):
        meanDist = dist[idx==i].mean()
        stdDist = dist[idx==i].std()
        print "class %d: %0.3f %0.3f" % (i, meanDist, stdDist)

