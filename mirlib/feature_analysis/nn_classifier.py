import os
import numpy as np
from tempfile import TemporaryFile
import matplotlib.pyplot as plt

CLASSIFIER_SAVE_FILENAME = 'classes.npy'

class LabelledPoint:
    def __init__(self, dataPoint, label):
        self.data = dataPoint
        self.label = label

class Classifier:

    def __init__(self):
        ''' If initializing this for the first time, send it an nClasses. Otherwise, it will get them from
        the file. '''

        self.ndims = 0
        self.trained_data = []

    def ContainsClass(self, classID):
        return classID < self.nClasses and classID < len(self.class_points)

    def Train(self, data, class_number):
        if self.ndims == 0:
            self.ndims = len(data)

        if len(data) != self.ndims:
            raise ValueError('This data has different dimensions from the other data given. (%d %d)' % (len(data), self.ndims))
            
        self.trained_data.append(LabelledPoint(data, class_number))

    def Save(self):
        f = open(CLASSIFIER_SAVE_FILENAME, 'w+b')
        np.save(f, self.ndims)
        np.save(f, self.trained_data)
        f.close()

    def Load(self):
        if os.path.isfile(CLASSIFIER_SAVE_FILENAME):
            f = open(CLASSIFIER_SAVE_FILENAME, 'r+b')
            self.ndims = np.load(f)
            self.trained_data = np.load(f).tolist()
            f.close()

    def GetSampleCount(self):
        return len(self.trained_data)

    def GetDataLabelHist(self):
        hist = dict()
        for entry in self.trained_data:
            label = entry.label

            if label in hist.keys():
                hist[label] += 1
            else:
                hist[label] = 1

        return hist

    def GetDataByLabel(self):
        retdata = dict()

        for entry in self.trained_data:
            d = entry.data
            l = entry.label

            if l in retdata.keys():
                retdata[l].append(d)
            else:
                retdata[l] = []
                retdata[l].append(d)

        for key in retdata.keys():
            retdata[key] = np.array(retdata[key])

        return retdata

    def GetLabels(self):
        hist = self.GetDataLabelHist()
        return hist.keys()

    def ConvertLabelToIndex(self, x):
        mapping = self.GetLabels()

        ret = np.array(x)
        for i in range(len(mapping)):
            ret[(ret == mapping[i]).nonzero()] = i + 1

        ret = np.array([int(x) if x.isdigit() else 0 for x in ret])

        return ret

    def Print(self):
        arrdata = np.array(self.trained_data)
        print "Shape:", arrdata.shape, "ndims:", self.ndims
        print "Labels:", self.GetDataLabelHist()

    def Clear(self):
        self.ndims = 0
        self.trained_data = []
        self.Save()

class NearestNeighbor(Classifier):

    def GetNearestNeighbor(self, x):
        ''' This is a similarity measure. We're finding the dot product of all of the vectors with this one,
        and then returning the label with the largest value. 
        indToSkip specifies how many indexes at the start of the vector to ignore
        '''
        
        nDataPoints = len(self.trained_data)
        results = np.zeros(nDataPoints)
        for i in range(nDataPoints):
            results[i] = x.dot(self.trained_data[i].data)

        return self.trained_data[results.argmax()].label


def main():
    print "Nearest Neighbor Classifier Test"

    dim = 2
    classes = 4

    classifier = NearestNeighbor()

    trainData = ( (np.array([4,4]), 0),
                  (np.array([4.5, 5]), 0),
                  (np.array([1.1, 2]), 0),
                  (np.array([-5, 5]), 1),
                  (np.array([-3, 2.34]), 1),
                  (np.array([-14, 6]), 1),
                  (np.array([-5, -5]), 2),
                  (np.array([-10, -5]), 2),
                  (np.array([-7, -2]), 2),
                  (np.array([4, -8]), 3),
                  (np.array([5, -5]), 3),
                  (np.array([6, -3]), 3))

    plt.figure()
    plt.grid(True)

    # Train the classifier with the training data
    for point in trainData:
        plt.scatter(point[0][0], point[0][1])
        classifier.Train(point[0], point[1])

    for point in classifier.trained_data:
        plt.scatter(point.data[0], point.data[1], color='r')

    classifier.Save()

    classifier = NearestNeighbor()
    classifier.Load()
    
    # Do some tests to show that they're at the right points.
    pt1 = np.array([10, 10])
    pt2 = np.array([2, 2])
    pt3 = np.array([-3, 12])
    pt4 = np.array([-6, 2])
    pt5 = np.array([-6, -6])
    pt6 = np.array([-43, -3])
    pt7 = np.array([10, -5])
    pt8 = np.array([5, -5])
    
    res1 = classifier.GetNearestNeighbor(pt1)
    print "Test1:", res1, res1 == 0

    res2 = classifier.GetNearestNeighbor(pt2)
    print "Test2:", res2, res2 == 0

    res3 = classifier.GetNearestNeighbor(pt3)
    print "Test3:", res3, res3 == 1

    res4 = classifier.GetNearestNeighbor(pt4)
    print "Test4:", res4, res4 == 1

    res5 = classifier.GetNearestNeighbor(pt5)
    print "Test5:", res5, res5 == 2

    res6 = classifier.GetNearestNeighbor(pt6)
    print "Test6:", res6, res6 == 2

    res7 = classifier.GetNearestNeighbor(pt7)
    print "Test7:", res7, res7 == 3

    res8 = classifier.GetNearestNeighbor(pt8)
    print "Test4:", res8, res8 == 3

    plt.show()
                   

if __name__ == "__main__":
    main()
