'''
   Christopher Jacoby

   benchmarking.py - methods & classes for handling benchmarking of data.'''

# for testing with arguments
import sys

# for handling matlab vectors
import MatlabData

import numpy as np
from scipy import *

class Benchmark_Handler(object):
    def __init__(self, resultData, groundTruthData, truthDist=0.05):
        self.result = resultData
        self.groundTruth = groundTruthData
        self.truthDist = truthDist

        self.c, self.fPlus, self.fMinus = self.CalculateResults()

    def CalcDistMatrix(self):
        # Generate a square NxN matrix with the ground truths in each row
        N = len(self.groundTruth)
        M = len(self.result)
        distMatrix = np.zeros([N,M])
        matr = []
        for n in range(N):
            for m in range(M):
                distMatrix[n,m] = self.groundTruth[n]

            # Get the distance vector for each 
            matr.append( abs(self.result - distMatrix[n]) )

        return array(matr)

    def CalculateResults(self):

        # calculate the distance between each point in the results and the ground truth
        distMtr = self.CalcDistMatrix()
        truthMatr = array(distMtr < self.truthDist, dtype=int32)
        
        # if the distance is < truthDist, we have a correct (c)
        # this is the sum of all zeros in truthMatr
        c = sum(truthMatr)
            
        # if there are results that don't have matched ground truths, it is f+
        # this is the number of rows of truthMatr that don't have a one
        fPlus = 0
        for i in truthMatr:
            if sum(i) == 0:
                fPlus += 1

        # if there are ground truths that don't have matched results, it is f-
        # this is the number of columns of truthMatr that don't have a one
        fMinus = 0
        for i in np.transpose(truthMatr):
            if sum(i) == 0:
                fMinus += 1

        return c, fPlus, fMinus
        

    def Precision(self):
        return self.c / float(self.c + self.fPlus)

    def Recall(self):
        return self.c / float(self.c + self.fMinus)

    def F_Measure(self):
        return 2 * self.c / float(2 * self.c + self.fMinus + self.fPlus)

def BenchmarkResults(resultData, groundTruthData, truthDist=0.05):
    bh = Benchmark_Handler(resultData, groundTruthData, truthDist)

    # Calculate Precision
    P = bh.Precision()
    
    # Calculate recall 
    R = bh.Recall()
    
   # Calculate F-measure
    F = bh.F_Measure()

    return P,R,F

def PrintResults(fs, h, onsets, matlabOnsets):

    on1 = GetTimesFromHop(fs, h, onsets[0])
    on2 = GetTimesFromHop(fs, h, onsets[1])
    on3 = GetTimesFromHop(fs, h, onsets[2])

    P1, R1, F1 = BenchmarkResults(on1, matlabOnsets)
    P2, R2, F2 = BenchmarkResults(on2, matlabOnsets)
    P3, R3, F3 = BenchmarkResults(on3, matlabOnsets)

    print "Ground Truth (matlab set) - %d onsets" % (len(matlabOnsets))
    print "Results & # Offsets & P & R & F"
    print "Log-Energy Derivative & %d & %0.3f & %0.3f &  %0.3f" % (len(on1), P1, R1, F1)
    print "Rectified Spectral Flux & %d & %0.3f & %0.3f & %0.3f" % (len(on2), P2, R2, F2)
    print "Rectified Complex Flux & %d & %0.3f & %0.3f & %0.3f" % (len(on3), P3, R3, F3)

def main():
    ''' This main function is for testing the benchmarking functions herein.'''

    # Read in the benchmark vector
    matlabFile = sys.argv[1]
    benchmarkVector = MatlabData.GetMatlabVectors(matlabFile)
    print "Benchmark Vector:", benchmarkVector

    # if a test result vector given, read it in
    if len(sys.argv) > 2:
        resultVector = np.load(sys.argv[2])
    else: # Else generate a random one from the benchmark vector
        resultVector = []
    print "Test Vector:", resultVector

    P,R,F = BenchmarkResults(resultVector, benchmarkVector)
    print "P:", P, "R:", R, "F:", F

    
if __name__=="__main__":
    main()
