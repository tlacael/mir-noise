mir-noise
=========

Repository for MIR Final Project

File Structure:
/			<= All top level code (main, plotting, etc.) goes here
mirlib			<= All useful code goes here.
--FFTParams.py		<= Class for passing FFTParameters easily
--MatlabData.py		<= Methods for handling reading matlab data files
--benchmarking.py	<= Methods for benchmarking
--featurevector.py	<= Methods for managing feature vectors
--mir_utils.py		<= MIR-related methods that don't fit in any other 
			   categories
--peakdetector.py	<= Methods for finding peaks in a function
--windowmanager.py	<= Windowing methods

mirlib/feature_extraction <= Modules related to feature extraction
--lowlevel_spectral.py	<= Methods for low-level spectral features
--lowlevel_temporal.py 	<= Methods for low-level temporal features
--BeatDetector.py	<= Onset Detection

mirlib/feature_analysis <= Modules related to feature analysis
--nn_classifier.py	<= Nearest Neighbor Classifier
--kmeans.py		<= Methods for handling k-means

mirlib_test		<= Unit testing and normal testing for mirlib


Package Dependencies - You must install these to run this!
* numpy		     http://numpy.scipy.org/
* scipy		     http://www.scipy.org/Installing_SciPy
