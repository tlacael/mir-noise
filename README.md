mir-noise
=========

Repository for MIR Final Project

File Structure:
/			<= All top level code (main, plotting, etc.) goes here
mirlib			<= All useful code goes here.
--FFTParams.py		<= Class for passing FFTParameters easily
--MatlabData.py		<= Methods for handling reading matlab data files [not in use]
--audiofile.py		<= Methods for handling complex audio file i/o
--benchmarking.py	<= Methods for benchmarking [not in use]
--featurevector.py	<= Methods for managing feature vectors
--mir_utils.py		<= MIR-related methods that don't fit in any other 
			   categories
--peakdetector.py	<= Methods for finding peaks in a function
--windowmanager.py	<= Windowing methods

mirlib/feature_extraction <= Modules related to feature extraction
--lowlevel_spectral.py	<= Methods for low-level spectral features
--lowlevel_temporal.py 	<= Methods for low-level temporal features [not in use]
--BeatDetector.py	<= Tempo Detection [not in use]
--OnsetDetector.py	<= Onset Detection [not in use]
--calcLoudness.py	<= Sones calculation
--eventDetect.py	<= find events from an energy function

mirlib/feature_analysis <= Modules related to feature analysis
--nn_classifier.py	<= Nearest Neighbor Classifier [not in use]
--kmeans.py		<= Methods for handling k-means
--discrimination.py	<= methods for handling classifier discrimination [not in use]

mirlib_test		<= Unit testing and normal testing for mirlib


Package Dependencies - You must install these to run this!
* numpy		     http://numpy.scipy.org/
* scipy		     http://www.scipy.org/Installing_SciPy
* marlib	     https://bitbucket.org/ejh333/marlib


Usage:
This program can be run in several modes. Use the "-h" on each (or the main program) for more details.

Modes:
python main.py getfeatures <audiofile>	Runs the feature extraction component,
       	       		   		which outputs .npy files storing the feature
					vectors.
python main.py featureselect <kmin> <kmax> <kinterval>
       	       		     	    	Runs features selection on the k's provided.
					Dislpays a plot of J0.
python main.py clustering <k> <-w>
					Run for a single k, and attempt to
					choose the best centroids for it.
					the -w parameter causes it to output
					the resultant classes to an audio file in 
					the results folder.
