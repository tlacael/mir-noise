import numpy as np
import os

def GetVectorLength(vector):
    if vector.ndim == 1:
        vecLen = 1
    else:
        vecLen = len(vector)
    return vecLen

class feature_holder:
    ''' Class to manage collections of feature vectors. 

    Exposed methods:
     get_nvectors
     add_vector
     add_feature

     create_vector
     set_vector
     set_feature
     save
     load

    '''
    def __init__(self, *args, **kwargs):
        if kwargs.has_key('filename'):
            filename = kwargs['filename']
            if os.path.exists(filename):
                self.load(filename)
            else:
                raise AssertionError("Error: bad file path")
        else:
            if len(args) > 0:
                vector_template = args[0]

                self.filename = None
                if len(args) > 1:
                    self.filename = args[1]
            
                self.vector_name_map, self.vector_index_map = self.construct_maps(vector_template)
                self.vector_length = sum(self.vector_index_map.values())
                
                # current time index
                self.clear()
                
                # if initial_size > 0, create vector of zeros
                self.vector = np.array([])
                
                # initalize a dict for mapping the indecies to time
                self.time_dict = dict()
            else:
                raise AssertionError("Inappropriate parameters!")

    def construct_maps(self, vector_template):
        arrtmp = np.asarray(vector_template)
        name_map = zip(arrtmp[:,0], np.asarray(arrtmp[:, 1], dtype=int))
        index_map = zip(np.asarray(arrtmp[:,1], dtype=int), np.asarray(arrtmp[:,2], dtype=int))
        return dict(name_map), dict(index_map)

    def get_nvectors(self):
        return len(self.vector)

    def clear(self):
        self.vector = np.array([])

    def isvalidindex(self, index, length=1):
        return index != None and index >= 0 and (index + length - 1) < self.get_nvectors()
    
    def add_vector(self, vector, timeindex=None, timelabel=None):
        vecLen = GetVectorLength(vector)
        
        # Case 1: timeindex=None
        if timeindex is None:
            startingIndex = self.vector.shape[0]
            self.create_vector(timelength=vecLen)
            self.set_vector(startingIndex, vector, timelabel)
        # Case 2: timeindex!=None
        else:
            self.create_vector(timeindex, vecLen)
            self.set_vector(timeindex, vector, timelabel)

    def add_feature(self, name, feature, timeindex=None, timelabel=None):
        vecLen = GetVectorLength(feature)
        
        # Case 1: timeindex=None
        if timeindex is None:
            startingIndex = self.vector.shape[0]
            self.create_vector(timelength=vecLen)
            self.set_feature(name, startingIndex, feature, timelabel)
        # Case 2: timeindex!=None
        else:
            self.create_vector(timeindex, vecLen)
            self.set_feature(name, timeindex, feature, timelabel)

    def create_vector(self, timeindex=None, timelength=1):
        startVec = self.vector

        if timeindex is None:
            nToAppend = timelength
        elif self.get_nvectors() < (timeindex + timelength):
            nToAppend = ((timeindex + timelength - 1) - self.get_nvectors()) + 1
        else:
            nToAppend = 0

        if nToAppend > 0:
            appendVec = np.zeros([nToAppend, self.vector_length])

            # Case 1: self.vector is []
            if self.get_nvectors() == 0:
                self.vector = appendVec
            # Case 2: self.vector is not []
            else: # self.get_nvectors() > 0:
                self.vector = np.concatenate([startVec, appendVec])

    def set_vector(self, timeindex, vector, timelabel):
        vecLen = GetVectorLength(vector)
            
        if self.isvalidindex(timeindex, vecLen):
            self.vector[timeindex:(timeindex + vecLen)] = vector
        else:
            # Throw invalid index exception
            raise IndexError("idx: %d, len: %d, max: %d" % (timeindex, vecLen, self.get_nvectors()))

        if timelabel is not None:
            self.set_timelabel(timelabel, timeindex, vecLen)

    def get_feature_range(self, name):
        try:
            feature_start = self.vector_name_map[name]
            feature_end = feature_start + self.vector_index_map[feature_start]
            return feature_start, feature_end
        except KeyError:
            print "Invalid Key:", name
            return None, None

    def set_feature(self, name, timeindex, values, timelabel):
        vecLen = GetVectorLength(values)
        
        if self.isvalidindex(timeindex, vecLen):
            feature_start, feature_end = self.get_feature_range(name)
            self.vector[timeindex:(timeindex + vecLen), feature_start:feature_end] = values
        else:
            # Throw invalid index exception
            raise IndexError("Time Index out of range")

        if timelabel is not None:
            self.set_timelabel(timelabel, timeindex, vecLen)

    def get_feature(self, name, timeindex=None):
        feature_start, feature_end = self.get_feature_range(name)

        ret = None
        # if none, return ALL the values for this feature
        if timeindex is None:
            ret = self.vector[:, feature_start:feature_end]
        else:
            if self.isvalidindex(timeindex):
                ret = self.vector[timeindex, feature_start:feature_end]
            else:
                # Throw invalid index exception
                raise IndexError("Time Index out of range")
        return ret

    def set_timelabel(self, timelabel, timeindex, vecLen):
        self.time_dict[timelabel] = (timeindex, vecLen)
        
    def get_vector_by_label(self, label):
        (timeind, length) = self.time_dict[label]

        if self.isvalidindex(timeind, length):
            return self.vector[timeind : (timeind + length)]
        else:
            raise IndexError("Stored Label contains incorrect index %d length %d" % (timeind, length))

    def get_feature_by_label(self, label, name):
        (timeind, length) = self.time_dict[label]
        feature_start, feature_end = self.get_feature_range(name)

        if self.isvalidindex(timeind, length):
            return self.vector[timeind : (timeind + length), feature_start:feature_end]
        else:
            raise IndexError("Time Index out of range")

    def get_event_start_indecies(self):
        return sorted([x for x,y in self.time_dict.values()])

    def get_index_time_map(self):
        return {v:k for k, v in self.time_dict.items()}
        
    def save(self, filename):
        with open(filename, 'w+b') as f:
            np.save(f, self.filename)
            np.save(f, self.vector_name_map)
            np.save(f, self.vector_index_map)
            np.save(f, self.vector_length)
            np.save(f, self.time_dict)
            np.save(f, self.vector)

        if os.path.exists(filename):
            return os.stat(filename).st_size
        else:
            return None 

    def load(self, filename):
        with open(filename, 'r+b') as f:
            self.filename = np.load(f).item()
            self.vector_name_map = dict(np.load(f).item())
            self.vector_index_map = dict(np.load(f).item())
            self.vector_length = np.load(f).item()
            self.time_dict = dict(np.load(f).item())
            self.vector = np.load(f)

    def __str__( self ):
        return "vector_holder instance\n" + \
          "---------------------------------\n" + \
          "Reference audio file - %s\n" % (self.filename) + \
          "Name Map - %s\n" % str(self.vector_name_map) + \
          "Index Map - %s\n" % str(self.vector_index_map) + \
          "Length - %s\n" % str(self.vector_length) + \
          "Vector Shape - %s\n" % str(self.vector.shape) + \
          "---------------------------------\n"
          #"Time Mapping - %s\n" % str(self.time_dict) + \
