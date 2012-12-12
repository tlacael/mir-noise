import numpy as np

class featurevector_holder:
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
    def __init__(self, vector_template, initial_size=0):
        self.vector_name_map, self.vector_index_map = self.construct_maps(vector_template)
        self.vector_length = sum(self.vector_index_map.values())

        # current time index
        self.clear()

        # if initial_size > 0, create vector of zeros
        self.vector = np.array([])
        #self.vector = np.zeros([initial_size, self.vector_length])

    def construct_maps(self, vector_template):
        arrtmp = np.asarray(vector_template)
        name_map = zip(arrtmp[:,0], np.asarray(arrtmp[:, 1], dtype=int))
        index_map = zip(np.asarray(arrtmp[:,1], dtype=int), np.asarray(arrtmp[:,2], dtype=int))
        return dict(name_map), dict(index_map)

    def get_nvectors(self):
        return len(self.vector)

    def clear(self):
        self.index = 0
        self.vector = np.array([])

    def isvalidindex(self, index):
        return index != None and index >= 0 and index < self.get_nvectors()
    
    def add_vector(self, vector, timeindex=None):
        # Case 1: timeindex=None
        if timeindex is None:
            self.create_vector()
            self.set_vector(self.index, vector)
        # Case 2: timeindex!=None
        else:
            self.create_vector(timeindex)
            self.set_vector(timeindex, vector)

    def add_feature(self, name, feature, timeindex=None):
        # Case 1: timeindex=None
        if timeindex is None:
            self.create_vector()
            self.set_feature(name, self.index, feature)
        # Case 2: timeindex!=None
        else:
            self.create_vector(timeindex)
            self.set_feature(name, timeindex, feature)

    def create_vector(self, timeindex=None):
        startVec = self.vector

        if timeindex is None:
            nToAppend = 1
        elif self.get_nvectors() < timeindex:
            nToAppend = (timeindex - self.get_nvectors()) + 1
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
                self.index += nToAppend

    def set_vector(self, timeindex, vector):
        if self.isvalidindex(timeindex):
            self.vector[timeindex] = vector
        else:
            # Throw invalid index exception
            raise IndexError

    def get_feature_range(self, name):
        try:
            feature_start = self.vector_name_map[name]
            feature_end = feature_start + self.vector_index_map[feature_start]
            return feature_start, feature_end
        except KeyError:
            print "Invalid Key:", name
            return None, None

    def set_feature(self, name, timeindex, values):
        if self.isvalidindex(timeindex):
            feature_start, feature_end = self.get_feature_range(name)
            self.vector[timeindex, feature_start:feature_end] = values
        else:
            # Throw invalid index exception
            raise IndexError("Time Indec out of range")

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

    
    def save(self, filename):
        with open(filename, 'w+b') as f:
            np.save(f, self.vector_name_map)
            np.save(f, self.vector_index_map)
            np.save(f, self.vector_length)
            np.save(f, self.index)
            np.save(f, self.vector)

    def load(self, filename):
        with open(filename, 'r+b') as f:
            self.vector_name_map = np.load(f)
            self.vector_index_map = np.load(f)
            self.vector_length = np.load(f)
            self.index = np.load(f)
            self.vector = np.load(f)
