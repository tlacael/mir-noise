import mirlib.featurevector as fv
import os
import numpy as np
import unittest

class TestFeatureVector_Basic(unittest.TestCase):

    def setUp(self):
        self.nMFCC = 20
        self.vector_template = [('sones', 0, 1),
                           ('mfcc', 1, self.nMFCC)]

        self.fvh = fv.featurevector_holder(self.vector_template)
        self.check_nfeats = sum([z for x,y,z in self.vector_template])

    def test_createvector(self):
        # Starts cleared
        self.assertTrue(self.fvh.get_nvectors() == 0, "Should be 0; is actually %d " % (self.fvh.get_nvectors()))

        # Add a new vector; test the empty case
        self.fvh.create_vector()
        # Check the count
        self.assertEqual(self.fvh.get_nvectors(), 1, "Should be 1; is actually %d " % (self.fvh.get_nvectors()))
        # Check the index
        self.assertEqual(self.fvh.index, 0)
        # Check the dimensions of the vector
        self.assertEqual(self.fvh.vector.shape[1], self.check_nfeats)

        # Add a new vector; test the non-empty case
        self.fvh.create_vector()
        self.assertEqual(self.fvh.get_nvectors(), 2, "Should be 2; is actuually %d " % (self.fvh.get_nvectors()))
        # Check the index
        self.assertEqual(self.fvh.index, 1)
        self.assertEqual(self.fvh.vector.shape[1], self.check_nfeats)

        # Add a new vector; test the case where we're adding more than one
        self.fvh.create_vector(5)
        self.assertEqual(self.fvh.get_nvectors(), 6, "Should be 6; is actually %d" %(self.fvh.get_nvectors()))
        # Check the index
        self.assertEqual(self.fvh.index, 5, "Should be 5; is actually %d" % (self.fvh.index))
        self.assertTrue(self.fvh.vector.shape[1] == self.check_nfeats)

    def test_addvector(self):
        # Starts cleared
        self.assertEquals(self.fvh.get_nvectors(), 0)

        # Add a new vector, and make sure it stuck.
        newVect = np.random.rand(self.check_nfeats)
        self.fvh.add_vector(newVect)
        self.assertEquals(self.fvh.get_nvectors(), 1)
        self.assertTrue((self.fvh.vector[0] == newVect).all())

        # Add a new vector at a time index, and make sure it stuck
        newVect = np.random.rand(self.check_nfeats)
        self.fvh.add_vector(newVect, 4)
        self.assertEquals(self.fvh.get_nvectors(), 5)
        self.assertTrue((self.fvh.vector[4] == newVect).all())

    def test_addfeature(self):
        # Starts cleared
        self.assertEquals(self.fvh.get_nvectors(), 0)

        # Add a new feature, and make sure it stuck.
        newFeature = np.random.rand(self.nMFCC)
        self.fvh.add_feature('mfcc', newFeature)
        self.assertEquals(self.fvh.get_nvectors(), 1)
        self.assertTrue((self.fvh.vector[0, 1:] == newFeature).all())
        self.assertTrue((self.fvh.get_feature('mfcc', 0) == newFeature).all())

        # Add a new feature at a specific time index, make sure it stuck.
        newFeature = np.random.rand(self.nMFCC)
        self.fvh.add_feature('mfcc', newFeature, 9)
        self.assertEquals(self.fvh.get_nvectors(), 10)
        self.assertTrue((self.fvh.vector[9, 1:] == newFeature).all())
        self.assertTrue((self.fvh.get_feature('mfcc', 9) == newFeature).all())

        # Add the other feature and make sure it didn't break the first one
        otherFeature = np.random.rand(1)
        self.fvh.add_feature('sones', otherFeature, 9)
        self.assertEquals(self.fvh.get_nvectors(), 10)
        self.assertTrue((self.fvh.vector[9, 1:] == newFeature).all())
        self.assertEquals(self.fvh.vector[9, 0], otherFeature)
        self.assertTrue((self.fvh.get_feature('sones', 9) == otherFeature).all())

        # Check getting all of the features
        result = self.fvh.get_feature('sones')
        self.assertEquals(len(result), 10)
        self.assertEquals(result[9], otherFeature)

    def test_loadsave(self):
        # Starts cleared
        self.assertEquals(self.fvh.get_nvectors(), 0)

        # Add a new vector at 3
        newVect = np.random.rand(self.check_nfeats)
        self.fvh.add_vector(newVect, 3)
        self.assertEquals(self.fvh.get_nvectors(), 4)
        self.assertTrue((self.fvh.vector[3] == newVect).all())

        filename = 'features_test.npy'
        self.fvh.save(filename)
        self.assertTrue(os.path.exists(filename))

        fvh2 = fv.featurevector_holder(self.vector_template)
        fvh2.load('features_test.npy')
        self.assertEquals(self.fvh.vector_name_map, fvh2.vector_name_map)
        self.assertEquals(self.fvh.vector_index_map, fvh2.vector_index_map)
        self.assertEquals(self.fvh.vector_length, fvh2.vector_length)
        self.assertEquals(self.fvh.index, fvh2.index)
        self.assertTrue( (self.fvh.vector == fvh2.vector).all() )
        

    

if __name__ == "__main__":
    unittest.main()
