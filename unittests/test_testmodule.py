import unittest
import numpy as np
from testmodule_cpp import sum_list, sum_nested, test_access, list_from_vector
from testmodule_cpp import sum1D, sum2D, sum3D
import sys
import gc

class ExampleObject(object):
    def __init__(self):
        self.dbl_attr = 0.0
        self.int_attr = 1
        self.list_attr = [1,2,3]
        self.np_attr = np.array([1,2,3]).astype(np.int32)

class TestFramework( unittest.TestCase ):
    def test_sumlist(self):
        data = [1,2,3,4]
        res = 10
        computed_sum = sum_list(data)
        self.assertEqual(computed_sum,res)

    def test_sumneste(self):
        data = [[0.0,1.0,2.0],[0.0,1.0,2.0],[0.0,1.0,2.0]]
        res = 9.0
        computed_sum = sum_nested(data)
        self.assertAlmostEqual(computed_sum, res)

    def test_list_from_vector(self):
        mylist = list_from_vector()
        referrers = gc.get_referrers(mylist)
        self.assertEqual(mylist, [0,1,2,3])
        self.assertEqual(len(referrers),1)

    def test_object_access(self):
        obj = ExampleObject()
        test_access(obj)

        # Test access function changes some of the attributes
        self.assertAlmostEqual(obj.dbl_attr, 10.0)
        self.assertEqual(obj.int_attr, 2)
        self.assertEqual(obj.list_attr, [10,2,3])
        self.assertTrue( np.allclose( obj.np_attr, np.array([7,2,3]) ) )

    def test1D(self):
        array = np.linspace(0.0,10.0,50.0)
        npsum = np.sum(array)
        cppsum = sum1D(array)
        self.assertAlmostEqual(npsum,cppsum)

    def test2D(self):
        array = np.random.rand(10,5)
        npsum = np.sum(array)
        cppsum = sum2D(array)
        self.assertAlmostEqual(npsum,cppsum)

    def test3D(self):
        array = np.random.rand(10,5,7)
        npsum = np.sum(array)
        cppsum = sum3D(array)
        self.assertAlmostEqual(npsum,cppsum)

if __name__ == "__main__":
    unittest.main()
