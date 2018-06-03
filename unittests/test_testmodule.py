import unittest
from testmodule_cpp import sum_list, sum_nested, test_access

class ExampleObject(object):
    def __init__(self):
        self.dbl_attr = 0.0
        self.int_attr = 1
        self.list_attr = [1,2,3]

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

    def test_object_access(self):
        obj = ExampleObject()
        test_access(obj)

        # Test access function changes some of the attributes
        self.assertAlmostEqual(obj.dbl_attr, 10.0)
        self.assertEqual(obj.int_attr, 2)
        self.assertEqual(obj.list_attr, [10,2,3])

if __name__ == "__main__":
    unittest.main()
