import unittest
from testmodule_cpp import sum_list, sum_nested

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

if __name__ == "__main__":
    unittest.main()
