import unittest
from testmodule_cpp import sum_list

class TestFramework( unittest.TestCase ):
    def test_sumlist(self):
        data = [1,2,3,4]
        res = 10
        computed_sum = sum_list(data)
        self.assertEqual(computed_sum,res)

if __name__ == "__main__":
    unittest.main()
