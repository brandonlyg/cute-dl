# coding=utf-8

import sys
sys.path.append("..")
sys.path.append("../cutedl")

import pdb
import numpy as np
import unittest
from unittest import TestCase
from cutedl import dlmath


class TestDlmath(TestCase):

    def test_categories_sample(self):
        print("test categories_sample")
        input = np.random.uniform(0, 1, (3, 3, 4))
        out = dlmath.categories_sample(input, 2)
        pdb.set_trace()
        print("out: ", out)



if '__main__' == __name__:
    unittest.main()
