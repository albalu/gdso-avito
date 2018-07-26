# coding: utf-8

from __future__ import unicode_literals, absolute_import
import pandas as pd
import os
import unittest
from utils import generate_periods_features

test_dir = os.path.dirname(__file__)

class TestPeriods(unittest.TestCase):
    def test_featurization(self):
        train_periods = pd.read_csv('data/periods_train_500000.csv')
        train_periods = generate_periods_features(train_periods)
        train_periods.set_index('item_id')


if __name__ == '__main__':
    unittest.main()