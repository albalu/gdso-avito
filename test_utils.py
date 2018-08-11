# coding: utf-8

from __future__ import unicode_literals, absolute_import
import pandas as pd
import os
import unittest
from utils import get_aggregate_periods

test_dir = os.path.dirname(__file__)

class TestPeriods(unittest.TestCase):
    def test_featurization(self):
        samples = pd.read_csv('data/train_100000.csv', usecols=['item_id', 'user_id'])
        # samples.to_csv('data/train_100000.csv')
        train_periods = pd.read_csv('data/periods_train_500000.csv',
                                    parse_dates=['date_to', 'date_from', 'activation_date'],
                                    infer_datetime_format=True)
        train_periods = get_aggregate_periods(all_periods=train_periods,
                                              all_samples=samples,
                                              write_to_csv=False)
        # This returns empty df as the data is limited but it tests that the function runs through the end



if __name__ == '__main__':
    unittest.main()