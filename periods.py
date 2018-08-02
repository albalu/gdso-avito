from collections import OrderedDict
import numpy as np
import pandas as pd
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import TimeSeriesSplit
pd.options.mode.chained_assignment = None

import pandas as pd
import gc
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles # requires pip install matplotlib_venn
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import scipy

russian_2017_holidays = pd.to_datetime([
    '2017-01-01', # new years day
    '2017-01-02',  # Bank Holiday
    '2017-01-03',  # Bank Holiday
    '2017-01-04',  # Bank Holiday
    '2017-01-05',  # Bank Holiday
    '2017-01-06',  # Bank Holiday
    '2017-01-07',  # Orthodox Christmas Day
    '2017-02-23',  # Defence of the Fatherland
    '2017-02-24',  # Defence of the Fatherland Holiday
    '2017-03-08',  # Women's Day
    '2017-05-01',  # Labour Day
    '2017-05-08',  # Bridge holiday compensated by Sat May 6th
    '2017-05-09',  # Victory Day
    '2017-06-12',  # National Day
    '2017-11-04',  # Day of Unity
    '2017-11-06',  # Day of Unity Holiday
    '2017-11-26',  # Mother's Day
], yearfirst=True)

def is_russian_2017_holiday(date):
    """
    Whether a given date was a national holiday in Russia or not.

    Note that
    there seem to be a package ( https://pypi.org/project/prod-cal/ ) that
    based on this blog ( https://habr.com/post/281040/ ) handles holiday and
    workdays of Russia and some other countries but it could not be installed
    as it's trying to import ProdCal from holidays but holidays package does not
    have ProdCal nor does it support Russia :|
    Args:
        date (datetime): datetime object to be checked.

    Returns (bool):
        is holiday (True) or not (False)?
    """
    return int(date in russian_2017_holidays)


def featurize_date_col(df, col, remove_when_done=False):
    """
    Returns some information (features) about the column date, added to the
    base dataframe.

    Args:
        df (pandas.DataFrame): original input dataframe
        col_name (str): column name or feature. It must be a date object dtype
        remove_when_done (bool): remove the original date column

    Returns (pandas.DataFrame):
    """
    df['{}_isholiday'.format(col)] = df[col].apply(is_russian_2017_holiday)
    df['{}_wday'.format(col)] = df[col].apply(lambda x: x.dayofweek)
    df['{}_yday'.format(col)] = df[col].apply(lambda x: x.dayofyear)
    if remove_when_done:
        df = df.drop(col, axis=1)
    return df


def get_aggregate_periods(all_periods, all_samples, write_to_csv=True):
    """
    Returns all_samples (tran and test dfs) with features related to users
    overall listings (aggregate periods features) to be merged with train data.

    Args:
        all_periods (pandas.DataFrame): must have "item_id", "activation_date",
            "date_from" and "date_to". dates must be datetime type
        all_samples (pandas.DataFrame): must have "item_id" and "user_id"
        write_to_csv (bool): whether to write featurized data to csv at the end

    Returns (pandas.DataFrame):
    """
    null_idx = all_periods['activation_date'].isna()
    all_periods['activation_date'][null_idx] = all_periods['date_from'][null_idx]
    assert (all_periods['date_to'] >= all_periods['date_from']).all()
    # we can do this below since they're all in 2017 and this way is faster
    all_periods['days_to_publish'] = all_periods['date_from'].dt.dayofyear - \
                                     all_periods['activation_date'].dt.dayofyear
    all_periods['days_online'] = all_periods['date_to'].dt.dayofyear - \
                                 all_periods['date_from'].dt.dayofyear
    for col in ['activation_date', 'date_from', 'date_to']:
        all_periods = featurize_date_col(all_periods, col)

    grouped = all_periods.groupby('item_id')
    base = grouped[['item_id']].count().rename(columns={'item_id': 'nlisted'})
    base['sum_days_online'] = grouped[['days_online']].sum()
    base['mean_days_online'] = grouped[['days_online']].mean()
    base['last_days_online'] = grouped[['days_online']].last()
    base['sum_days_to_publish'] = grouped[['days_to_publish']].sum()
    base['mean_days_to_publish'] = grouped[['days_to_publish']].mean()
    base['median_date_to_isholiday'] = grouped[['date_to_isholiday']].median()
    base['median_date_to_wday'] = grouped[['date_to_wday']].median()
    base['median_date_to_yday'] = grouped[['date_to_yday']].median()

    base['start_date'] = grouped[['date_from']].min()
    base['end_date'] = grouped[['date_to']].max()
    for col in ['start_date', 'end_date']:
        base = featurize_date_col(base, col, remove_when_done=True)
    if 'item_id' not in all_periods:
        all_periods = all_periods.reset_index()
    if 'item_id' not in base:
        base = base.reset_index()
    all_periods = all_periods.drop_duplicates(['item_id'])
    all_periods = all_periods.merge(base, on='item_id', how='left')
    all_periods = all_periods.merge(all_samples, on='item_id', how='left')
    avg_per_user_periods = all_periods.drop(
        ['item_id', 'activation_date', 'date_from', 'date_to'], axis=1
                                            ).groupby('user_id').mean()
    avg_per_user_periods['nitems'] = all_periods[
        ['user_id', 'item_id']].groupby('user_id').count().reset_index()['item_id']
    if write_to_csv:
        avg_per_user_periods.to_csv('data/periods_aggregate_features.csv')
    return avg_per_user_periods



