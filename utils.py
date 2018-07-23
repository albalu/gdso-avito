import pandas as pd
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
        is holiday (True) or not? (False)?
    """
    return int(date in russian_2017_holidays)


def generate_periods_features(df):
    """
    Given a dataframe with features item_id, activation_date, date_from, date_to,
    it generates features that help the training the machine learning (ML) model.
    It then drops the original columns that are not directly usable by ML models.

    Args:
        df (pandas.DataFrame): must include 'item_id', 'activation_date',
            'date_from', 'date_to' columns.

    Returns (pandas.DaraFrame): featurized input with the original columns dropped
    """
    for col in ['item_id', 'activation_date', 'date_from', 'date_to']:
        assert col in df
    null_idx = df['activation_date'].isnull()
    # checked train and test and date_from, date_to don't have nulls
    assert df.isnull().sum().sum() == 0
    df['activation_date'][null_idx] = df['date_from'][null_idx]
    for col in ['activation_date', 'date_from', 'date_to']:
        df[col] = df[col].apply(pd.to_datetime, yearfirst=True)
    assert len(df) == len(df.drop_duplicates('item_id'))
    assert (df['date_to'] > df['date_from']).all()
    assert (df['date_from'] >= df['activation_date']).all()

    # feature generation
    df['days_to_publish'] = (df['date_from']-df['activation_date']).dt.days
    df['days_online'] = (df['date_to'] - df['date_from']).dt.days
    for col in ['activation_date', 'date_from', 'date_to']:
        df['{}_isholiday'.format(col)] = df[col].apply(is_russian_2017_holiday)
        df['{}_dayofweek'.format(col)] = df[col].apply(lambda x: x.dayofweek)
        df['{}_dayofyear'.format(col)] = df[col].apply(lambda x: x.dayofyear)
    return df.drop(['activation_date', 'date_from', 'date_to'], axis=1)