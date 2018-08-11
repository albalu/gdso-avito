import warnings
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


config_dict = {'sklearn.ensemble.GradientBoostingRegressor': {
        'n_estimators': [100, 200, 400],
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },
        'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [100, 200, 400],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },
              }

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


def TpotAutoml(mode, feature_names=None, **kwargs):
    """
    WARNING UNPUBLISHED WORK! Please don't distribute

    Returns a class wrapped on TPOTClassifier or TPOTRegressor (differentiated
    via mode argument) but with additional visualization and
    post-processing methods for easier analysis.

    Args:
        mode (str): determines TPOTClassifier or TPOTRegressor to be used
            For example "Classification" or "regressor" are valid options.
        feature_names ([str]): list of feature/column names that is optionally
            passed for post-training analyses.
        **kwargs: keyword arguments accepted by TpotWrapper which have a few
            more arguments in addition to TPOTClassifier or TPOTRegressor
            For example: scoring='r2'; see TpotWrapper and TPOT documentation
            for more details.

    Returns (instantiated TpotWrapper class):
        TpotWrapper that has all methods of TPOTClassifier and TPOTRegressor as
        well as additional analysis methods.
    """
    kwargs['feature_names'] = feature_names
    if mode.lower() in ['classifier', 'classification', 'classify']:
        return _tpot_class_wrapper(TPOTClassifier, **kwargs)
    elif mode.lower() in ['regressor', 'regression', 'regress']:
        return _tpot_class_wrapper(TPOTRegressor, **kwargs)
    else:
        raise ValueError('Unsupported mode: "{}"'.format(mode))


def _tpot_class_wrapper(tpot_class, **kwargs):
    """
    WARNING UNPUBLISHED WORK! Please don't distribute

    Internal function to instantiate and return the child of the right class
    inherited from the two choices that TPOT package provides: TPOTClassifier
    and TPOTRegressor. The difference is that this new class has additional
    analyis and visualization methods.
    Args:
        tpot_class (class object): TPOTClassifier or TPOTRegressor
        **kwargs: keyword arguments related to TPOTClassifier or TPOTRegressor

    Returns (class instance): instantiated TpotWrapper
    """
    class TpotWrapper(tpot_class):

        def __init__(self, **kwargs):
            self.models  = None
            self.top_models = OrderedDict()
            self.top_models_scores = OrderedDict()
            self.feature_names = kwargs.pop('feature_names', None)
            if tpot_class.__name__ == 'TPOTClassifier':
                self.mode = 'classification'
            elif tpot_class.__name__ == 'TPOTRegressor':
                self.mode = 'regression'
            self.random_state = kwargs.get('random_state', None)
            if self.random_state is not None:
                np.random.seed(self.random_state)

            kwargs['cv'] = kwargs.get('cv', 5)
            kwargs['n_jobs'] = kwargs.get('n_jobs', -1)
            super(tpot_class, self).__init__(**kwargs)


        def get_top_models(self, return_scores=True):
            """
            Get a dictionary of top performing run for each sklearn model that
            was tried in TPOT. Must be called after the fit method. It also
            populates the instance variable "models" to a dictionary of all
            models tried and all their run history.

            Args:
                return_scores (bool): whether to return the score of the top
                    (selected) models (True) or their full parameters.

            Returns (dict):
                Top performing run for each sklearn model
            """
            self.greater_score_is_better = is_greater_better(self.scoring_function)
            model_names = list(set([key.split('(')[0] for key in
                                          self.evaluated_individuals_.keys()]))
            models = OrderedDict({model: [] for model in model_names})
            for k in self.evaluated_individuals_:
                models[k.split('(')[0]].append(self.evaluated_individuals_[k])
            for model_name in model_names:
                models[model_name]=sorted(models[model_name],
                                          key=lambda x: x['internal_cv_score'],
                                          reverse=self.greater_score_is_better)
                self.models = models
                top_models = {model: models[model][0] for model in models}
                self.top_models = OrderedDict(
                    sorted(top_models.items(),
                           key=lambda x:x[1]['internal_cv_score'],
                           reverse=self.greater_score_is_better))
                scores = {model: self.top_models[model]['internal_cv_score']\
                          for model in self.top_models}
                self.top_models_scores = OrderedDict(sorted(
                    scores.items(), key=lambda x: x[1],
                    reverse=self.greater_score_is_better))
            if return_scores:
                return self.top_models_scores
            else:
                return self.top_models


        def fit(self, features, target, **kwargs):
            """
            Wrapper function that is identical to the fit method of
            TPOTClassifier or TPOTRegressor. The purpose is to store the
            feature and target and use it in other methods of TpotAutoml

            Args:
                please see the documentation of TPOT for a full description.

            Returns:
                please see the documentation of TPOT for a full description.
            """
            self.features = features
            self.target = target
            super(tpot_class, self).fit(features, target, **kwargs)
    return TpotWrapper(**kwargs)


def is_greater_better(scoring_function):
    """
    Determines whether scoring_function being greater is more favorable/better.

    Args:
        scoring_function (str): the name of the scoring function supported by
            TPOT and sklearn. Please see below for more information.

    Returns (bool):
    """
    if scoring_function in [
        'accuracy', 'adjusted_rand_score', 'average_precision',
        'balanced_accuracy','f1', 'f1_macro', 'f1_micro', 'f1_samples',
        'f1_weighted', 'precision', 'precision_macro', 'precision_micro',
        'precision_samples','precision_weighted', 'recall',
        'recall_macro', 'recall_micro','recall_samples',
        'recall_weighted', 'roc_auc'] + \
            ['r2', 'neg_median_absolute_error', 'neg_mean_absolute_error',
            'neg_mean_squared_error']:
        return True
    elif scoring_function in ['median_absolute_error',
                              'mean_absolute_error',
                              'mean_squared_error']:
        return False
    else:
        warnings.warn('The scoring_function: "{}" not found; continuing assuming'
                      ' greater score is better'.format(scoring_function))
        return True


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




if __name__ == '__main__':
    # inputs
    target = 'deal_probability'
    RS = 23
    LIMIT = 100000
    TIMEOUT_MINS = 3
    SCORING = 'neg_mean_squared_error' # could be set to neg_mean_squared_error and then take np.sqrt(abs()) of that to compare with the leaderboard

    # training
    train = pd.read_csv('data/train.csv')[:LIMIT].dropna()
    train = pd.get_dummies(train[['price', 'category_name', 'user_type', target]])
    print('mean of the deal_probability in the selected subset')
    print(train[target].mean())
    tss = TimeSeriesSplit(n_splits=4)
    X = (train.drop(target, axis=1)).values
    y = train[target].values
    # tss.split(X) is a generator object used for cross-validation
    train_index, test_index = list(tss.split(X))[-1]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    tpot = TpotAutoml(mode='regression',
                      max_time_mins=TIMEOUT_MINS,
                      scoring=SCORING,
                      random_state=RS,
                      n_jobs=-1,
                      verbosity=2,
                      cv=TimeSeriesSplit(n_splits=3))

    tpot.fit(X_train, y_train)
    top_scores = tpot.get_top_models(return_scores=True)
    print('\ntop cv scores:')
    print(top_scores)
    print('\ntop models')
    print(tpot.top_models)
    print('\nthe best test score:')
    test_score = tpot.score(X_test, y_test)
    print(test_score)
