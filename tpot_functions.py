from collections import OrderedDict
import numpy as np
import pandas as pd
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import TimeSeriesSplit

pd.options.mode.chained_assignment = None

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
    WARNING UNPUBLISHED WORK! Please don't distribute

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
        raise ValueError('Unsupported scoring_function: "{}"'.format(
            scoring_function))






