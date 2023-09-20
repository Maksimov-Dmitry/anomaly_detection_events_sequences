import numpy as np
import scipy.stats as stats
import pandas as pd
from collections import OrderedDict
from statsmodels.distributions.empirical_distribution import ECDF
import pickle
import os
import argparse


def detect_rand(test_set):
    """
    Creates random scores for test sequences.

    Args:
        test_set (list of dict): The test sequences data.

    Returns:
        pd.DataFrame: A DataFrame containing the results with random scores.
    """
    result = []
    for seq_i, seq in enumerate(test_set):
        vt = seq['time_test']
        vlabel = seq['label_test']
        score = np.random.rand(len(vt) - 1)
        result.append(pd.DataFrame(OrderedDict({
            'seq': seq_i,
            'time': vt[1:],
            'score_omiss': score,
            'score_commiss': -score,
            'label': vlabel,
        })))
    result = pd.concat(result)
    return result


def fit_len(train_set):
    """
    Fits the length distribution of events in the training set.

    Args:
        train_set (list of dict): The training sequences data.

    Returns:
        ECDF: Empirical cumulative distribution function for lengths.
    """
    lens = []
    for seq in train_set:
        vt = seq['time_target']
        if len(vt) > 0:
            t_min = seq['start']
            if vt[0] != t_min:
                vt = np.insert(vt, 0, t_min)
            lens.extend(np.diff(vt))
    return ECDF(lens)


def detect_len(test_set, ecdf):
    """
    Detects scores based on lengths for test sequences.

    Args:
        test_set (list of dict): The test sequences data.
        ecdf (ECDF): Empirical cumulative distribution function for lengths.

    Returns:
        pd.DataFrame: A DataFrame containing the scores based on lengths.
    """
    result = []
    for seq_i, seq in enumerate(test_set):
        vt = seq['time_test']
        vlabel = seq['label_test']
        dt = np.diff(vt)
        score_omiss = dt
        p_left = ecdf(dt)
        p_right = 1 - p_left
        score_commiss = -np.minimum(p_left, p_right)
        result.append(pd.DataFrame(OrderedDict({
            'seq': seq_i,
            'time': vt[1:],
            'score_omiss': score_omiss,
            'score_commiss': score_commiss,
            'label': vlabel,
        })))
    result = pd.concat(result)
    return result


def detect_model_true(test_set):
    """
    Detects true scores based on the true model for test sequences.

    Args:
        test_set (list of dict): The test sequences data.

    Returns:
        pd.DataFrame: A DataFrame containing the true scores.
    """
    result = []
    for seq_i, seq in enumerate(test_set):
        vt = seq['time_test']
        vlabel = seq['label_test']
        t_x = seq['t_x']
        lambda_x = seq['lambda_x']
        lambda_ = np.interp(vt, t_x, lambda_x)
        Lambda = []
        for i in range(len(vt) - 1):
            idx = (t_x > vt[i]) & (t_x < vt[i + 1])
            y = np.concatenate(([lambda_[i]], lambda_x[idx], [lambda_[i + 1]]))
            x = np.concatenate(([vt[i]], t_x[idx], [vt[i + 1]]))
            Lambda.append(np.trapz(y, x))
        Lambda = np.array(Lambda)
        result.append(pd.DataFrame(OrderedDict({
            'seq': seq_i,
            'time': vt[1:],
            'score_omiss': Lambda,
            'score_commiss': -lambda_[1:],
            'label': vlabel,
        })))
    result = pd.concat(result)
    return result


def detect_model_pois(test_set, param):
    """
    Detects scores based on the Poisson model for test sequences.

    Args:
        test_set (list of dict): The test sequences data.
        param (array): Parameters for the Poisson model.

    Returns:
        pd.DataFrame: A DataFrame containing the scores based on the Poisson model.
    """
    result = []
    for seq_i, seq in enumerate(test_set):
        vt = seq['time_test']
        vlabel = seq['label_test']
        t_max = seq['stop']
        vt_z = np.append(seq['time_context'], t_max)
        vz = seq['mark_context']
        id = seq['id']
        n = len(vt)
        score = np.zeros((n - 1, 2))
        for i in range(n - 1):
            # find z's inbetween
            t_beg = vt[i]
            t_end = vt[i + 1]
            j_beg = np.nonzero(vt_z > t_beg)[0][0] - 1
            j_end = np.nonzero(vt_z >= t_end)[0][0]
            Lambda = 0
            for j in range(j_beg, j_end, 1):
                t_span = np.min([vt_z[j + 1], t_end]) - np.max([vt_z[j], t_beg])
                Lambda = Lambda + param[id][vz[j]] * t_span
            lambda_ = param[id][vz[j_end - 1]]
            score[i, 0] = Lambda
            score[i, 1] = -lambda_
        result.append(pd.DataFrame(OrderedDict({
            'seq': seq_i,
            'time': vt[1:],
            'score_omiss': score[:, 0],
            'score_commiss': score[:, 1],
            'label': vlabel,
        })))
    result = pd.concat(result)
    return result


def fit_model_pois(train_set, n_mode):
    """
    Fits the Poisson model to the training sequences data.

    Args:
        train_set (list of dict): The training sequences data.
        n_mode (int): The number of modes.

    Returns:
        array: Parameters for the Poisson model.
    """
    data = [None] * n_mode
    for seq in train_set:
        vt_event = seq['time_target']
        t_max = seq['stop']
        vt_z = np.append(seq['time_context'], t_max)
        vz = seq['mark_context']
        for i in range(len(vz)):
            vt = vt_event[(vt_event > vt_z[i]) & (vt_event <= vt_z[i + 1])]
            new_data = np.diff(vt)
            if data[vz[i]] is None:
                data[vz[i]] = new_data
            else:
                data[vz[i]] = np.append(data[vz[i]], new_data)
    param = np.zeros(n_mode)
    for k in range(n_mode):
        param[k] = 1 / data[k].mean()
    return param


def detect_model_gam(test_set, param):
    """
    Detects scores based on the Gamma model for test sequences.

    Args:
        test_set (list of dict): The test sequences data.
        param (array): Parameters for the Gamma model.

    Returns:
        pd.DataFrame: A DataFrame containing the scores based on the Gamma model.
    """
    result = []
    for seq_i, seq in enumerate(test_set):
        vt_event = seq['time_target']
        vt = seq['time_test']
        vlabel = seq['label_test']
        t_max = seq['stop']
        vt_z = np.append(seq['time_context'], t_max)
        vz = seq['mark_context']
        n = len(vt)
        Lambda = np.zeros(n - 1)
        score = np.zeros((n - 1, 2))
        vt_ref = np.concatenate(([0], vt_event))
        for i in range(n - 1):
            # find z's inbetween
            t_beg = vt[i]
            t_end = vt[i + 1]
            j_beg = np.nonzero(vt_z > t_beg)[0][0] - 1
            j_end = np.nonzero(vt_z >= t_end)[0][0]
            t_ref = vt_ref[vt_ref <= t_beg][-1]
            Lambda[i] = 0
            for j in range(j_beg, j_end, 1):
                t_s_beg = np.max([vt_z[j], t_beg]) - t_ref
                t_s_end = np.min([vt_z[j + 1], t_end]) - t_ref
                Lambda[i] = Lambda[i] \
                    - stats.gamma.logsf(t_s_end, param[vz[j], 0], scale=1 / param[vz[j], 1]) \
                    + stats.gamma.logsf(t_s_beg, param[vz[j], 0], scale=1 / param[vz[j], 1])
            a = param[vz[j_end - 1], 0]
            b = 1 / param[vz[j_end - 1], 1]
            lambda_ = stats.gamma.pdf(t_end - t_beg, a, scale=b) / stats.gamma.sf(t_end - t_beg, a, scale=b)
            score[i, 0] = Lambda[i]
            score[i, 1] = -lambda_
            assert (not np.any(np.isnan(score)))
        result.append(pd.DataFrame(OrderedDict({
            'seq': seq_i,
            'time': vt[1:],
            'score_omiss': score[:, 0],
            'score_commiss': score[:, 1],
            'label': vlabel,
        })))
    result = pd.concat(result)
    return result


def fit_model_gam(train_set, n_mode):
    """
    Fits the Gamma model to the training sequences data.

    Args:
        train_set (list of dict): The training sequences data.
        n_mode (int): The number of modes.

    Returns:
        array: Parameters for the Gamma model.
    """
    data = [None] * n_mode
    for seq in train_set:
        vt_event = seq['time_target']
        t_max = seq['stop']
        vt_z = np.append(seq['time_context'], t_max)
        vz = seq['mark_context']
        for i in range(len(vz)):
            vt = vt_event[(vt_event > vt_z[i]) & (vt_event <= vt_z[i + 1])]
            new_data = np.diff(vt)
            if data[vz[i]] is None:
                data[vz[i]] = new_data
            else:
                data[vz[i]] = np.append(data[vz[i]], new_data)
    K = len(data)
    param = np.zeros((K, 2))
    for k in range(K):
        param[k, 0], _, param[k, 1] = stats.gamma.fit(data[k], floc=0)
    param[:, 1] = 1 / param[:, 1]
    return param


def detect(name, method, test_set, result_path, p):
    """
    Detects scores using the given method and saves to a CSV file.

    Args:
        name (str): Name for the result.
        method (function): Method to apply for detection.
        test_set (list of dict): The test sequences data.
        result_path (str): Path to save the results.
        p (float): Parameter 'p' for the method.
    """
    for outlier in outliers:
        np.random.seed(0)
        result = method(test_set[outlier][p])
        result.to_csv(f'{result_path}/{outlier}/{name}_{p}.csv')


def detect_with_param(method, param):
    """
    Creates a lambda function for detection with given method and parameters.

    Args:
        method (function): Method to apply for detection.
        param (array): Parameters for the method.

    Returns:
        function: Lambda function for detection.
    """
    return lambda x: method(x, param)


if __name__ == '__main__':
    datasets = ['pois']
    outliers = ['commiss', 'omiss']
    parser = argparse.ArgumentParser(description='Your script description.')
    parser.add_argument('--p', type=str, default="0.1", help='Value for p. Default is "0.1".')
    args = parser.parse_args()

    for dataset in datasets:
        folder = f'data/raw/{dataset}'
        result_path = f'results/{dataset}'
        with open(f'{folder}/train.pkl', 'rb') as f:
            train_set = pickle.load(f)

        test_set = {}
        for outlier in outliers:
            test_set[outlier] = {}
            os.makedirs(f'{result_path}/{outlier}', exist_ok=True)
            with open(f'{folder}/test_{outlier}_{args.p}.pkl', 'rb') as f:
                test_set[outlier][args.p] = pickle.load(f)

        detect('rand', detect_rand, test_set, result_path, args.p)

        param = fit_len(train_set)
        detect('len', detect_with_param(detect_len, param), test_set, result_path, args.p)

        if dataset == 'pois':
            with open(f'{folder}/param.pkl', 'rb') as f:
                param = pickle.load(f)
            detect('true', detect_with_param(detect_model_pois, param), test_set, result_path, args.p)
        elif dataset == 'gam':
            with open(f'{folder}/param.pkl', 'rb') as f:
                param = pickle.load(f)
            detect('true', detect_with_param(detect_model_gam, param), test_set, result_path, args.p)
        elif test_set.get('lambda_x') is not None:
            detect('true', detect_model_true, test_set, result_path, args.p)
