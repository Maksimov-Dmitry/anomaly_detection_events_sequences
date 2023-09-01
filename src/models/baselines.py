import numpy as np
import scipy.stats as stats
import pandas as pd
from collections import OrderedDict
from statsmodels.distributions.empirical_distribution import ECDF
import pickle
import os


def detect_rand(test_set):
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
    lens = []
    for seq in train_set:
        vt = seq['time_target']
        t_min = seq['start']
        if vt[0] != t_min:
            vt = np.insert(vt, 0, t_min)
        lens.extend(np.diff(vt))
    return ECDF(lens)


def detect_len(test_set, ecdf):
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


def detect(name, method, test_set, result_path):
    for outlier in outliers:
        for p in ps:
            np.random.seed(0)
            result = method(test_set[outlier][p])
            result.to_csv(f'{result_path}/{outlier}/{name}_{p}.csv')


def detect_with_param(method, param):
    return lambda x: method(x, param)


if __name__ == '__main__':
    datasets = ['pois']
    outliers = ['commiss', 'omiss']
    ps = ["0.1"]

    for dataset in datasets:
        folder = f'data/raw/{dataset}'
        result_path = f'results/{dataset}'
        with open(f'{folder}/train.pkl', 'rb') as f:
            train_set = pickle.load(f)

        test_set = {}
        for outlier in outliers:
            test_set[outlier] = {}
            os.makedirs(f'{result_path}/{outlier}', exist_ok=True)
            for p in ps:
                with open(f'{folder}/test_{outlier}_{p}.pkl', 'rb') as f:
                    test_set[outlier][p] = pickle.load(f)

        detect('rand', detect_rand, test_set, result_path)

        param = fit_len(train_set)
        detect('len', detect_with_param(detect_len, param), test_set, result_path)

        K = 2
        if dataset == 'pois':
            # param = fit_model_pois(train_set, K)
            # detect('model', detect_with_param(detect_model_pois, param), test_set, result_path)
            with open(f'{folder}/param.pkl', 'rb') as f:
                param = pickle.load(f)
            detect('true', detect_with_param(detect_model_pois, param), test_set, result_path)
        elif dataset == 'gam':
            # param = fit_model_gam(train_set, K)
            # detect('model', detect_with_param(detect_model_gam, param), test_set, result_path)
            with open(f'{folder}/param.pkl', 'rb') as f:
                param = pickle.load(f)
            detect('true', detect_with_param(detect_model_gam, param), test_set, result_path)
        elif test_set.get('lambda_x') is not None:
            detect('true', detect_model_true, test_set, result_path)
